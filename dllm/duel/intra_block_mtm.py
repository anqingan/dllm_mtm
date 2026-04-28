"""Intra-Block DUEL-MTM: Multiple-Try Metropolis using DUEL conditional
likelihood as the inference-time local refinement score.

This module only revises the *current* block.  It does NOT perform
inter-block suffix resampling.
"""

from __future__ import annotations

import math
import time

import torch

from dllm.duel.block_utils import make_block_mask, make_rollback_state
from dllm.duel.config import DuelMTMConfig
from dllm.duel.duel_sampler import compute_duel_proposal_logprob, duel_generate_region
from dllm.duel.duel_scorer import compute_duel_conditional_loglikelihood
from dllm.duel.types import DuelDiagnostics


class IntraBlockDuelMTM:
    """Intra-Block DUEL-guided Multiple-Try Metropolis."""

    def __init__(
        self,
        model: torch.nn.Module,
        config: DuelMTMConfig,
        mask_token_id: int,
        tokenizer=None,
    ):
        self.model = model
        self.config = config
        self.mask_token_id = mask_token_id
        self.tokenizer = tokenizer

    @torch.inference_mode()
    def step(
        self,
        current_state: torch.Tensor,  # [1, T]
        block_start: int,
        block_end: int,
        block_index: int = 0,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        generation_trace: dict | None = None,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, DuelDiagnostics]:
        """Execute one MTM step on the block [block_start, block_end).

        Returns (new_state, diagnostics).
        Positions outside [block_start, block_end) are never modified.
        """
        cfg = self.config
        proposal_top_k = cfg.proposal_top_k
        proposal_top_p = cfg.proposal_top_p
        T = current_state.shape[1]
        block_mask = make_block_mask(
            T, block_start, block_end, device=current_state.device
        )
        diag = DuelDiagnostics(
            block_index=block_index,
            block_start=block_start,
            block_end=block_end,
            method=cfg.method,
            unmask_rule=cfg.unmask_rule,
            beta=cfg.beta,
            K=cfg.K,
        )

        t0 = time.perf_counter()

        # 1. Rollback: re-mask part of the current block
        rollback_state, rollback_mask, preserved_mask = make_rollback_state(
            current_state=current_state,
            block_mask=block_mask,
            policy=cfg.rollback_policy,
            rollback_ratio=cfg.rollback_ratio,
            generation_trace=generation_trace,
            mask_token_id=self.mask_token_id,
            generator=generator,
        )
        diag.rollback_positions = torch.where(rollback_mask[0])[0].tolist()

        # Determine score_mask
        if cfg.score_scope == "rollback_region":
            score_mask = rollback_mask
        else:
            score_mask = block_mask & (current_state[0] != self.mask_token_id).unsqueeze(0)

        # Determine candidate_mask for DUEL: only the positions that were
        # actually re-masked and will be re-generated (rollback_mask positions
        # that are currently masked).
        candidate_mask = rollback_mask & (rollback_state == self.mask_token_id)

        # 2. Forward proposals: generate K candidates from rollback state
        forward_candidates = []
        forward_lls = []
        for _k in range(cfg.K):
            proposal = duel_generate_region(
                model=self.model,
                initial_state=rollback_state.clone(),
                generation_mask=rollback_mask.clone(),
                unmask_rule=cfg.unmask_rule,
                positions_per_step=cfg.positions_per_step,
                mask_token_id=self.mask_token_id,
                temperature=cfg.proposal_temperature,
                top_k=proposal_top_k,
                top_p=proposal_top_p,
                attention_mask=attention_mask,
                position_ids=position_ids,
                generator=generator,
            )
            candidate = proposal.sequence
            # Sanity: block-external tokens unchanged
            assert not (
                (candidate != current_state) & ~block_mask
            ).any(), "Candidate modified tokens outside the block"
            forward_candidates.append((candidate, proposal.log_q))

        # 3. Forward scoring — use candidate_mask = only positions that
        #    the generator was asked to fill (rollback_mask & originally masked).
        #    This matches exactly what duel_generate_region produces.
        scoring_candidate_mask = candidate_mask.clone()
        forward_log_qs = []
        for candidate, log_q in forward_candidates:
            ll, _ = compute_duel_conditional_loglikelihood(
                model=self.model,
                initial_state=rollback_state,
                target_tokens=candidate,
                score_mask=score_mask,
                candidate_mask=scoring_candidate_mask,
                unmask_rule=cfg.unmask_rule,
                positions_per_step=cfg.positions_per_step,
                mask_token_id=self.mask_token_id,
                target_temperature=cfg.target_temperature,
                attention_mask=attention_mask,
                position_ids=position_ids,
            )
            forward_lls.append(ll.item())
            forward_log_qs.append(log_q.item())

        diag.forward_loglikelihoods = forward_lls
        diag.forward_log_qs = forward_log_qs
        forward_log_weights = [
            cfg.alpha * ll - log_q for ll, log_q in zip(forward_lls, forward_log_qs)
        ]
        diag.forward_log_weights = forward_log_weights

        # 4. Select y_star
        if cfg.method == "duel_rerank":
            selected_index = int(torch.tensor(forward_lls).argmax().item())
            diag.selected_index = selected_index
            diag.accept_prob = 1.0
            diag.accepted = True
            diag.elapsed_time = time.perf_counter() - t0
            return forward_candidates[selected_index][0], diag

        # method == "duel_mtm"
        log_w_tensor = torch.tensor(forward_log_weights)
        probs = torch.softmax(log_w_tensor, dim=0)
        selected_index = int(torch.multinomial(probs, 1, generator=generator).item())
        diag.selected_index = selected_index
        selected_state = forward_candidates[selected_index][0]

        # 5. Backward reference set
        # Re-mask the rollback region of selected_state
        bwd_initial = selected_state.clone()
        bwd_initial[rollback_mask] = self.mask_token_id
        bwd_candidate_mask = rollback_mask.clone()

        backward_candidates = []
        backward_lls = []

        # Generate K-1 auxiliary backward candidates
        for _k in range(cfg.K - 1):
            bwd_proposal = duel_generate_region(
                model=self.model,
                initial_state=bwd_initial.clone(),
                generation_mask=rollback_mask.clone(),
                unmask_rule=cfg.unmask_rule,
                positions_per_step=cfg.positions_per_step,
                mask_token_id=self.mask_token_id,
                temperature=cfg.proposal_temperature,
                top_k=proposal_top_k,
                top_p=proposal_top_p,
                attention_mask=attention_mask,
                position_ids=position_ids,
                generator=generator,
            )
            backward_candidates.append((bwd_proposal.sequence, bwd_proposal.log_q))

        # Add original current_state as the K-th backward candidate
        current_log_q = compute_duel_proposal_logprob(
            model=self.model,
            initial_state=bwd_initial,
            target_state=current_state,
            generation_mask=bwd_candidate_mask,
            unmask_rule=cfg.unmask_rule,
            positions_per_step=cfg.positions_per_step,
            mask_token_id=self.mask_token_id,
            proposal_temperature=cfg.proposal_temperature,
            top_k=proposal_top_k,
            top_p=proposal_top_p,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )
        backward_candidates.append((current_state, current_log_q))

        backward_log_qs = []
        for bwd_cand, log_q in backward_candidates:
            ll, _ = compute_duel_conditional_loglikelihood(
                model=self.model,
                initial_state=bwd_initial,
                target_tokens=bwd_cand,
                score_mask=score_mask,
                candidate_mask=bwd_candidate_mask,
                unmask_rule=cfg.unmask_rule,
                positions_per_step=cfg.positions_per_step,
                mask_token_id=self.mask_token_id,
                target_temperature=cfg.target_temperature,
                attention_mask=attention_mask,
                position_ids=position_ids,
            )
            backward_lls.append(ll.item())
            backward_log_qs.append(log_q.item())

        diag.backward_loglikelihoods = backward_lls
        diag.backward_log_qs = backward_log_qs
        backward_log_weights = [
            cfg.alpha * ll - log_q for ll, log_q in zip(backward_lls, backward_log_qs)
        ]
        diag.backward_log_weights = backward_log_weights

        # 6. MTM accept/reject
        log_W_fwd = torch.logsumexp(torch.tensor(forward_log_weights), dim=0).item()
        log_W_bwd = torch.logsumexp(torch.tensor(backward_log_weights), dim=0).item()
        diag.log_W_fwd = log_W_fwd
        diag.log_W_bwd = log_W_bwd

        log_accept_ratio = log_W_fwd - log_W_bwd
        if log_accept_ratio >= 0:
            accept_prob = 1.0
        else:
            accept_prob = math.exp(log_accept_ratio)
        diag.accept_prob = accept_prob

        u = torch.rand(1, generator=generator).item()
        accepted = u < accept_prob
        diag.accepted = accepted
        diag.elapsed_time = time.perf_counter() - t0

        if accepted:
            return selected_state, diag
        else:
            return current_state, diag
