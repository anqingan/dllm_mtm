"""
reference: https://github.com/ML-GSAI/LLaDA/blob/main/generate.py
"""

import math
import warnings
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from dllm.core.samplers.base import BaseSampler, BaseSamplerConfig, BaseSamplerOutput
from dllm.core.samplers.utils import (
    add_gumbel_noise,
    compute_confidence_scores,
    compute_kl_divergence,
    get_num_transfer_tokens,
    oracle_block_enumerate,
    select_transfer_positions,
)
from dllm.duel.config import DuelMTMConfig
from dllm.duel.diagnostics import save_diagnostics
from dllm.duel.intra_block_mtm import IntraBlockDuelMTM


def _normalise_duel_mtm_config(
    config: DuelMTMConfig | dict | None,
) -> DuelMTMConfig:
    if config is None:
        return DuelMTMConfig(enabled=False)
    if isinstance(config, DuelMTMConfig):
        return config
    if isinstance(config, dict):
        inter_keys = [key for key in config if "inter" in key.lower()]
        if inter_keys:
            raise ValueError(
                "Inter-block MTM is intentionally disabled in DUEL-guided "
                "Intra-Block MTM."
            )
        return DuelMTMConfig(**config)
    raise TypeError("duel_mtm must be a DuelMTMConfig, dict, or None")


def _make_duel_generator(cfg: DuelMTMConfig, device: torch.device) -> torch.Generator:
    try:
        generator = torch.Generator(device=device)
    except RuntimeError:
        generator = torch.Generator()
    return generator.manual_seed(cfg.seed)


@dataclass
class MDLMSamplerConfig(BaseSamplerConfig):
    max_new_tokens: int = 128
    max_length: int = (
        None  # There's no explicit length_limit except for the tokenizer/model context
    )
    block_size: int = 128
    steps: int = 128
    temperature: float = 0.0
    remasking: str = "low_confidence"
    stochastic_transfer: bool = False
    cfg_scale: float = 0.0
    cfg_keep_tokens: list[int] | None = None
    suppress_tokens: list[int] | None = None
    begin_suppress_tokens: list[int] | None = None
    right_shift_logits: bool = False
    # --- New strategy parameters ---
    threshold: float | None = None  # confidence threshold for confidence_threshold / klass
    kl_threshold: float | None = None  # nu for KL divergence ceiling in klass
    oracle_max_positions: int = 5  # max masked positions for oracle enumeration
    duel_mtm: DuelMTMConfig | dict | None = None


@dataclass
class MDLMSampler(BaseSampler):
    @torch.no_grad()
    def sample(
        self,
        inputs: list[torch.Tensor | list],
        config: MDLMSamplerConfig | None = None,
        **kwargs,
    ) -> BaseSamplerOutput | torch.Tensor:
        """
        Generate text using masked diffusion language modeling.

        Iteratively unmasks tokens over multiple diffusion steps, starting from
        fully masked sequences appended to the input prompts.

        Args:
            inputs: List of input prompts (token tensors or lists of token IDs).
            config: Sampler configuration, or None to use defaults.
            **kwargs: Override specific config parameters.

        Returns:
            BaseSamplerOutput with generated sequences, or raw tensor if return_dict=False.
        """
        if config is None:
            config = MDLMSamplerConfig()

        # ----- pull args from config, allow kwargs to override -----
        steps = kwargs.get("steps", config.steps)
        max_new_tokens = kwargs.get("max_new_tokens", config.max_new_tokens)
        max_length = kwargs.get("max_length", config.max_length)
        block_size = kwargs.get("block_size", config.block_size)
        temperature = kwargs.get("temperature", config.temperature)
        cfg_scale = kwargs.get("cfg_scale", config.cfg_scale)
        cfg_keep_tokens = kwargs.get("cfg_keep_tokens", config.cfg_keep_tokens)
        remasking = kwargs.get("remasking", config.remasking)
        suppress_tokens = kwargs.get("suppress_tokens", config.suppress_tokens)
        stochastic_transfer = kwargs.get(
            "stochastic_transfer", config.stochastic_transfer
        )
        return_dict = kwargs.get("return_dict", config.return_dict)
        right_shift_logits = kwargs.get("right_shift_logits", config.right_shift_logits)
        begin_suppress_tokens = kwargs.get(
            "begin_suppress_tokens", config.begin_suppress_tokens
        )
        threshold = kwargs.get("threshold", config.threshold)
        kl_threshold = kwargs.get("kl_threshold", config.kl_threshold)
        oracle_max_positions = kwargs.get(
            "oracle_max_positions", config.oracle_max_positions
        )
        duel_mtm_cfg = _normalise_duel_mtm_config(
            kwargs.get("duel_mtm", config.duel_mtm)
        )

        assert 1 <= block_size
        assert 1 <= steps
        mask_id = self.tokenizer.mask_token_id
        bos_id = self.tokenizer.bos_token_id
        eos_id = self.tokenizer.eos_token_id

        # ----- Shape bookkeeping: per-sample prompt lengths and final canvas width -----
        # If right_shift_logits is true and a sequence has length 0, replace that sequence with [bos].
        if right_shift_logits:
            inputs = [
                [bos_id] if isinstance(p, list) and len(p) == 0 else p for p in inputs
            ]

        if isinstance(inputs[0], list):
            inputs = [
                torch.as_tensor(p, dtype=torch.long, device=self.model.device)
                for p in inputs
            ]
        prompt_lens = [p.shape[0] for p in inputs]

        if max_new_tokens:
            max_length = max_new_tokens + max(prompt_lens)
        else:
            max_new_tokens = max_length - max(prompt_lens)

        B = len(inputs)
        T = max_length

        # ----- Initialize canvas with EOS, copy inputs, and append mask tail -----
        x = torch.full((B, T), eos_id, dtype=torch.long, device=self.model.device)
        for i, p in enumerate(inputs):
            x[i, : prompt_lens[i]] = p  # keep original prompt tokens
            x[i, prompt_lens[i] : prompt_lens[i] + max_new_tokens] = (
                mask_id  # append `max_new_tokens` masks to be generated
            )
        attention_mask = torch.zeros((B, T), dtype=torch.long, device=self.model.device)
        for i, pl in enumerate(prompt_lens):
            valid_end = min(pl + max_new_tokens, T)
            attention_mask[i, :valid_end] = 1

        # Tokens that were *given* at the start (non-mask, non-EOS).
        # These will be masked in the unconditional forward pass for CFG.
        # Tokens from `cfg_keep_tokens` should *not* be treated as "given" for CFG
        unmasked_index = (x != mask_id) & attention_mask.bool()
        if not (cfg_keep_tokens is None or len(cfg_keep_tokens) == 0):
            keep_mask = torch.isin(
                x, torch.as_tensor(cfg_keep_tokens, device=self.model.device)
            )
            unmasked_index = unmasked_index & ~keep_mask

        # ----- Block scheduling over the appended mask tail -----
        num_blocks = math.ceil(max_new_tokens / block_size)
        steps = math.ceil(steps / num_blocks)  # per-block step budget
        histories = [x.clone()] if return_dict else None

        # State for KLASS: stores previous step's probability distribution
        prev_probs = None
        duel_diagnostics = []
        duel_generator = (
            _make_duel_generator(duel_mtm_cfg, self.model.device)
            if duel_mtm_cfg.enabled
            else None
        )
        duel_refiner = (
            IntraBlockDuelMTM(self.model, duel_mtm_cfg, mask_id, self.tokenizer)
            if duel_mtm_cfg.enabled
            else None
        )

        for b in range(num_blocks):
            # Build a per-sample mask *within this block* (aligned to each prompt's tail)
            block_mask_index = torch.zeros(
                (B, block_size), dtype=torch.bool, device=x.device
            )

            for j in range(B):
                start = prompt_lens[j] + b * block_size
                end = min(start + block_size, prompt_lens[j] + max_new_tokens, T)
                if start < end:
                    width = end - start
                    block_mask_index[j, :width] = (
                        x[j, start:end] == mask_id
                    )  # which positions in this block are still masked

            # Decide how many tokens to reveal per step in this block
            num_transfer_tokens = get_num_transfer_tokens(
                mask_index=block_mask_index,
                steps=steps,
                scheduler=self.scheduler,
                stochastic=stochastic_transfer,
            )

            # Some steps may be skipped if there are no transfers
            effective_steps = num_transfer_tokens.size(1)

            # Reset KLASS state at the start of each new block
            if remasking == "klass":
                prev_probs = None

            # ----- Oracle: special handling for the entire block -----
            if remasking == "oracle" and B == 1:
                block_start = prompt_lens[0] + b * block_size
                actual_block_len = min(
                    block_size,
                    max_new_tokens - b * block_size,
                )
                if actual_block_len <= oracle_max_positions:
                    x = oracle_block_enumerate(
                        x=x,
                        block_start=block_start,
                        block_len=actual_block_len,
                        model=self.model,
                        temperature=temperature,
                        mask_id=mask_id,
                        max_positions=oracle_max_positions,
                        attention_mask=attention_mask,
                    )
                    if histories is not None:
                        histories.append(x.clone())
                    continue
                # else fall through to standard greedy per-step logic
                warnings.warn(
                    f"Oracle: block has {actual_block_len} masked positions, "
                    f"exceeds oracle_max_positions={oracle_max_positions}. "
                    f"Falling back to greedy_confidence.",
                    stacklevel=2,
                )

            # ----- Iterative reveal inside the current block -----
            for i in range(effective_steps):
                mask_index = x == mask_id  # current global mask map

                # Optional CFG: second forward where original prompt tokens are masked out
                if cfg_scale > 0.0:
                    un_x = x.clone()
                    un_x[unmasked_index] = mask_id
                    x_ = torch.cat([x, un_x], dim=0)
                    logits = self.model(
                        x_, attention_mask=attention_mask.repeat(2, 1)
                    ).logits
                    logits, un_logits = torch.chunk(logits, 2, dim=0)
                    logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
                else:
                    logits = self.model(
                        x, attention_mask=attention_mask
                    ).logits  # Use attention mask here

                if suppress_tokens is not None and len(suppress_tokens) > 0:
                    for token_id in suppress_tokens:
                        logits[:, :, token_id] = -torch.inf

                if right_shift_logits:
                    logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)

                # Argmax decoding with optional Gumbel-Max noise for exploration
                logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
                x0 = torch.argmax(
                    logits_with_noise, dim=-1
                )  # [B, T] predicted token ids

                if begin_suppress_tokens is not None and len(begin_suppress_tokens) > 0:
                    for token_id in begin_suppress_tokens:
                        logits[:, :, token_id] = -torch.inf

                # --- Compute confidence scores via unified helper ---
                x0_p, current_probs = compute_confidence_scores(logits, x0, remasking)

                # Restrict selection window to the *current block's* tail region
                for j in range(B):
                    x0_p[j, prompt_lens[j] + (b + 1) * block_size :] = -np.inf

                # Only allow updates at currently masked positions; keep others fixed
                x0 = torch.where(mask_index, x0, x)
                confidence = torch.where(
                    mask_index, x0_p, -np.inf
                )  # consider masked positions only

                # --- KL divergence for KLASS ---
                kl_div = None
                if remasking == "klass" and prev_probs is not None and current_probs is not None:
                    kl_div = compute_kl_divergence(prev_probs, current_probs)

                # --- Select positions to unmask via unified helper ---
                transfer_index = select_transfer_positions(
                    confidence=confidence,
                    mask_index=mask_index,
                    num_transfer_tokens=num_transfer_tokens[:, i],
                    remasking=remasking,
                    threshold=threshold,
                    kl_threshold=kl_threshold,
                    kl_divergence=kl_div,
                )

                # Commit chosen predictions into the canvas
                x[transfer_index] = x0[transfer_index]

                # Update KLASS state
                if remasking == "klass" and current_probs is not None:
                    prev_probs = current_probs

                if histories is not None:
                    histories.append(x.clone())

            if duel_refiner is not None:
                for j in range(B):
                    block_start = prompt_lens[j] + b * block_size
                    block_end = min(
                        block_start + block_size,
                        prompt_lens[j] + max_new_tokens,
                        T,
                    )
                    if block_start >= block_end:
                        continue
                    for _ in range(duel_mtm_cfg.num_mtm_steps_per_block):
                        refined, diag = duel_refiner.step(
                            current_state=x[j : j + 1],
                            block_start=block_start,
                            block_end=block_end,
                            block_index=b,
                            attention_mask=attention_mask[j : j + 1],
                            generator=duel_generator,
                        )
                        x[j : j + 1] = refined
                        duel_diagnostics.append(diag)
                        if histories is not None:
                            histories.append(x.clone())

        # ----- Output format -----
        if not return_dict:
            return x
        else:
            if duel_mtm_cfg.enabled and duel_mtm_cfg.save_diagnostics:
                save_diagnostics(duel_diagnostics, duel_mtm_cfg.diagnostics_path)
            output = BaseSamplerOutput(sequences=x, histories=histories)
            if duel_mtm_cfg.enabled:
                output.duel_diagnostics = duel_diagnostics
            return output

    @torch.no_grad()
    def infill(
        self, inputs: list[torch.Tensor | list], config, **kwargs
    ) -> BaseSamplerOutput | torch.Tensor:
        """
        Fill in-place the <|mdm_mask|> tokens contained in `inputs`.
        The whole (padded) sequence is split into block windows of length
        `block_size`; within each window we progressively "unmask" positions
        according to the scheduler and chosen remasking strategy.

        Notes:
        - Right padding uses EOS.
        - CFG masks out *originally known* (non-mask, non-EOS) tokens in the
        unconditional branch, identical to `generate`.
        - Only masked positions are ever updated; non-mask tokens are left intact.
        """
        # ----- pull args from config, allow kwargs to override -----
        steps = kwargs.get("steps", config.steps)
        block_size = kwargs.get("block_size", config.block_size)
        temperature = kwargs.get("temperature", config.temperature)
        cfg_scale = kwargs.get("cfg_scale", config.cfg_scale)
        cfg_keep_tokens = kwargs.get("cfg_keep_tokens", config.cfg_keep_tokens)
        remasking = kwargs.get("remasking", config.remasking)
        suppress_tokens = kwargs.get("suppress_tokens", config.suppress_tokens)
        stochastic_transfer = kwargs.get(
            "stochastic_transfer", config.stochastic_transfer
        )
        return_dict = kwargs.get("return_dict", config.return_dict)
        right_shift_logits = kwargs.get("right_shift_logits", config.right_shift_logits)
        begin_suppress_tokens = kwargs.get(
            "begin_suppress_tokens", config.begin_suppress_tokens
        )
        threshold = kwargs.get("threshold", config.threshold)
        kl_threshold = kwargs.get("kl_threshold", config.kl_threshold)
        oracle_max_positions = kwargs.get(
            "oracle_max_positions", config.oracle_max_positions
        )
        duel_mtm_cfg = _normalise_duel_mtm_config(
            kwargs.get("duel_mtm", getattr(config, "duel_mtm", None))
        )

        mask_id = self.tokenizer.mask_token_id
        bos_id = self.tokenizer.bos_token_id
        eos_id = self.tokenizer.eos_token_id

        # ----- Build canvas: right-pad with EOS to the max length in the batch -----
        # If right_shift_logits is true and a sequence has length 0, replace that sequence with [bos].
        if right_shift_logits:
            inputs = [
                [bos_id] if isinstance(p, list) and len(p) == 0 else p for p in inputs
            ]

        if isinstance(inputs[0], list):
            inputs = [
                torch.as_tensor(p, dtype=torch.long, device=self.model.device)
                for p in inputs
            ]

        B = len(inputs)
        seq_lens = [t.shape[0] for t in inputs]
        T = max(seq_lens)

        # Default to a single block spanning the whole sequence
        if block_size is None:
            block_size = T

        assert 1 <= block_size
        assert 1 <= steps

        x = torch.full((B, T), eos_id, dtype=torch.long, device=self.model.device)
        for i, t in enumerate(inputs):
            x[i, : seq_lens[i]] = t

        attention_mask = torch.zeros((B, T), dtype=torch.long, device=self.model.device)
        for i, L in enumerate(seq_lens):
            if L > 0:
                attention_mask[i, :L] = 1

        # Tokens that were *given* at the start (non-mask, non-EOS).
        # These will be masked in the unconditional forward pass for CFG.
        # Tokens from `cfg_keep_tokens` should *not* be treated as "given" for CFG
        unmasked_index = (x != mask_id) & attention_mask.bool()
        if not (cfg_keep_tokens is None or len(cfg_keep_tokens) == 0):
            keep_mask = torch.isin(
                x, torch.as_tensor(cfg_keep_tokens, device=self.model.device)
            )
            unmasked_index = unmasked_index & ~keep_mask

        # ----- Blockwise schedule over the *entire* (padded) sequence -----
        num_blocks = math.ceil(T / block_size)
        steps_per_block = math.ceil(steps / num_blocks)
        histories = [x.clone()] if return_dict else None

        # State for KLASS
        prev_probs = None
        duel_diagnostics = []
        duel_generator = (
            _make_duel_generator(duel_mtm_cfg, self.model.device)
            if duel_mtm_cfg.enabled
            else None
        )
        duel_refiner = (
            IntraBlockDuelMTM(self.model, duel_mtm_cfg, mask_id, self.tokenizer)
            if duel_mtm_cfg.enabled
            else None
        )

        for b in range(num_blocks):
            start = b * block_size
            stop = min(start + block_size, T)

            # Per-sample view of which positions in this block are masks
            block_mask_index = torch.zeros(
                (B, block_size), dtype=torch.bool, device=self.model.device
            )
            widths = []
            for j in range(B):
                # Width limited by sample's true length and sequence end
                width = max(0, min(seq_lens[j], stop) - start)
                widths.append(width)
                if width > 0:
                    block_mask_index[j, :width] = x[j, start : start + width] == mask_id

            # Decide how many tokens to reveal at each step in this block
            num_transfer_tokens = get_num_transfer_tokens(
                mask_index=block_mask_index,
                steps=steps_per_block,
                scheduler=self.scheduler,
                stochastic=stochastic_transfer,
            )

            # Some blocks may have no masks => effective_steps == 0
            effective_steps = num_transfer_tokens.size(1)

            # Reset KLASS state at the start of each new block
            if remasking == "klass":
                prev_probs = None

            for s in range(effective_steps):
                mask_index_full = x == mask_id

                # ----- Forward pass (+ optional CFG) -----
                if cfg_scale > 0.0:
                    un_x = x.clone()
                    un_x[unmasked_index] = mask_id
                    x_ = torch.cat([x, un_x], dim=0)
                    logits = self.model(
                        x_, attention_mask=attention_mask.repeat(2, 1)
                    ).logits
                    logits, un_logits = torch.chunk(logits, 2, dim=0)
                    logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
                else:
                    logits = self.model(
                        x, attention_mask=attention_mask
                    ).logits  # Use attention mask here

                if suppress_tokens is not None and len(suppress_tokens) > 0:
                    for token_id in suppress_tokens:
                        logits[:, :, token_id] = -torch.inf

                if right_shift_logits:
                    logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)

                # Greedy with optional Gumbel-Max noise
                logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
                x0 = torch.argmax(logits_with_noise, dim=-1)  # [B, T]

                if begin_suppress_tokens is not None and len(begin_suppress_tokens) > 0:
                    for token_id in begin_suppress_tokens:
                        logits[:, :, token_id] = -torch.inf

                # --- Compute confidence scores via unified helper ---
                x0_p, current_probs = compute_confidence_scores(logits, x0, remasking)

                # Restrict selection to the *current* block only
                for j in range(B):
                    end_j = start + widths[j]
                    # Outside current block => impossible to select
                    x0_p[j, :start] = -np.inf
                    x0_p[j, end_j:] = -np.inf

                # Only consider currently-masked positions as candidates
                x0 = torch.where(mask_index_full, x0, x)
                confidence = torch.where(mask_index_full, x0_p, -np.inf)

                # --- KL divergence for KLASS ---
                kl_div = None
                if remasking == "klass" and prev_probs is not None and current_probs is not None:
                    kl_div = compute_kl_divergence(prev_probs, current_probs)

                # --- Select positions to unmask via unified helper ---
                transfer_index = select_transfer_positions(
                    confidence=confidence,
                    mask_index=mask_index_full,
                    num_transfer_tokens=num_transfer_tokens[:, s],
                    remasking=remasking,
                    threshold=threshold,
                    kl_threshold=kl_threshold,
                    kl_divergence=kl_div,
                )

                # Commit selected predictions into the canvas
                x[transfer_index] = x0[transfer_index]

                # Update KLASS state
                if remasking == "klass" and current_probs is not None:
                    prev_probs = current_probs

                if histories is not None:
                    histories.append(x.clone())

            if duel_refiner is not None:
                for j in range(B):
                    block_start = start
                    block_end = min(stop, seq_lens[j])
                    if block_start >= block_end:
                        continue
                    for _ in range(duel_mtm_cfg.num_mtm_steps_per_block):
                        refined, diag = duel_refiner.step(
                            current_state=x[j : j + 1],
                            block_start=block_start,
                            block_end=block_end,
                            block_index=b,
                            attention_mask=attention_mask[j : j + 1],
                            generator=duel_generator,
                        )
                        x[j : j + 1] = refined
                        duel_diagnostics.append(diag)
                        if histories is not None:
                            histories.append(x.clone())

        # ----- Output format -----
        if not return_dict:
            return x
        else:
            if duel_mtm_cfg.enabled and duel_mtm_cfg.save_diagnostics:
                save_diagnostics(duel_diagnostics, duel_mtm_cfg.diagnostics_path)
            output = BaseSamplerOutput(sequences=x, histories=histories)
            if duel_mtm_cfg.enabled:
                output.duel_diagnostics = duel_diagnostics
            return output
