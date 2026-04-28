"""
DUEL-guided Intra-Block MTM Decoding.

This module implements DUEL (Deterministic Unmasking Exact Likelihood) scoring
and Multiple-Try Metropolis (MTM) for inference-time local refinement in
masked diffusion language models.

Key concepts:
- DUEL scorer computes exact conditional likelihood under a deterministic
  unmasking rule F.
- DUEL-guided Intra-Block MTM uses DUEL conditional likelihood as an
  inference-time local refinement score.
- With beta > 1, the target is a tempered DUEL distribution.
- This method only revises the current block and does NOT perform inter-block
  suffix resampling.
"""
