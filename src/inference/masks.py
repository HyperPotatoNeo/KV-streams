"""Inference-specific attention masks for KV Self-Compaction.

Two masks complement the training mask (build_4d_attention_mask):

build_decode_mask: For single-token autoregressive decode steps.
    Shape: (B, num_heads, 1, kv_len)
    All KV positions visible; compact_kv columns have learnable bias.

build_compaction_mask: For compaction-only forward pass (P queries).
    Shape: (B, num_heads, P, P_past + W + P)
    Compact_kv columns: bias. Text columns: fully visible to all P queries.
    Compaction columns: lower-triangular causal among themselves.

    CRITICAL: This differs from build_4d_attention_mask because in training,
    queries = [W text + P compaction], and the causal tril with diagonal=0
    means compaction row W sees columns 0..W. When processing ONLY P compaction
    queries, we must explicitly make all W text columns visible (not causal).
"""

from __future__ import annotations

import torch
import torch.nn as nn


def build_decode_mask(
    B: int,
    kv_len: int,
    P_past: int,
    compact_attn_bias: nn.Parameter | torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Build attention mask for single-token decode step.

    Single query token at the end of the sequence — all KV positions are
    visible (causal is trivially satisfied). Compact_kv positions get the
    learned bias; all other positions get 0.0.

    Args:
        B: Batch size.
        kv_len: Total KV length (P_past + tokens_in_block + 1 for new token).
        P_past: Number of compact_kv entries in the cache (0 or P).
        compact_attn_bias: (num_heads,) learned per-head bias.
        device: Torch device.
        dtype: Torch dtype.

    Returns:
        mask: (B, num_heads, 1, kv_len) float attention mask.
    """
    num_heads = compact_attn_bias.shape[0]

    # Start with 0.0 everywhere (all positions visible)
    mask = torch.zeros(B, num_heads, 1, kv_len, device=device, dtype=dtype)

    # Apply bias to compact_kv columns
    if P_past > 0:
        bias = compact_attn_bias.view(1, num_heads, 1, 1)
        mask[:, :, :, :P_past] = bias

    return mask


def build_compaction_mask(
    B: int,
    P: int,
    P_past: int,
    W: int,
    compact_attn_bias: nn.Parameter | torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Build attention mask for compaction-only forward pass.

    P compaction queries attend to P_past + W + P KV positions:
      - Columns 0:P_past         -> compact_kv from previous: bias
      - Columns P_past:P_past+W  -> text tokens: fully visible (0.0)
      - Columns P_past+W:end     -> compaction tokens: causal (lower-tri)

    This matches training rows W..W+P-1 of build_4d_attention_mask output.
    In training, query_len=W+P and tril(W+P, W+P) with diagonal=0 means
    row W sees columns 0..W. Here we have P rows that correspond to those
    training rows, so all W text columns must be visible.

    Args:
        B: Batch size.
        P: Number of compaction tokens (query length).
        P_past: Number of compact_kv entries (0 or P).
        W: Block size (number of text tokens in current block).
        compact_attn_bias: (num_heads,) learned per-head bias.
        device: Torch device.
        dtype: Torch dtype.

    Returns:
        mask: (B, num_heads, P, P_past + W + P) float attention mask.
    """
    min_val = torch.finfo(dtype).min
    num_heads = compact_attn_bias.shape[0]
    kv_len = P_past + W + P

    # Start with -inf everywhere
    mask = torch.full(
        (B, num_heads, P, kv_len), min_val, device=device, dtype=dtype
    )

    # Compact_kv columns (0:P_past): all visible with bias
    if P_past > 0:
        bias = compact_attn_bias.view(1, num_heads, 1, 1)
        mask[:, :, :, :P_past] = bias

    # Text columns (P_past:P_past+W): fully visible to ALL compaction queries
    mask[:, :, :, P_past:P_past + W] = 0.0

    # Compaction columns (P_past+W:end): lower-triangular causal among themselves
    # Query i can attend to compaction tokens 0..i (including self)
    causal = torch.tril(
        torch.zeros(P, P, device=device, dtype=dtype),
        diagonal=0,
    )
    upper = torch.triu(
        torch.ones(P, P, device=device, dtype=torch.bool),
        diagonal=1,
    )
    causal.masked_fill_(upper, min_val)
    mask[:, :, :, P_past + W:] = causal.unsqueeze(0).unsqueeze(0)

    return mask
