"""DynamicCache management for KV Self-Compaction.

Functions to extract, create, detach, and randomize compact_kv states.
Compact_kv is a list of (key, value) tensors, one per layer, each shaped
(B, num_kv_heads, P, head_dim) — matching HF DynamicCache's internal layout.

CRITICAL DESIGN DECISIONS (from Phase 1 debugging):
1. Use .clone() when extracting from cache — raw slices become invalid when
   the cache's internal tensor is overwritten by torch.cat() in the next block.
   clone() creates an independent tensor that preserves the gradient graph.
2. Use DynamicCache() without config= argument — passing config may create
   SlidingWindowLayer instead of DynamicLayer, breaking our extraction logic.
3. Use .update() per layer to pre-seed cache — do NOT use ddp_cache_data kwarg.
"""

from typing import Optional

import torch
from transformers import DynamicCache


def extract_compact_kv(
    cache: DynamicCache,
    P: int,
    num_layers: int,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Extract last P KV entries from each layer of a DynamicCache.

    After processing a block of W+P tokens, the cache contains KV states for all
    positions. We extract only the last P entries (the compaction token positions)
    to carry forward as the compact_kv for the next block.

    CRITICAL: Uses .clone() on each slice. Without this, the slice is a view into
    the cache's internal tensor, which gets invalidated when cache.update() calls
    torch.cat() in the next forward pass. clone() creates an independent copy that
    preserves the gradient graph (grad_fn = CloneBackward), enabling BPTT through
    compact_kv across blocks.

    Args:
        cache: DynamicCache after a forward pass. Each layer contains KV states
            shaped (B, num_kv_heads, seq_len, head_dim) where seq_len = P_past + W + P.
        P: Number of compaction tokens (last P positions to extract).
        num_layers: Number of decoder layers (28 for Qwen3-0.6B).

    Returns:
        List of (key, value) tuples, one per layer. Each tensor is shaped
        (B, num_kv_heads, P, head_dim) with gradient graph intact.
    """
    compact_kv = []
    for layer_idx in range(num_layers):
        k, v = cache[layer_idx]  # (B, num_kv_heads, seq_len, head_dim) each
        compact_kv.append((
            k[:, :, -P:, :].clone(),
            v[:, :, -P:, :].clone(),
        ))
    return compact_kv


def create_cache_with_compact_kv(
    compact_kv: list[tuple[torch.Tensor, torch.Tensor]],
    num_layers: int,
) -> DynamicCache:
    """Create a new DynamicCache pre-seeded with compact_kv from the previous block.

    For block_idx > 0, the cache needs to start with P compact_kv entries so that
    current-block tokens can attend to the compressed history. This function creates
    a fresh DynamicCache and populates it using .update() per layer.

    CRITICAL: Do NOT pass config= to DynamicCache() — it may create
    SlidingWindowLayer entries instead of DynamicLayer, which changes eviction
    behavior and breaks extraction. Do NOT use ddp_cache_data= kwarg either;
    the per-layer .update() API is the safe path.

    Args:
        compact_kv: List of (key, value) tuples, one per layer. Each tensor is
            shaped (B, num_kv_heads, P, head_dim).
        num_layers: Number of decoder layers. Must match len(compact_kv).

    Returns:
        DynamicCache with P entries per layer, ready to be passed as
        past_key_values to the model's forward pass.
    """
    cache = DynamicCache()
    for layer_idx in range(num_layers):
        k, v = compact_kv[layer_idx]
        cache.update(k, v, layer_idx)
    return cache


def detach_compact_kv(
    compact_kv: list[tuple[torch.Tensor, torch.Tensor]],
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Detach compact_kv tensors from the computation graph.

    Called after backward() at BPTT boundaries (every K blocks). This breaks the
    gradient chain so that the next K-block window starts a fresh computation graph,
    preventing memory from growing unboundedly with sequence length.

    Args:
        compact_kv: List of (key, value) tuples with gradient history.

    Returns:
        New list of (key, value) tuples with .detach() applied to each tensor.
        The returned tensors share storage with the originals but have no grad_fn.
    """
    return [(k.detach(), v.detach()) for k, v in compact_kv]


def random_compact_kv(
    B: int,
    P: int,
    num_layers: int,
    num_kv_heads: int,
    head_dim: int,
    device: torch.device,
    dtype: torch.dtype,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Generate random compact_kv for Condition D baseline.

    Condition D tests whether the compact_kv mechanism provides any benefit when
    the compact_kv content is uninformative (random noise). If Condition B
    (learned compact_kv) outperforms Condition D, the learned compaction is
    actually capturing useful information.

    Random values are scaled by 0.02 to match the initialization scale of
    compaction_embeddings, ensuring the attention logits stay in a reasonable range.

    Args:
        B: Batch size.
        P: Number of compaction tokens.
        num_layers: Number of decoder layers.
        num_kv_heads: Number of key/value heads (8 for Qwen3-0.6B GQA).
        head_dim: Dimension per head (128 for Qwen3-0.6B).
        device: Torch device.
        dtype: Torch dtype (typically torch.bfloat16).

    Returns:
        List of (key, value) tuples, one per layer. Each tensor is shaped
        (B, num_kv_heads, P, head_dim) with small random values (no gradient).
    """
    return [
        (
            torch.randn(B, num_kv_heads, P, head_dim, device=device, dtype=dtype) * 0.02,
            torch.randn(B, num_kv_heads, P, head_dim, device=device, dtype=dtype) * 0.02,
        )
        for _ in range(num_layers)
    ]
