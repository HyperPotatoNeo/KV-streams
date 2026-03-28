"""Transformers v5 compatibility layer.

The training code uses transformers v4.57.6. The inference engine uses v5.4.0+.
This module provides wrappers that handle API differences:

1. Decoder layer return type: v4 returns tensor, v5 may return tuple
2. DynamicCache API:
   - v4: cache[layer_idx] returns (keys, values)
   - v5: cache.layers[layer_idx].keys / .values
   - Both: cache.update(k, v, layer_idx) works
3. Qwen3RotaryEmbedding: interface unchanged (verified)
"""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers.cache_utils import DynamicCache


def _get_cache_kv(cache: DynamicCache, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Get (keys, values) from cache for a given layer. Works with v4 and v5."""
    # v5: cache.layers[layer_idx].keys / .values
    if hasattr(cache, "layers") and len(cache.layers) > layer_idx:
        layer = cache.layers[layer_idx]
        return layer.keys, layer.values
    # v4 fallback: cache[layer_idx]
    try:
        return cache[layer_idx]
    except (TypeError, KeyError, IndexError):
        pass
    # v4 fallback: cache.key_cache / cache.value_cache
    return cache.key_cache[layer_idx], cache.value_cache[layer_idx]


def _get_inner_model(model: nn.Module) -> nn.Module:
    """Get the inner Qwen3Model, handling both PEFT-wrapped and plain models.

    PEFT structure: PeftModelForCausalLM.base_model.model.model = Qwen3Model
    Plain structure: Qwen3ForCausalLM.model = Qwen3Model
    """
    # Try PEFT path first
    if hasattr(model, "base_model"):
        try:
            return model.base_model.model.model
        except AttributeError:
            pass
    # Plain Qwen3ForCausalLM
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model
    raise AttributeError(f"Cannot find inner model in {type(model)}")


def _get_lm_head(model: nn.Module) -> nn.Module:
    """Get lm_head, handling both PEFT-wrapped and plain models."""
    if hasattr(model, "base_model"):
        try:
            return model.base_model.model.lm_head
        except AttributeError:
            pass
    if hasattr(model, "lm_head"):
        return model.lm_head
    raise AttributeError(f"Cannot find lm_head in {type(model)}")


def _embed_tokens(model: nn.Module, input_ids: torch.Tensor) -> torch.Tensor:
    """Embed tokens, handling both PEFT-wrapped and plain models."""
    return _get_inner_model(model).embed_tokens(input_ids)


def _get_rotary_embeddings(
    model: nn.Module,
    hidden_states: torch.Tensor,
    position_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Get RoPE embeddings, handling both PEFT-wrapped and plain models."""
    assert position_ids.ndim == 2, f"position_ids must be 2D, got {position_ids.shape}"
    return _get_inner_model(model).rotary_emb(hidden_states, position_ids)


def forward_layers_compat(
    model: nn.Module,
    hidden_states: torch.Tensor,
    attention_mask_4d: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    past_key_values: DynamicCache,
    cache_position: torch.Tensor,
    use_cache: bool = True,
) -> tuple[torch.Tensor, DynamicCache]:
    """v5-compatible forward through all decoder layers + final norm.

    Handles the case where decoder layers return a tuple instead of a tensor.
    Works with both PEFT-wrapped and plain models.
    """
    inner = _get_inner_model(model)

    for layer in inner.layers:
        result = layer(
            hidden_states,
            attention_mask=attention_mask_4d,
            position_ids=None,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
        )
        # v4: returns tensor directly
        # v5: may return tuple (hidden_states, ...)
        if isinstance(result, tuple):
            hidden_states = result[0]
        else:
            hidden_states = result

    hidden_states = inner.norm(hidden_states)
    return hidden_states, past_key_values


def extract_compact_kv_compat(
    cache: DynamicCache,
    P: int,
    num_layers: int,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """v5-compatible compact_kv extraction from DynamicCache.

    Uses _get_cache_kv to handle v4/v5 differences.
    Uses .clone() to preserve tensor independence (as in training code).
    """
    compact_kv = []
    for layer_idx in range(num_layers):
        k, v = _get_cache_kv(cache, layer_idx)
        compact_kv.append((
            k[:, :, -P:, :].clone(),
            v[:, :, -P:, :].clone(),
        ))
    return compact_kv


def create_cache_with_compact_kv_compat(
    compact_kv: list[tuple[torch.Tensor, torch.Tensor]],
    num_layers: int,
) -> DynamicCache:
    """v5-compatible cache creation pre-seeded with compact_kv.

    cache.update(k, v, layer_idx) works in both v4 and v5.
    """
    cache = DynamicCache()
    for layer_idx in range(num_layers):
        k, v = compact_kv[layer_idx]
        cache.update(k, v, layer_idx)
    return cache
