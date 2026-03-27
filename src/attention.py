"""Attention mask construction and layer-by-layer forward pass.

build_4d_attention_mask: Constructs the 4D float attention mask with learnable bias
    on compact_kv columns for HF eager attention (additive mask).

forward_layers: Bypasses Qwen3Model.forward() to iterate decoder layers directly
    with our custom mask, avoiding HF's internal mask creation.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers.cache_utils import Cache, DynamicCache

from src.model import get_inner_model


def build_4d_attention_mask(
    B: int,
    query_len: int,
    kv_len: int,
    P_past: int,
    compact_attn_bias: nn.Parameter,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Build 4D float attention mask with learnable bias on compact_kv columns.

    HF eager attention applies this additively:
        attn_weights = attn_weights + causal_mask
    So 0.0 = no effect, -inf = masked out, bias value = suppress/amplify.

    Layout (kv_len = P_past + current_block_len):
        Columns 0:P_past         -> compact_kv: all attendable with learnable bias
        Columns P_past:kv_len    -> current block: standard lower-triangular causal

    Args:
        B: Batch size.
        query_len: Number of query positions (W + P for text + compaction tokens).
        kv_len: Number of KV positions (P_past + W + P).
        P_past: Number of compact_kv tokens from previous block. 0 for block 0.
        compact_attn_bias: (num_q_heads,) learnable bias, typically init to -2.0.
        device: Torch device.
        dtype: Torch dtype (bfloat16 typically).

    Returns:
        mask: (B, num_q_heads, query_len, kv_len) float tensor.
    """
    min_val = torch.finfo(dtype).min
    num_heads = compact_attn_bias.shape[0]

    # Start with -inf everywhere
    mask = torch.full(
        (B, num_heads, query_len, kv_len), min_val, device=device, dtype=dtype
    )

    # Compact_kv columns (0:P_past): attendable with learnable bias
    if P_past > 0:
        # bias shape: (num_heads,) -> (1, num_heads, 1, 1) for broadcasting
        bias = compact_attn_bias.view(1, num_heads, 1, 1)
        mask[:, :, :, :P_past] = bias

    # Current block region: standard lower-triangular causal mask
    # Query position q can attend to KV positions P_past through P_past+q (inclusive)
    current_block_len = kv_len - P_past
    # Create a (query_len, current_block_len) lower-triangular mask
    causal = torch.tril(
        torch.zeros(query_len, current_block_len, device=device, dtype=dtype),
        diagonal=0,
    )
    # Upper triangle (above diagonal) = -inf
    upper_mask = torch.triu(
        torch.ones(query_len, current_block_len, device=device, dtype=torch.bool),
        diagonal=1,
    )
    causal.masked_fill_(upper_mask, min_val)

    # Place into the current block region of the full mask
    # broadcast across B and num_heads
    mask[:, :, :, P_past:] = causal.unsqueeze(0).unsqueeze(0)

    return mask


def forward_layers(
    model: nn.Module,
    hidden_states: torch.Tensor,
    attention_mask_4d: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    past_key_values: Cache,
    cache_position: torch.Tensor,
    use_cache: bool = True,
) -> tuple[torch.Tensor, Cache]:
    """Forward through all Qwen3 decoder layers + final norm.

    Bypasses Qwen3Model.forward() to avoid its internal mask creation
    which would override our custom 4D mask with attention bias.

    Why this works with LoRA: peft replaces nn.Linear modules (q_proj, k_proj, etc.)
    with LoraLinear wrappers. Each layer's forward() calls self.q_proj(x) which goes
    through LoRA. So calling decoder layers directly does NOT break LoRA.

    Qwen3DecoderLayer.forward() signature:
        def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[Cache] = None,
            use_cache: Optional[bool] = False,
            cache_position: Optional[torch.LongTensor] = None,
            position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
            **kwargs,
        ) -> torch.Tensor

    It returns hidden_states directly (not a tuple).

    Args:
        model: PeftModelForCausalLM wrapper.
        hidden_states: (B, seq_len, hidden_size) — embedded text + compaction tokens.
        attention_mask_4d: (B, num_q_heads, seq_len, kv_len) — our custom mask.
        position_embeddings: (cos, sin) tuple from rotary_emb, each (B, seq_len, head_dim).
        past_key_values: DynamicCache (empty or pre-seeded with compact_kv).
        cache_position: (seq_len,) 1D tensor of slot indices for the cache.
        use_cache: Whether to update the KV cache (True for blockwise compaction).

    Returns:
        (hidden_states, past_key_values): Normed hidden states and updated cache.
    """
    inner = get_inner_model(model)

    for layer in inner.layers:
        # hidden_states is passed as positional arg (required for gradient checkpointing compat)
        hidden_states = layer(
            hidden_states,
            attention_mask=attention_mask_4d,
            position_ids=None,  # deprecated in Qwen3, use position_embeddings
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
        )

    hidden_states = inner.norm(hidden_states)
    return hidden_states, past_key_values
