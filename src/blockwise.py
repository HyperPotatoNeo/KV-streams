"""
Blockwise forward with compaction — the core mechanism.

Processes input sequences in blocks of W tokens, appending P compaction embeddings
per block. Compact KV from each block carries cross-block information to the next.
BPTT truncation every K blocks bounds memory and gradient graph size.

Two entry points:
  - blockwise_train_step(): gradient-enabled, calls backward every K blocks
  - blockwise_forward_eval(): no gradient, returns logits for all text positions
"""

import torch
import torch.nn.functional as F
from transformers.cache_utils import DynamicCache

from src.config import CompactionConfig
from src.model import embed_tokens, get_lm_head, get_rotary_embeddings, get_inner_model
from src.attention import build_4d_attention_mask, forward_layers
from src.kv_manager import (
    extract_compact_kv,
    create_cache_with_compact_kv,
    detach_compact_kv,
    random_compact_kv,
)


def blockwise_train_step(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    attention_mask: torch.Tensor,
    config: CompactionConfig,
) -> float:
    """One micro-batch blockwise training step with compaction.

    Processes input_ids in blocks of W tokens, appending P compaction embeddings
    per block. BPTT: calls backward() every K blocks, then detaches compact_kv.

    Args:
        model: PeftModel wrapping Qwen3ForCausalLM with compaction params.
        input_ids: (B, seq_len) token IDs, seq_len must be divisible by W.
        labels: (B, seq_len) target token IDs, -100 for ignored positions.
        attention_mask: (B, seq_len) binary mask (1=real, 0=padding).
        config: CompactionConfig with W, P, K, condition, etc.

    Returns:
        Mean per-token loss (float, detached) for logging.
    """
    W = config.W
    P = config.P
    K = config.K

    B, seq_len = input_ids.shape
    device = input_ids.device
    dtype = next(model.parameters()).dtype
    num_layers = model.config.num_hidden_layers   # 28
    num_kv_heads = model.config.num_key_value_heads  # 8
    head_dim = getattr(model.config, "head_dim", 128)
    num_blocks = seq_len // W

    compact_kv = None          # None for block 0, list of (k, v) tuples after
    window_loss = 0.0          # accumulated loss within BPTT window
    window_tokens = 0          # token count within BPTT window
    blocks_in_window = 0       # blocks processed in current BPTT window
    total_loss_sum = 0.0       # total loss for logging (detached)
    total_tokens = 0           # total valid tokens for logging

    for block_idx in range(num_blocks):
        start = block_idx * W
        end = start + W

        # 1. Embed text tokens
        block_ids = input_ids[:, start:end]          # (B, W)
        block_labels = labels[:, start:end]           # (B, W)
        text_embeds = embed_tokens(model, block_ids)  # (B, W, hidden_size)

        # 2. Concatenate compaction embeddings
        comp_embeds = model.compaction_embeddings.unsqueeze(0).expand(B, -1, -1)  # (B, P, hidden_size)
        block_embeds = torch.cat([text_embeds, comp_embeds], dim=1)  # (B, W+P, hidden_size)

        # 3. Position IDs — MUST be 2D (B, seq_len) for Qwen3RotaryEmbedding
        block_start_pos = block_idx * (W + P)
        position_ids = torch.arange(
            block_start_pos, block_start_pos + W + P, device=device
        ).unsqueeze(0).expand(B, -1)  # (B, W+P) — 2D!

        # 4. Cache position (slot indices within the KV cache)
        # Condition E: no compact_kv between blocks (within-block only baseline)
        if config.condition == "E":
            P_past = 0
        else:
            P_past = 0 if compact_kv is None else P
        cache_position = torch.arange(P_past, P_past + W + P, device=device)

        # 5. Build past_key_values cache
        if compact_kv is not None and config.condition != "E":
            if config.condition == "D":
                # Condition D: random compact_kv baseline
                past_cache = create_cache_with_compact_kv(
                    random_compact_kv(B, P, num_layers, num_kv_heads, head_dim, device, dtype),
                    num_layers,
                )
            else:
                past_cache = create_cache_with_compact_kv(compact_kv, num_layers)
        else:
            # Block 0 or Condition E: empty cache.
            past_cache = DynamicCache()

        # 6. Build 4D attention mask with learnable bias + padding masking
        block_padding = attention_mask[:, start:end]  # (B, W) — 1=real, 0=padding
        attn_mask = build_4d_attention_mask(
            B, W + P, P_past + W + P, P_past,
            model.compact_attn_bias, device, dtype,
            padding_mask=block_padding,
        )

        # 7. RoPE position embeddings (cos, sin)
        position_embeddings = get_rotary_embeddings(model, block_embeds, position_ids)

        # 8. Forward through all decoder layers + final norm
        hidden, past_cache = forward_layers(
            model, block_embeds, attn_mask, position_embeddings,
            past_cache, cache_position, use_cache=True,
        )

        # 9. Logits on text positions only (not compaction token positions)
        text_h = hidden[:, :W, :]
        logits = get_lm_head(model)(text_h).float()  # (B, W, vocab_size)

        # 10. Loss (sum reduction for BPTT accumulation)
        valid_mask = block_labels != -100
        num_valid = valid_mask.sum().item()
        if num_valid > 0:
            block_loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                block_labels.reshape(-1),
                ignore_index=-100,
                reduction="sum",
            )
            window_loss = window_loss + block_loss
            window_tokens += num_valid
            total_loss_sum += block_loss.detach().item()
            total_tokens += num_valid

        blocks_in_window += 1

        # 11. Extract compact_kv — .clone() to preserve gradient graph!
        compact_kv = extract_compact_kv(past_cache, P, num_layers)

        # 12. BPTT: backward every K blocks, then detach
        if blocks_in_window >= K:
            if window_tokens > 0:
                (window_loss / window_tokens).backward()
            compact_kv = detach_compact_kv(compact_kv)
            window_loss = 0.0
            window_tokens = 0
            blocks_in_window = 0

    # Handle remaining blocks if num_blocks % K != 0
    if window_tokens > 0:
        (window_loss / window_tokens).backward()

    return total_loss_sum / max(total_tokens, 1)


@torch.no_grad()
def blockwise_forward_eval(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    config: CompactionConfig,
) -> torch.Tensor:
    """Blockwise forward WITHOUT gradient. Returns logits for all text positions.

    Same blockwise loop as blockwise_train_step but:
      - No backward / BPTT
      - No loss computation
      - Collects and concatenates logits from all blocks

    Args:
        model: PeftModel wrapping Qwen3ForCausalLM with compaction params.
        input_ids: (B, seq_len) token IDs, seq_len must be divisible by W.
        attention_mask: (B, seq_len) binary mask (1=real, 0=padding).
        config: CompactionConfig with W, P, condition, etc.

    Returns:
        logits: (B, seq_len, vocab_size) logits for all text positions concatenated.
    """
    W = config.W
    P = config.P

    B, seq_len = input_ids.shape
    device = input_ids.device
    dtype = next(model.parameters()).dtype
    num_layers = model.config.num_hidden_layers
    num_kv_heads = model.config.num_key_value_heads
    head_dim = getattr(model.config, "head_dim", 128)
    num_blocks = seq_len // W

    compact_kv = None
    all_logits = []

    for block_idx in range(num_blocks):
        start = block_idx * W
        end = start + W

        # 1. Embed text tokens
        block_ids = input_ids[:, start:end]
        text_embeds = embed_tokens(model, block_ids)

        # 2. Concatenate compaction embeddings
        comp_embeds = model.compaction_embeddings.unsqueeze(0).expand(B, -1, -1)
        block_embeds = torch.cat([text_embeds, comp_embeds], dim=1)

        # 3. Position IDs — 2D (B, W+P)
        block_start_pos = block_idx * (W + P)
        position_ids = torch.arange(
            block_start_pos, block_start_pos + W + P, device=device
        ).unsqueeze(0).expand(B, -1)

        # 4. Cache position
        if config.condition == "E":
            P_past = 0
        else:
            P_past = 0 if compact_kv is None else P
        cache_position = torch.arange(P_past, P_past + W + P, device=device)

        # 5. Build past_key_values cache
        if compact_kv is not None and config.condition != "E":
            if config.condition == "D":
                past_cache = create_cache_with_compact_kv(
                    random_compact_kv(B, P, num_layers, num_kv_heads, head_dim, device, dtype),
                    num_layers,
                )
            else:
                past_cache = create_cache_with_compact_kv(compact_kv, num_layers)
        else:
            past_cache = DynamicCache()

        # 6. Build 4D attention mask with bias + padding masking
        block_padding = attention_mask[:, start:end]  # (B, W)
        attn_mask = build_4d_attention_mask(
            B, W + P, P_past + W + P, P_past,
            model.compact_attn_bias, device, dtype,
            padding_mask=block_padding,
        )

        # 7. RoPE position embeddings
        position_embeddings = get_rotary_embeddings(model, block_embeds, position_ids)

        # 8. Forward through layers
        hidden, past_cache = forward_layers(
            model, block_embeds, attn_mask, position_embeddings,
            past_cache, cache_position, use_cache=True,
        )

        # 9. Logits on text positions only
        text_h = hidden[:, :W, :]
        logits = get_lm_head(model)(text_h).float()  # (B, W, vocab_size)
        all_logits.append(logits)

        # 10. Extract compact_kv for next block (.clone()!)
        compact_kv = extract_compact_kv(past_cache, P, num_layers)

    # Concatenate logits from all blocks: (B, num_blocks*W, vocab_size) = (B, seq_len, vocab_size)
    return torch.cat(all_logits, dim=1)
