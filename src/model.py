"""CompactionModel: Qwen3-0.6B-Base + LoRA + compaction parameters.

Loads the pretrained model with eager attention, applies LoRA to all linear layers,
and adds two learnable compaction parameters:
  - compaction_embeddings: (P, hidden_size) — learned "summary tokens"
  - compact_attn_bias: (num_q_heads,) — per-head bias on compact_kv attention logits
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.config import CompactionConfig, Qwen3Dims


def setup_model(config: CompactionConfig) -> nn.Module:
    """Load Qwen3-0.6B-Base + LoRA + compaction parameters.

    CRITICAL RULES:
    1. attn_implementation="eager" — SDPA ignores our custom 4D mask
    2. NO gradient_checkpointing_enable() — breaks use_cache
    3. Freeze all base params, train only LoRA + compaction_embeddings + compact_attn_bias

    The peft wrapper structure is:
        PeftModelForCausalLM
          .base_model -> LoraModel
            .model -> Qwen3ForCausalLM
              .model -> Qwen3Model (layers, embed_tokens, norm, rotary_emb)
              .lm_head -> nn.Linear

    Args:
        config: CompactionConfig with model name, LoRA params, compaction init values.

    Returns:
        PeftModelForCausalLM with compaction_embeddings and compact_attn_bias attached.
    """
    from transformers import AutoModelForCausalLM
    from peft import LoraConfig, get_peft_model, TaskType

    dtype = torch.bfloat16 if config.bf16 else torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        dtype=dtype,
        attn_implementation=config.attn_implementation,
        trust_remote_code=True,
    )

    lora_config = LoraConfig(
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        target_modules=config.lora_targets,
        lora_dropout=0.0,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)

    # Add compaction parameters (NOT part of LoRA — separate param group)
    dims = Qwen3Dims()

    # Compaction embeddings: (P, hidden_size), learned "summary tokens"
    model.compaction_embeddings = nn.Parameter(
        torch.randn(config.P, dims.hidden_size, dtype=dtype) * config.embed_init_std
    )

    # Per-head attention bias (shared across layers): (num_q_heads,)
    # Suppresses compact_kv attention weight by ~exp(-2) ~ 0.14
    model.compact_attn_bias = nn.Parameter(
        torch.full((dims.num_q_heads,), config.bias_init, dtype=dtype)
    )

    return model


def get_inner_model(model: nn.Module) -> nn.Module:
    """Navigate peft wrapper to get Qwen3Model.

    Wrapper structure:
        model (PeftModelForCausalLM)
          -> .base_model (LoraModel)
            -> .model (Qwen3ForCausalLM)
              -> .model (Qwen3Model)

    The Qwen3Model holds: layers, embed_tokens, norm, rotary_emb.
    """
    return model.base_model.model.model


def get_lm_head(model: nn.Module) -> nn.Module:
    """Navigate peft wrapper to get lm_head.

    Wrapper structure:
        model (PeftModelForCausalLM)
          -> .base_model (LoraModel)
            -> .model (Qwen3ForCausalLM)
              -> .lm_head (nn.Linear)
    """
    return model.base_model.model.lm_head


def embed_tokens(model: nn.Module, input_ids: torch.Tensor) -> torch.Tensor:
    """Embed text tokens via the inner model's embedding layer.

    Args:
        model: PeftModelForCausalLM wrapper.
        input_ids: (B, T) token IDs.

    Returns:
        (B, T, hidden_size) embeddings.
    """
    return get_inner_model(model).embed_tokens(input_ids)


def get_rotary_embeddings(
    model: nn.Module,
    hidden_states: torch.Tensor,
    position_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Get (cos, sin) tuple for RoPE from the inner model's rotary embedding.

    CRITICAL: position_ids MUST be 2D (batch_size, seq_len).
    Qwen3RotaryEmbedding.forward() does:
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
    The [:, None, :] indexing requires position_ids to have exactly 2 dimensions.

    Args:
        model: PeftModelForCausalLM wrapper.
        hidden_states: (B, seq_len, hidden_size) — used for dtype/device inference.
        position_ids: (B, seq_len) — MUST be 2D.

    Returns:
        (cos, sin) each of shape (B, seq_len, head_dim).
    """
    assert position_ids.ndim == 2, (
        f"position_ids must be 2D (batch_size, seq_len), got shape {position_ids.shape}"
    )
    return get_inner_model(model).rotary_emb(hidden_states, position_ids)
