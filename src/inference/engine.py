"""CompactionInferenceEngine: blockwise generation with KV self-compaction.

Reuses training code (forward_layers, build_4d_attention_mask, extract_compact_kv)
for guaranteed correctness. Custom masks for decode and compaction-only passes.

Algorithm:
  1. Process prompt blockwise: full W-sized blocks with compaction, partial block without
  2. Autoregressive decode with custom mask (bias on compact_kv positions)
  3. Every W decoded tokens, run compaction pass (P embeddings → compact_kv)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache

from src.attention import build_4d_attention_mask
from src.inference.compat import (
    forward_layers_compat,
    extract_compact_kv_compat,
    create_cache_with_compact_kv_compat,
    _get_inner_model,
    _get_lm_head,
    _embed_tokens,
    _get_rotary_embeddings,
)
from src.inference.masks import build_decode_mask, build_compaction_mask
from src.inference.sampling import sample_token, compute_logprob


@dataclass
class CompactionState:
    """Mutable state for blockwise generation."""

    compact_kv: list[tuple[torch.Tensor, torch.Tensor]] | None
    cache: DynamicCache
    block_idx: int
    tokens_in_block: int  # Tokens already in current block (BEFORE next token)
    generated_ids: list[int] = field(default_factory=list)
    logprobs: list[float] = field(default_factory=list)


@dataclass
class GenerationResult:
    """Output of a generation call."""

    token_ids: list[int]
    text: str
    logprobs: list[float] | None
    finish_reason: str  # "stop" or "length"
    prompt_token_ids: list[int]


class CompactionInferenceEngine:
    """HuggingFace-based inference with KV self-compaction.

    Reuses training code for guaranteed correctness:
      - build_4d_attention_mask for full-block prompt processing
      - forward_layers_compat for all forward passes
      - extract_compact_kv_compat for KV extraction
    """

    def __init__(
        self,
        base_model_name: str,
        adapter_path: Optional[str] = None,
        compaction_params_path: Optional[str] = None,
        W: int = 512,
        P: int = 64,
        device: str = "cuda",
        dtype_str: str = "bfloat16",
    ):
        self.W = W
        self.P = P
        self.device = torch.device(device)
        self.dtype = getattr(torch, dtype_str)

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            base_model_name, trust_remote_code=True
        )

        # Load base model with eager attention
        self.model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            dtype=self.dtype,
            attn_implementation="eager",
            trust_remote_code=True,
        ).to(self.device)

        # Load LoRA adapter if provided
        if adapter_path:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(
                self.model, adapter_path
            ).to(self.device)

        # Load compaction parameters
        if compaction_params_path:
            state = torch.load(
                compaction_params_path, map_location=self.device, weights_only=True
            )
            self.model.compaction_embeddings = nn.Parameter(
                state["compaction_embeddings"].to(self.device, self.dtype),
                requires_grad=False,
            )
            self.model.compact_attn_bias = nn.Parameter(
                state["compact_attn_bias"].to(self.device, self.dtype),
                requires_grad=False,
            )

        # Model dimensions
        config = self.model.config
        self.num_layers = config.num_hidden_layers
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = getattr(config, "head_dim", 128)

        self.model.eval()

    @torch.inference_mode()
    def generate(
        self,
        prompt_ids: list[int],
        max_new_tokens: int = 2048,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = -1,
        stop_token_ids: Optional[list[int]] = None,
        return_logprobs: bool = False,
    ) -> GenerationResult:
        """Generate text with blockwise KV self-compaction.

        Args:
            prompt_ids: Tokenized prompt as list of ints.
            max_new_tokens: Maximum tokens to generate.
            temperature: Sampling temperature (0 = greedy).
            top_p: Nucleus sampling threshold.
            top_k: Top-k filtering (-1 = disabled).
            stop_token_ids: Token IDs that stop generation.
            return_logprobs: Whether to compute per-token logprobs.

        Returns:
            GenerationResult with generated tokens, text, logprobs, finish_reason.
        """
        if stop_token_ids is None:
            stop_token_ids = [self.tokenizer.eos_token_id]

        # Process prompt blockwise
        state, prompt_logits = self._process_prompt_blockwise(prompt_ids)

        # Get initial logits from last prompt position
        logits = prompt_logits  # (1, 1, vocab_size)

        finish_reason = "length"

        for _ in range(max_new_tokens):
            # Sample from current logits
            token_id, logprob = sample_token(
                logits.squeeze(0).squeeze(0), temperature, top_p, top_k
            )

            state.generated_ids.append(token_id)
            if return_logprobs:
                state.logprobs.append(logprob)

            # Check stop condition
            if token_id in stop_token_ids:
                finish_reason = "stop"
                break

            # Decode step: single token forward pass
            logits, state = self._decode_step(token_id, state)

            # Compaction trigger
            if state.tokens_in_block == self.W:
                state = self._run_compaction(state)

        text = self.tokenizer.decode(state.generated_ids, skip_special_tokens=True)

        return GenerationResult(
            token_ids=state.generated_ids,
            text=text,
            logprobs=state.logprobs if return_logprobs else None,
            finish_reason=finish_reason,
            prompt_token_ids=prompt_ids,
        )

    def _process_prompt_blockwise(
        self, prompt_ids: list[int]
    ) -> tuple[CompactionState, torch.Tensor]:
        """Process prompt in W-sized blocks with compaction.

        Full blocks get compaction embeddings appended (matching training).
        Last partial block is processed WITHOUT compaction (model never
        trained on partial-block compaction).

        Returns:
            (state, logits): CompactionState and logits from last position.
        """
        W, P = self.W, self.P
        device, dtype = self.device, self.dtype

        input_ids = torch.tensor([prompt_ids], device=device)  # (1, seq_len)
        seq_len = input_ids.shape[1]

        num_full_blocks = seq_len // W
        partial_len = seq_len % W

        compact_kv = None
        block_idx = 0

        # Process full blocks with compaction
        for bi in range(num_full_blocks):
            start = bi * W
            end = start + W
            block_ids = input_ids[:, start:end]  # (1, W)

            text_embeds = _embed_tokens(self.model, block_ids)  # (1, W, hidden)
            comp_embeds = self.model.compaction_embeddings.unsqueeze(0)  # (1, P, hidden)
            block_embeds = torch.cat([text_embeds, comp_embeds], dim=1)  # (1, W+P, hidden)

            P_past = 0 if compact_kv is None else P
            block_start_pos = block_idx * (W + P)
            position_ids = torch.arange(
                block_start_pos, block_start_pos + W + P, device=device
            ).unsqueeze(0)  # (1, W+P) — MUST be 2D

            cache_position = torch.arange(P_past, P_past + W + P, device=device)

            if compact_kv is not None:
                cache = create_cache_with_compact_kv_compat(compact_kv, self.num_layers)
            else:
                cache = DynamicCache()

            attn_mask = build_4d_attention_mask(
                1, W + P, P_past + W + P, P_past,
                self.model.compact_attn_bias, device, dtype,
            )

            position_embeddings = _get_rotary_embeddings(
                self.model, block_embeds, position_ids
            )

            hidden, cache = forward_layers_compat(
                self.model, block_embeds, attn_mask, position_embeddings,
                cache, cache_position,
            )

            compact_kv = extract_compact_kv_compat(cache, P, self.num_layers)
            block_idx += 1

        # Process partial block WITHOUT compaction
        if partial_len > 0:
            start = num_full_blocks * W
            partial_ids = input_ids[:, start:]  # (1, partial_len)
            text_embeds = _embed_tokens(self.model, partial_ids)  # (1, partial_len, hidden)

            P_past = 0 if compact_kv is None else P
            block_start_pos = block_idx * (W + P)
            position_ids = torch.arange(
                block_start_pos, block_start_pos + partial_len, device=device
            ).unsqueeze(0)

            cache_position = torch.arange(P_past, P_past + partial_len, device=device)

            if compact_kv is not None:
                cache = create_cache_with_compact_kv_compat(compact_kv, self.num_layers)
            else:
                cache = DynamicCache()

            # Use build_4d_attention_mask with partial_len instead of W+P
            attn_mask = build_4d_attention_mask(
                1, partial_len, P_past + partial_len, P_past,
                self.model.compact_attn_bias, device, dtype,
            )

            position_embeddings = _get_rotary_embeddings(
                self.model, text_embeds, position_ids
            )

            hidden, cache = forward_layers_compat(
                self.model, text_embeds, attn_mask, position_embeddings,
                cache, cache_position,
            )

            tokens_in_block = partial_len
        elif num_full_blocks > 0:
            # Prompt is exact multiple of W — last full block already processed
            # with compaction. Cache was replaced with compact_kv.
            # hidden is from last full block: (1, W+P, hidden_size)
            # Last TEXT token is at position W-1, NOT -1 (which is last compaction token)
            cache = create_cache_with_compact_kv_compat(compact_kv, self.num_layers)
            tokens_in_block = 0
        else:
            # Empty prompt — run BOS token through model for meaningful logits
            bos_id = self.tokenizer.bos_token_id or 0
            bos_tensor = torch.tensor([[bos_id]], device=device)
            bos_embed = _embed_tokens(self.model, bos_tensor)  # (1, 1, hidden_size)

            cache = DynamicCache()
            cache_position = torch.arange(1, device=device)
            position_ids = torch.zeros(1, 1, device=device, dtype=torch.long)
            mask = torch.zeros(1, self.model.compact_attn_bias.shape[0], 1, 1,
                               device=device, dtype=self.dtype)
            position_embeddings = _get_rotary_embeddings(self.model, bos_embed, position_ids)

            hidden, cache = forward_layers_compat(
                self.model, bos_embed, mask, position_embeddings, cache, cache_position,
            )
            tokens_in_block = 1  # BOS is in the cache now

        # Compute logits from the last TEXT position
        if partial_len > 0:
            # Partial block: last position is the last text token
            logits = _get_lm_head(self.model)(hidden[:, -1:, :]).float()
        elif num_full_blocks > 0:
            # Full blocks only: hidden has shape (1, W+P, hidden_size)
            # Last text position is at index W-1, compaction tokens are at W..W+P-1
            logits = _get_lm_head(self.model)(hidden[:, W - 1:W, :]).float()
        else:
            # Empty prompt: logits from forwarded BOS token
            logits = _get_lm_head(self.model)(hidden[:, -1:, :]).float()

        state = CompactionState(
            compact_kv=compact_kv,
            cache=cache,
            block_idx=block_idx,
            tokens_in_block=tokens_in_block,
        )

        return state, logits

    def _decode_step(
        self, token_id: int, state: CompactionState
    ) -> tuple[torch.Tensor, CompactionState]:
        """Single autoregressive decode step with correct mask.

        Returns (logits, updated_state) where logits predict the NEXT token.
        """
        W, P = self.W, self.P
        device, dtype = self.device, self.dtype

        token_tensor = torch.tensor([[token_id]], device=device)
        token_embed = _embed_tokens(self.model, token_tensor)  # (1, 1, hidden)

        P_past = P if state.compact_kv is not None else 0
        # KV length: P_past compact_kv + tokens_in_block already in cache + 1 new
        kv_len = P_past + state.tokens_in_block + 1

        mask = build_decode_mask(
            1, kv_len, P_past, self.model.compact_attn_bias, device, dtype
        )  # (1, num_heads, 1, kv_len)

        # Position ID for this token
        pos = state.block_idx * (W + P) + state.tokens_in_block
        position_ids = torch.tensor([[pos]], device=device)  # (1, 1) — 2D

        # Cache position: where to write in the cache
        cache_pos = torch.tensor([P_past + state.tokens_in_block], device=device)

        position_embeddings = _get_rotary_embeddings(
            self.model, token_embed, position_ids
        )

        hidden, state.cache = forward_layers_compat(
            self.model, token_embed, mask, position_embeddings,
            state.cache, cache_pos,
        )

        logits = _get_lm_head(self.model)(hidden).float()  # (1, 1, vocab_size)
        state.tokens_in_block += 1

        return logits, state

    def _run_compaction(self, state: CompactionState) -> CompactionState:
        """Run compaction pass: P compaction embeddings → compact_kv.

        After W decode tokens, append P compaction embeddings and forward
        through the model. Extract the last P KV entries as new compact_kv.
        Reset cache to compact_kv only.
        """
        W, P = self.W, self.P
        device, dtype = self.device, self.dtype

        comp_embeds = self.model.compaction_embeddings.unsqueeze(0)  # (1, P, hidden)

        P_past = P if state.compact_kv is not None else 0

        # Compaction mask: P queries, P_past + W + P KVs
        mask = build_compaction_mask(
            1, P, P_past, W, self.model.compact_attn_bias, device, dtype
        )  # (1, num_heads, P, P_past+W+P)

        # Position IDs for compaction tokens
        comp_start = state.block_idx * (W + P) + W
        position_ids = torch.arange(
            comp_start, comp_start + P, device=device
        ).unsqueeze(0)  # (1, P)

        # Cache position: compaction tokens go after P_past + W text tokens
        cache_pos = torch.arange(P_past + W, P_past + W + P, device=device)

        position_embeddings = _get_rotary_embeddings(
            self.model, comp_embeds, position_ids
        )

        hidden, state.cache = forward_layers_compat(
            self.model, comp_embeds, mask, position_embeddings,
            state.cache, cache_pos,
        )

        # Extract compact_kv and reset cache
        state.compact_kv = extract_compact_kv_compat(state.cache, P, self.num_layers)
        state.cache = create_cache_with_compact_kv_compat(state.compact_kv, self.num_layers)
        state.block_idx += 1
        state.tokens_in_block = 0

        return state

    def update_weights(self, weight_dir: str):
        """Reload model weights from exported checkpoint directory.

        Expects:
          weight_dir/adapter/  — PEFT adapter
          weight_dir/compaction_params.pt  — compaction parameters
        """
        import os

        adapter_path = os.path.join(weight_dir, "adapter")
        compaction_path = os.path.join(weight_dir, "compaction_params.pt")

        # Preserve compaction params before replacing model
        old_comp_embeddings = getattr(self.model, "compaction_embeddings", None)
        old_comp_bias = getattr(self.model, "compact_attn_bias", None)

        # Reload LoRA adapter
        if os.path.exists(adapter_path):
            from peft import PeftModel
            if hasattr(self.model, "base_model"):
                base = self.model.get_base_model()
            else:
                base = self.model
            self.model = PeftModel.from_pretrained(base, adapter_path).to(self.device)

        # Reload compaction parameters (or reattach old ones)
        if os.path.exists(compaction_path):
            state = torch.load(compaction_path, map_location=self.device, weights_only=True)
            self.model.compaction_embeddings = nn.Parameter(
                state["compaction_embeddings"].to(self.device, self.dtype),
                requires_grad=False,
            )
            self.model.compact_attn_bias = nn.Parameter(
                state["compact_attn_bias"].to(self.device, self.dtype),
                requires_grad=False,
            )
        else:
            # No new compaction params file — reattach old ones
            if old_comp_embeddings is not None:
                self.model.compaction_embeddings = old_comp_embeddings
            if old_comp_bias is not None:
                self.model.compact_attn_bias = old_comp_bias

        self.model.eval()

    def load_adapter(self, lora_name: str, lora_path: str):
        """Load a LoRA adapter directly from its path.

        Unlike update_weights which expects a parent directory,
        this takes the adapter directory directly.
        """
        from peft import PeftModel

        # Preserve compaction params
        old_comp_embeddings = getattr(self.model, "compaction_embeddings", None)
        old_comp_bias = getattr(self.model, "compact_attn_bias", None)

        if hasattr(self.model, "base_model"):
            base = self.model.get_base_model()
        else:
            base = self.model
        self.model = PeftModel.from_pretrained(base, lora_path).to(self.device)

        # Reattach compaction params
        if old_comp_embeddings is not None:
            self.model.compaction_embeddings = old_comp_embeddings
        if old_comp_bias is not None:
            self.model.compact_attn_bias = old_comp_bias

        self.model.eval()
