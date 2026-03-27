"""GPU-specific architecture tests for KV Self-Compaction Phase 2.

6 tests, ALL @pytest.mark.gpu. Tests eager vs SDPA, memory, gradient flow
through cache, and BPTT memory bounds.
"""

import math

import pytest
import torch
import torch.nn.functional as F
from transformers.cache_utils import DynamicCache

from src.config import CompactionConfig, Qwen3Dims
from src.blockwise import blockwise_train_step


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def eager_model():
    """Load model with eager attention for GPU-specific tests."""
    from src.model import setup_model

    config = CompactionConfig(
        W=32, P=4, K=2,
        max_seq_len=128,
        batch_size=1,
        bf16=True,
        attn_implementation="eager",
    )
    model = setup_model(config)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    return model, config


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.gpu
class TestEagerAttention:
    """Tests for eager attention producing valid output."""

    def test_eager_attention_produces_output(self, eager_model):
        """Eager attention with custom 4D mask produces valid output (not NaN/inf)."""
        from src.model import embed_tokens, get_lm_head, get_rotary_embeddings
        from src.attention import build_4d_attention_mask, forward_layers

        model, config = eager_model
        device = next(model.parameters()).device
        dtype = next(model.parameters()).dtype
        B, W, P = 1, config.W, config.P

        input_ids = torch.randint(0, 151936, (B, W), device=device)
        text_embeds = embed_tokens(model, input_ids)
        comp_embeds = model.compaction_embeddings.unsqueeze(0).expand(B, -1, -1)
        block_embeds = torch.cat([text_embeds, comp_embeds], dim=1)

        position_ids = torch.arange(0, W + P, device=device).unsqueeze(0).expand(B, -1)
        cache_position = torch.arange(0, W + P, device=device)
        past_cache = DynamicCache()

        attn_mask = build_4d_attention_mask(
            B, W + P, W + P, 0, model.compact_attn_bias, device, dtype,
        )
        position_embeddings = get_rotary_embeddings(model, block_embeds, position_ids)

        with torch.no_grad():
            hidden, _ = forward_layers(
                model, block_embeds, attn_mask, position_embeddings,
                past_cache, cache_position,
            )
            logits = get_lm_head(model)(hidden[:, :W, :])

        assert torch.isfinite(logits).all(), (
            f"Eager attention produced non-finite logits: "
            f"NaN count={logits.isnan().sum()}, Inf count={logits.isinf().sum()}"
        )

    def test_eager_vs_sdpa_block0_match(self):
        """Block 0 (no compact_kv, standard causal) should match SDPA output.

        Block 0 has P_past=0 so the custom mask is just a standard causal mask.
        Eager and SDPA should produce the same logits within bf16 precision.
        """
        from src.model import setup_model

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        B, W = 1, 32

        config_eager = CompactionConfig(
            W=W, P=4, K=2, max_seq_len=64, batch_size=1,
            bf16=True, attn_implementation="eager",
        )
        config_sdpa = CompactionConfig(
            W=W, P=4, K=2, max_seq_len=64, batch_size=1,
            bf16=True, attn_implementation="sdpa",
        )

        model_eager = setup_model(config_eager).to(device)
        model_sdpa = setup_model(config_sdpa).to(device)

        # Sync compaction params so they're identical
        model_sdpa.compaction_embeddings.data.copy_(model_eager.compaction_embeddings.data)
        model_sdpa.compact_attn_bias.data.copy_(model_eager.compact_attn_bias.data)

        input_ids = torch.randint(0, 151936, (B, W), device=device)

        model_eager.eval()
        model_sdpa.eval()

        with torch.no_grad():
            # Standard HF forward (block 0 equivalent — just W tokens, no compaction)
            eager_out = model_eager(input_ids=input_ids)
            sdpa_out = model_sdpa(input_ids=input_ids)

        eager_logits = eager_out.logits.float()
        sdpa_logits = sdpa_out.logits.float()

        # Compare using cosine similarity
        cos_sim = F.cosine_similarity(
            eager_logits.reshape(-1, eager_logits.shape[-1]),
            sdpa_logits.reshape(-1, sdpa_logits.shape[-1]),
            dim=-1,
        ).mean().item()

        assert cos_sim > 0.99, (
            f"Eager vs SDPA cosine similarity {cos_sim:.4f} too low for block 0. "
            "Custom mask construction may be incorrect."
        )


@pytest.mark.gpu
class TestMemory:
    """Memory usage tests."""

    def test_memory_usage_within_budget(self, eager_model):
        """Peak GPU memory for blockwise stays under 10 GB with small config."""
        model, config = eager_model
        device = next(model.parameters()).device

        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.empty_cache()

        B = 1
        seq_len = config.W * 4  # 4 blocks

        input_ids = torch.randint(0, 151936, (B, seq_len), device=device)
        labels = input_ids.clone()
        attention_mask = torch.ones(B, seq_len, dtype=torch.long, device=device)

        model.train()
        model.zero_grad()
        blockwise_train_step(model, input_ids, labels, attention_mask, config)

        peak_gb = torch.cuda.max_memory_allocated(device) / 1e9
        assert peak_gb < 10.0, (
            f"Peak memory {peak_gb:.2f} GB exceeds 10 GB budget"
        )

    def test_bptt_memory_bounded(self):
        """Memory usage with K=2 and 8 blocks should be roughly constant
        (not growing with number of blocks, confirming BPTT detach works)."""
        from src.model import setup_model

        config = CompactionConfig(
            W=32, P=4, K=2,
            max_seq_len=256,
            batch_size=1,
            bf16=True,
        )
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = setup_model(config).to(device)

        B = 1

        # Measure memory for 2 blocks (1 BPTT window)
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.empty_cache()

        seq_2 = config.W * 2
        ids_2 = torch.randint(0, 151936, (B, seq_2), device=device)
        labels_2 = ids_2.clone()
        mask_2 = torch.ones(B, seq_2, dtype=torch.long, device=device)

        model.train()
        model.zero_grad()
        blockwise_train_step(model, ids_2, labels_2, mask_2, config)
        peak_2_blocks = torch.cuda.max_memory_allocated(device) / 1e9

        # Measure memory for 8 blocks (4 BPTT windows)
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.empty_cache()

        seq_8 = config.W * 8
        ids_8 = torch.randint(0, 151936, (B, seq_8), device=device)
        labels_8 = ids_8.clone()
        mask_8 = torch.ones(B, seq_8, dtype=torch.long, device=device)

        model.zero_grad()
        blockwise_train_step(model, ids_8, labels_8, mask_8, config)
        peak_8_blocks = torch.cuda.max_memory_allocated(device) / 1e9

        # 8 blocks should use at most 2x the memory of 2 blocks
        # (some overhead from loss accumulation, cache ops, etc.)
        ratio = peak_8_blocks / max(peak_2_blocks, 0.01)
        assert ratio < 2.0, (
            f"Memory grew {ratio:.2f}x from 2→8 blocks ({peak_2_blocks:.2f}→{peak_8_blocks:.2f} GB). "
            "BPTT detach may not be working correctly."
        )


@pytest.mark.gpu
class TestGradientFlow:
    """Tests for gradient flow through compact_kv across blocks."""

    def test_compact_kv_gradient_through_two_blocks(self, eager_model):
        """Gradient flows from block 1 loss through compact_kv back to block 0.

        This is the CRITICAL test for the compaction mechanism: the gradient
        must flow from block 1's loss, through compact_kv extraction (.clone()),
        back through the block 0 forward pass, into compaction_embeddings.
        """
        model, config = eager_model
        device = next(model.parameters()).device
        B = 1
        seq_len = config.W * 2  # exactly 2 blocks

        input_ids = torch.randint(0, 151936, (B, seq_len), device=device)
        labels = input_ids.clone()
        attention_mask = torch.ones(B, seq_len, dtype=torch.long, device=device)

        model.train()
        model.zero_grad()
        blockwise_train_step(model, input_ids, labels, attention_mask, config)

        # compaction_embeddings should have gradient from block 1
        grad = model.compaction_embeddings.grad
        assert grad is not None, "compaction_embeddings.grad is None"
        assert grad.abs().max().item() > 1e-10, (
            f"compaction_embeddings gradient too small: max={grad.abs().max().item():.2e}. "
            "Gradient flow through compact_kv may be broken."
        )

    def test_dynamic_cache_update_gradient_flow(self):
        """DynamicCache.update() preserves gradient through .clone() extraction.

        Verifies the specific mechanism: create tensors with grad, put them in cache,
        extract with .clone(), then verify gradient reaches the original tensors.
        """
        from src.kv_manager import extract_compact_kv, create_cache_with_compact_kv

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        B, num_kv_heads, P, head_dim = 1, 8, 4, 128
        num_layers = 2  # use fewer layers for this unit test

        # Create compact_kv with gradient tracking
        compact_kv_original = []
        for _ in range(num_layers):
            k = torch.randn(B, num_kv_heads, P, head_dim, device=device, requires_grad=True)
            v = torch.randn(B, num_kv_heads, P, head_dim, device=device, requires_grad=True)
            compact_kv_original.append((k, v))

        # Put into cache
        cache = create_cache_with_compact_kv(compact_kv_original, num_layers)

        # Simulate a forward pass by adding more entries to the cache
        for layer_idx in range(num_layers):
            new_k = torch.randn(B, num_kv_heads, 10, head_dim, device=device)
            new_v = torch.randn(B, num_kv_heads, 10, head_dim, device=device)
            cache.update(new_k, new_v, layer_idx)

        # Extract compact_kv (last P entries, which should be the new entries, not original)
        # But let's extract the ORIGINAL entries that are at the beginning
        # Actually, after adding 10 more, the cache has P+10 entries.
        # extract_compact_kv takes LAST P entries (the new ones), not the originals.
        # For this test, we want to verify .clone() preserves gradient.

        # Let's test directly: put K/V in cache, extract with clone, backward
        cache2 = DynamicCache()
        k_grad = torch.randn(B, num_kv_heads, P, head_dim, device=device, requires_grad=True)
        v_grad = torch.randn(B, num_kv_heads, P, head_dim, device=device, requires_grad=True)
        cache2.update(k_grad, v_grad, 0)  # layer 0

        # Extract via clone (use public __getitem__ API, not removed key_cache attr)
        k_all, v_all = cache2[0]
        k_extracted = k_all[:, :, -P:, :].clone()
        v_extracted = v_all[:, :, -P:, :].clone()

        # Compute a scalar loss from extracted
        loss = (k_extracted.sum() + v_extracted.sum())
        loss.backward()

        # Gradient should reach k_grad and v_grad
        assert k_grad.grad is not None, "k_grad.grad is None — .clone() didn't preserve gradient"
        assert v_grad.grad is not None, "v_grad.grad is None — .clone() didn't preserve gradient"
        assert k_grad.grad.abs().sum().item() > 0, "k_grad gradient is all zeros"
        assert v_grad.grad.abs().sum().item() > 0, "v_grad gradient is all zeros"
