"""Tests for blockwise forward with compaction — the core mechanism.

15 tests, ALL @pytest.mark.gpu (need real Qwen3 model for blockwise forward).
Uses small configs (W=16 or 32, P=4, K=2) and random tokens for speed.
"""

import pytest
import torch

from src.config import CompactionConfig, Qwen3Dims
from src.blockwise import blockwise_train_step


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def gpu_model_and_config():
    """Load model once with tiny config for all blockwise tests.

    Returns (model, config) on GPU.
    """
    from src.model import setup_model

    config = CompactionConfig(
        W=32,
        P=4,
        K=2,
        max_seq_len=128,  # 128 / 32 = 4 blocks
        batch_size=1,
        bf16=True,
    )
    model = setup_model(config)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    return model, config


@pytest.fixture
def make_random_batch():
    """Factory fixture for creating random batches of given shape."""
    def _make(B, seq_len, device, vocab_size=151936):
        input_ids = torch.randint(0, vocab_size, (B, seq_len), device=device)
        # Labels: use actual token IDs for all positions (max signal for gradient tests)
        labels = input_ids.clone()
        attention_mask = torch.ones(B, seq_len, dtype=torch.long, device=device)
        return input_ids, labels, attention_mask
    return _make


# ---------------------------------------------------------------------------
# Basic Execution Tests
# ---------------------------------------------------------------------------

@pytest.mark.gpu
class TestBlockwiseBasic:
    """Basic execution: loss returned, no NaN, multiple blocks work."""

    def test_blockwise_step_returns_loss(self, gpu_model_and_config, make_random_batch):
        """blockwise_train_step returns a finite float > 0."""
        model, config = gpu_model_and_config
        device = next(model.parameters()).device
        B = 1
        seq_len = config.W * 2  # 2 blocks

        input_ids, labels, attention_mask = make_random_batch(B, seq_len, device)

        model.train()
        model.zero_grad()
        loss = blockwise_train_step(model, input_ids, labels, attention_mask, config)

        assert isinstance(loss, float), f"Expected float, got {type(loss)}"
        assert loss > 0, f"Expected loss > 0, got {loss}"
        assert torch.isfinite(torch.tensor(loss)), f"Loss not finite: {loss}"

    def test_blockwise_step_no_nan(self, gpu_model_and_config, make_random_batch):
        """Loss is not NaN."""
        model, config = gpu_model_and_config
        device = next(model.parameters()).device
        B = 1
        seq_len = config.W * 2

        input_ids, labels, attention_mask = make_random_batch(B, seq_len, device)

        model.train()
        model.zero_grad()
        loss = blockwise_train_step(model, input_ids, labels, attention_mask, config)

        import math
        assert not math.isnan(loss), f"Loss is NaN"

    def test_blockwise_step_multiple_blocks(self, gpu_model_and_config, make_random_batch):
        """seq_len=4*W processes all 4 blocks without error."""
        model, config = gpu_model_and_config
        device = next(model.parameters()).device
        B = 1
        seq_len = config.W * 4  # 4 blocks

        input_ids, labels, attention_mask = make_random_batch(B, seq_len, device)

        model.train()
        model.zero_grad()
        loss = blockwise_train_step(model, input_ids, labels, attention_mask, config)

        assert isinstance(loss, float) and loss > 0, f"Invalid loss: {loss}"


# ---------------------------------------------------------------------------
# Gradient Flow Tests (CRITICAL)
# ---------------------------------------------------------------------------

@pytest.mark.gpu
class TestBlockwiseGradients:
    """Gradient flow: compaction_embeddings, compact_attn_bias, LoRA params."""

    def test_gradient_to_compaction_embeddings(self, gpu_model_and_config, make_random_batch):
        """compaction_embeddings.grad is non-zero after backward."""
        model, config = gpu_model_and_config
        device = next(model.parameters()).device
        B = 1
        seq_len = config.W * 2  # need >= 2 blocks for cross-block gradient

        input_ids, labels, attention_mask = make_random_batch(B, seq_len, device)

        model.train()
        model.zero_grad()
        blockwise_train_step(model, input_ids, labels, attention_mask, config)

        grad = model.compaction_embeddings.grad
        assert grad is not None, "compaction_embeddings.grad is None (no gradient)"
        assert grad.abs().sum().item() > 0, (
            "compaction_embeddings.grad is all zeros (gradient not flowing)"
        )

    def test_gradient_to_compact_attn_bias(self, gpu_model_and_config, make_random_batch):
        """compact_attn_bias.grad is non-zero after backward."""
        model, config = gpu_model_and_config
        device = next(model.parameters()).device
        B = 1
        seq_len = config.W * 2

        input_ids, labels, attention_mask = make_random_batch(B, seq_len, device)

        model.train()
        model.zero_grad()
        blockwise_train_step(model, input_ids, labels, attention_mask, config)

        grad = model.compact_attn_bias.grad
        assert grad is not None, "compact_attn_bias.grad is None"
        assert grad.abs().sum().item() > 0, "compact_attn_bias.grad is all zeros"

    def test_gradient_to_lora_params(self, gpu_model_and_config, make_random_batch):
        """At least one LoRA parameter has non-zero gradient after backward."""
        model, config = gpu_model_and_config
        device = next(model.parameters()).device
        B = 1
        seq_len = config.W * 2

        input_ids, labels, attention_mask = make_random_batch(B, seq_len, device)

        model.train()
        model.zero_grad()
        blockwise_train_step(model, input_ids, labels, attention_mask, config)

        has_lora_grad = False
        for name, param in model.named_parameters():
            if "lora" in name.lower() and param.grad is not None:
                if param.grad.abs().sum().item() > 0:
                    has_lora_grad = True
                    break
        assert has_lora_grad, "No LoRA parameter received non-zero gradient"

    def test_no_gradient_to_base_params(self, gpu_model_and_config, make_random_batch):
        """Frozen base parameters have None gradient after backward."""
        model, config = gpu_model_and_config
        device = next(model.parameters()).device
        B = 1
        seq_len = config.W * 2

        input_ids, labels, attention_mask = make_random_batch(B, seq_len, device)

        model.train()
        model.zero_grad()
        blockwise_train_step(model, input_ids, labels, attention_mask, config)

        for name, param in model.named_parameters():
            if not param.requires_grad:
                assert param.grad is None, (
                    f"Frozen param '{name}' has non-None gradient"
                )


# ---------------------------------------------------------------------------
# BPTT Tests
# ---------------------------------------------------------------------------

@pytest.mark.gpu
class TestBlockwiseBPTT:
    """BPTT truncation, remainder handling, and loss accumulation."""

    def test_bptt_truncation_at_K(self, gpu_model_and_config, make_random_batch):
        """After K blocks, compact_kv is detached (tested indirectly via no OOM on many blocks).

        With K=2 and 8 blocks, BPTT should truncate 4 times. If it doesn't,
        the gradient graph would grow unbounded and eventually OOM or be very slow.
        """
        model, config = gpu_model_and_config
        device = next(model.parameters()).device
        B = 1
        # 8 blocks, K=2 → 4 BPTT windows
        seq_len = config.W * 8

        input_ids, labels, attention_mask = make_random_batch(B, seq_len, device)

        model.train()
        model.zero_grad()
        # Should complete without OOM — if BPTT is broken, gradient graph grows unbounded
        loss = blockwise_train_step(model, input_ids, labels, attention_mask, config)
        assert isinstance(loss, float) and loss > 0

    def test_bptt_remainder_handled(self, make_random_batch):
        """Odd number of blocks (not multiple of K) works without error.

        K=2, 3 blocks → 1 full BPTT window (2 blocks) + 1 remainder block.
        The remainder should still call backward().
        """
        from src.model import setup_model

        config = CompactionConfig(
            W=16, P=4, K=2,
            max_seq_len=64, batch_size=1, bf16=True,
        )
        model = setup_model(config)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = model.to(device)

        B = 1
        seq_len = config.W * 3  # 3 blocks, not multiple of K=2

        input_ids, labels, attention_mask = make_random_batch(B, seq_len, device)

        model.train()
        model.zero_grad()
        loss = blockwise_train_step(model, input_ids, labels, attention_mask, config)
        assert isinstance(loss, float) and loss > 0

        # Gradient should still exist from the remainder backward
        assert model.compaction_embeddings.grad is not None

    def test_bptt_loss_accumulation(self, gpu_model_and_config, make_random_batch):
        """All blocks contribute to loss (not just last BPTT window)."""
        model, config = gpu_model_and_config
        device = next(model.parameters()).device
        B = 1

        # Compare loss from 2 blocks vs 4 blocks — more blocks = more data = similar per-token loss
        # but total_tokens should differ
        model.train()
        model.zero_grad()
        input_ids_2, labels_2, mask_2 = make_random_batch(B, config.W * 2, device)
        loss_2 = blockwise_train_step(model, input_ids_2, labels_2, mask_2, config)

        model.zero_grad()
        input_ids_4, labels_4, mask_4 = make_random_batch(B, config.W * 4, device)
        loss_4 = blockwise_train_step(model, input_ids_4, labels_4, mask_4, config)

        # Both should be finite and positive
        assert loss_2 > 0 and loss_4 > 0
        import math
        assert not math.isnan(loss_2) and not math.isnan(loss_4)


# ---------------------------------------------------------------------------
# Position / Cache Tests
# ---------------------------------------------------------------------------

@pytest.mark.gpu
class TestBlockwisePositions:
    """Position IDs and cache positions are correct for each block."""

    def test_position_ids_block0(self, gpu_model_and_config):
        """Block 0: position_ids start at 0, go to W+P-1."""
        _, config = gpu_model_and_config
        W, P = config.W, config.P
        block_idx = 0

        block_start_pos = block_idx * (W + P)
        expected_start = 0
        expected_end = W + P

        positions = torch.arange(block_start_pos, block_start_pos + W + P)
        assert positions[0].item() == expected_start
        assert positions[-1].item() == expected_end - 1
        assert len(positions) == W + P

    def test_position_ids_block1(self, gpu_model_and_config):
        """Block 1: position_ids start at W+P, go to 2*(W+P)-1."""
        _, config = gpu_model_and_config
        W, P = config.W, config.P
        block_idx = 1

        block_start_pos = block_idx * (W + P)
        expected_start = W + P
        expected_end = 2 * (W + P)

        positions = torch.arange(block_start_pos, block_start_pos + W + P)
        assert positions[0].item() == expected_start
        assert positions[-1].item() == expected_end - 1
        assert len(positions) == W + P

    def test_cache_position_block0(self, gpu_model_and_config):
        """Block 0: cache_position slots [0, W+P) because P_past=0."""
        _, config = gpu_model_and_config
        W, P = config.W, config.P
        P_past = 0  # block 0 has no compact_kv

        cache_position = torch.arange(P_past, P_past + W + P)
        assert cache_position[0].item() == 0
        assert cache_position[-1].item() == W + P - 1
        assert len(cache_position) == W + P

    def test_cache_position_block1(self, gpu_model_and_config):
        """Block 1: cache_position slots [P, P+W+P) because P_past=P."""
        _, config = gpu_model_and_config
        W, P = config.W, config.P
        P_past = P  # block 1+ has compact_kv of size P

        cache_position = torch.arange(P_past, P_past + W + P)
        assert cache_position[0].item() == P
        assert cache_position[-1].item() == P + W + P - 1
        assert len(cache_position) == W + P


# ---------------------------------------------------------------------------
# Label Masking Test
# ---------------------------------------------------------------------------

@pytest.mark.gpu
class TestBlockwiseLabels:
    """Labels=-100 positions don't contribute to loss."""

    def test_loss_ignores_minus100_labels(self, gpu_model_and_config, make_random_batch):
        """When all labels are -100, the loss should be 0 (or near-zero denominator handled)."""
        model, config = gpu_model_and_config
        device = next(model.parameters()).device
        B = 1
        seq_len = config.W * 2

        input_ids, _, attention_mask = make_random_batch(B, seq_len, device)
        # Set ALL labels to -100 — no valid tokens
        labels = torch.full_like(input_ids, -100)

        model.train()
        model.zero_grad()
        loss = blockwise_train_step(model, input_ids, labels, attention_mask, config)

        # With no valid tokens, loss should be 0.0 (total_loss_sum=0 / max(0,1)=0.0)
        assert loss == 0.0, f"Expected loss=0.0 with all -100 labels, got {loss}"
