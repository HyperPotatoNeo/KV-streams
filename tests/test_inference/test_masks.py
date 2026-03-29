"""Tests for inference masks: build_decode_mask and build_compaction_mask.

Key correctness property: build_compaction_mask(P, P_past, W) must produce
the same attention pattern as rows W..W+P-1 of build_4d_attention_mask(W+P, ...).
"""

import pytest
import torch
import torch.nn as nn

from src.inference.masks import build_decode_mask, build_compaction_mask
from src.attention import build_4d_attention_mask


# ---- Fixtures ----

@pytest.fixture
def device():
    return torch.device("cpu")

@pytest.fixture
def dtype():
    return torch.float32

@pytest.fixture
def bias(device, dtype):
    """Per-head bias (16 heads, like Qwen3-0.6B)."""
    return nn.Parameter(torch.full((16,), -2.0, device=device, dtype=dtype))

@pytest.fixture
def varied_bias(device, dtype):
    """Per-head bias with different values per head."""
    values = torch.linspace(-4.0, -0.5, 16, device=device, dtype=dtype)
    return nn.Parameter(values)


# ---- build_decode_mask tests ----

class TestDecodeMask:

    def test_shape(self, bias, device, dtype):
        mask = build_decode_mask(B=2, kv_len=10, P_past=4, compact_attn_bias=bias,
                                 device=device, dtype=dtype)
        assert mask.shape == (2, 16, 1, 10)

    def test_bias_on_compact_columns(self, bias, device, dtype):
        P_past = 4
        mask = build_decode_mask(B=1, kv_len=10, P_past=P_past, compact_attn_bias=bias,
                                 device=device, dtype=dtype)
        # First P_past columns should have bias value
        assert torch.allclose(mask[0, 0, 0, :P_past], torch.tensor([-2.0] * P_past))
        # Remaining columns should be 0.0
        assert torch.allclose(mask[0, 0, 0, P_past:], torch.zeros(10 - P_past))

    def test_no_compact_kv(self, bias, device, dtype):
        """When P_past=0, all columns are 0.0."""
        mask = build_decode_mask(B=1, kv_len=5, P_past=0, compact_attn_bias=bias,
                                 device=device, dtype=dtype)
        assert torch.allclose(mask, torch.zeros_like(mask))

    def test_per_head_bias(self, varied_bias, device, dtype):
        """Each head should have its own bias value on compact_kv columns."""
        P_past = 3
        mask = build_decode_mask(B=1, kv_len=8, P_past=P_past,
                                 compact_attn_bias=varied_bias, device=device, dtype=dtype)
        for h in range(16):
            expected_bias = varied_bias[h].item()
            for col in range(P_past):
                assert abs(mask[0, h, 0, col].item() - expected_bias) < 1e-6, \
                    f"Head {h}, col {col}: expected {expected_bias}, got {mask[0, h, 0, col].item()}"

    def test_batch_independence(self, bias, device, dtype):
        """Mask should be identical across batch dimension."""
        mask = build_decode_mask(B=3, kv_len=10, P_past=4, compact_attn_bias=bias,
                                 device=device, dtype=dtype)
        assert torch.equal(mask[0], mask[1])
        assert torch.equal(mask[1], mask[2])

    def test_dtype_preserved(self, device):
        """Output dtype should match input dtype."""
        for dt in [torch.float32, torch.bfloat16]:
            b = nn.Parameter(torch.full((4,), -2.0, device=device, dtype=dt))
            mask = build_decode_mask(B=1, kv_len=5, P_past=2, compact_attn_bias=b,
                                     device=device, dtype=dt)
            assert mask.dtype == dt


# ---- build_compaction_mask tests ----

class TestCompactionMask:

    def test_shape(self, bias, device, dtype):
        mask = build_compaction_mask(B=1, P=4, P_past=4, W=8,
                                     compact_attn_bias=bias, device=device, dtype=dtype)
        assert mask.shape == (1, 16, 4, 4 + 8 + 4)  # (1, 16, P, P_past+W+P)

    def test_compact_kv_columns_have_bias(self, bias, device, dtype):
        P_past = 4
        mask = build_compaction_mask(B=1, P=4, P_past=P_past, W=8,
                                     compact_attn_bias=bias, device=device, dtype=dtype)
        # All P queries should see compact_kv columns with bias
        for q in range(4):
            for col in range(P_past):
                assert abs(mask[0, 0, q, col].item() - (-2.0)) < 1e-6

    def test_text_columns_fully_visible(self, bias, device, dtype):
        P_past = 4
        W = 8
        mask = build_compaction_mask(B=1, P=4, P_past=P_past, W=W,
                                     compact_attn_bias=bias, device=device, dtype=dtype)
        # ALL text columns visible to ALL compaction queries
        for q in range(4):
            for col in range(P_past, P_past + W):
                assert mask[0, 0, q, col].item() == 0.0, \
                    f"Query {q}, text col {col - P_past}: should be 0.0"

    def test_compaction_columns_causal(self, bias, device, dtype):
        P_past = 4
        W = 8
        P = 4
        min_val = torch.finfo(dtype).min
        mask = build_compaction_mask(B=1, P=P, P_past=P_past, W=W,
                                     compact_attn_bias=bias, device=device, dtype=dtype)
        # Compaction tokens: causal among themselves
        comp_start = P_past + W
        for q in range(P):
            for k in range(P):
                val = mask[0, 0, q, comp_start + k].item()
                if k <= q:
                    assert val == 0.0, f"q={q}, k={k}: should be visible (0.0), got {val}"
                else:
                    assert val == min_val, f"q={q}, k={k}: should be masked (-inf), got {val}"

    def test_no_compact_kv(self, bias, device, dtype):
        """When P_past=0, no compact_kv columns at all."""
        W = 8
        P = 4
        mask = build_compaction_mask(B=1, P=P, P_past=0, W=W,
                                     compact_attn_bias=bias, device=device, dtype=dtype)
        assert mask.shape == (1, 16, P, 0 + W + P)
        # Text columns still visible
        for q in range(P):
            for col in range(W):
                assert mask[0, 0, q, col].item() == 0.0

    def test_per_head_bias(self, varied_bias, device, dtype):
        P_past = 2
        mask = build_compaction_mask(B=1, P=3, P_past=P_past, W=4,
                                     compact_attn_bias=varied_bias, device=device, dtype=dtype)
        for h in range(16):
            expected = varied_bias[h].item()
            for q in range(3):
                for col in range(P_past):
                    assert abs(mask[0, h, q, col].item() - expected) < 1e-6


# ---- Critical: Equivalence with training mask ----

class TestMaskEquivalence:
    """build_compaction_mask output must match rows W..W+P-1 of build_4d_attention_mask."""

    @pytest.mark.parametrize("W,P,P_past", [
        (8, 4, 0),   # Block 0
        (8, 4, 4),   # Block 1+
        (16, 2, 2),  # Small P
        (4, 8, 8),   # Large P relative to W
        (128, 16, 16),  # Training-scale W=128
    ])
    def test_compaction_mask_matches_training(self, W, P, P_past, device, dtype):
        """Compaction mask rows must equal training mask rows W..W+P-1."""
        bias = nn.Parameter(torch.full((16,), -1.5, device=device, dtype=dtype))

        # Training mask: (1, 16, W+P, P_past+W+P)
        training_mask = build_4d_attention_mask(
            1, W + P, P_past + W + P, P_past, bias, device, dtype
        )

        # Compaction mask: (1, 16, P, P_past+W+P)
        compaction_mask = build_compaction_mask(
            1, P, P_past, W, bias, device, dtype
        )

        # Extract rows W..W+P-1 from training mask
        training_comp_rows = training_mask[:, :, W:W + P, :]

        # They should be identical
        assert torch.allclose(compaction_mask, training_comp_rows, atol=1e-6), \
            f"Mismatch for W={W}, P={P}, P_past={P_past}: " \
            f"max diff = {(compaction_mask - training_comp_rows).abs().max().item()}"

    @pytest.mark.parametrize("W,P,P_past", [
        (8, 4, 0),
        (8, 4, 4),
    ])
    def test_with_varied_bias(self, W, P, P_past, device, dtype):
        """Test equivalence with per-head varied bias."""
        bias = nn.Parameter(torch.linspace(-3.0, -0.5, 16, device=device, dtype=dtype))

        training_mask = build_4d_attention_mask(
            1, W + P, P_past + W + P, P_past, bias, device, dtype
        )
        compaction_mask = build_compaction_mask(
            1, P, P_past, W, bias, device, dtype
        )

        training_comp_rows = training_mask[:, :, W:W + P, :]
        assert torch.allclose(compaction_mask, training_comp_rows, atol=1e-6)
