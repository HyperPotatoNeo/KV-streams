"""Tests for attention mask construction and forward_layers.

14 tests covering:
- Mask shape (3 tests)
- Causal correctness (4 tests)
- Bias injection (4 tests)
- Forward layers (3 tests, 2 require GPU)
"""

import pytest
import torch
import torch.nn as nn

from src.attention import build_4d_attention_mask, forward_layers
from src.config import Qwen3Dims


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_bias(num_heads: int, value: float = -2.0, dtype=torch.float32) -> nn.Parameter:
    """Create a compact_attn_bias parameter for testing."""
    return nn.Parameter(torch.full((num_heads,), value, dtype=dtype))


def _min_val(dtype=torch.float32) -> float:
    """Return the -inf sentinel for the given dtype."""
    return torch.finfo(dtype).min


# ---------------------------------------------------------------------------
# Mask Shape Tests
# ---------------------------------------------------------------------------

class TestMaskShape:
    """Tests for the output shape of build_4d_attention_mask."""

    def test_mask_shape_block0(self, tiny_config, qwen3_dims):
        """Block 0: P_past=0, mask is (B, num_q_heads, W+P, W+P).

        For block 0, there is no compact_kv from a previous block, so P_past=0.
        kv_len = 0 + W + P = W + P, query_len = W + P.
        """
        W, P = tiny_config.W, tiny_config.P
        B = 2
        num_heads = qwen3_dims.num_q_heads  # 16
        bias = _make_bias(num_heads)

        query_len = W + P
        kv_len = W + P  # P_past=0
        mask = build_4d_attention_mask(B, query_len, kv_len, 0, bias, torch.device("cpu"), torch.float32)

        assert mask.shape == (B, num_heads, W + P, W + P)

    def test_mask_shape_block1(self, tiny_config, qwen3_dims):
        """Block 1+: P_past=P, mask is (B, num_q_heads, W+P, P+W+P).

        For block >= 1, compact_kv from the previous block is prepended,
        so kv_len = P + W + P.
        """
        W, P = tiny_config.W, tiny_config.P
        B = 2
        num_heads = qwen3_dims.num_q_heads

        query_len = W + P
        kv_len = P + W + P  # P_past=P
        bias = _make_bias(num_heads)
        mask = build_4d_attention_mask(B, query_len, kv_len, P, bias, torch.device("cpu"), torch.float32)

        assert mask.shape == (B, num_heads, W + P, P + W + P)

    def test_mask_dtype_matches(self, tiny_config, qwen3_dims):
        """Output dtype matches the requested dtype."""
        W, P = tiny_config.W, tiny_config.P
        num_heads = qwen3_dims.num_q_heads

        for dtype in [torch.float32, torch.bfloat16, torch.float16]:
            bias = _make_bias(num_heads, dtype=dtype)
            mask = build_4d_attention_mask(
                1, W + P, W + P, 0, bias, torch.device("cpu"), dtype
            )
            assert mask.dtype == dtype, f"Expected {dtype}, got {mask.dtype}"


# ---------------------------------------------------------------------------
# Causal Correctness Tests (use small W=8, P=2 for clarity)
# ---------------------------------------------------------------------------

class TestCausalCorrectness:
    """Verify the causal structure of the mask within the current-block region.

    Using W=8, P=2 so the mask is small and easy to reason about.
    """

    @pytest.fixture
    def small_mask_block0(self):
        """Build mask for block 0 with W=8, P=2, P_past=0."""
        W, P, B = 8, 2, 1
        num_heads = 16
        bias = _make_bias(num_heads, value=-2.0)
        query_len = W + P  # 10
        kv_len = W + P     # 10 (P_past=0)
        mask = build_4d_attention_mask(B, query_len, kv_len, 0, bias, torch.device("cpu"), torch.float32)
        return mask, W, P

    @pytest.fixture
    def small_mask_block1(self):
        """Build mask for block 1 with W=8, P=2, P_past=2."""
        W, P, B = 8, 2, 1
        num_heads = 16
        bias = _make_bias(num_heads, value=-2.0)
        P_past = P  # 2
        query_len = W + P      # 10
        kv_len = P_past + W + P  # 12
        mask = build_4d_attention_mask(B, query_len, kv_len, P_past, bias, torch.device("cpu"), torch.float32)
        return mask, W, P, P_past

    def test_causal_no_future_attention(self, small_mask_block0):
        """For each query q, mask[q, k > q] == -inf in the block region.

        A query at position q in the block should not attend to any future
        position k > q within the current block (standard causal constraint).
        """
        mask, W, P = small_mask_block0
        min_val = _min_val()
        query_len = W + P  # 10

        # Check batch=0, head=0 (causal structure is the same across all heads)
        m = mask[0, 0]  # (query_len, kv_len)
        for q in range(query_len):
            for k in range(q + 1, query_len):
                assert m[q, k].item() == min_val, (
                    f"query {q} should not attend to future position {k}, "
                    f"but mask value is {m[q, k].item()}"
                )

    def test_causal_self_attention_allowed(self, small_mask_block0):
        """mask[q, q] == 0.0 — each position can attend to itself.

        In block 0 with P_past=0, KV position q corresponds to query position q.
        The self-attention value should be 0.0 (no masking, no bias).
        """
        mask, W, P = small_mask_block0
        query_len = W + P

        m = mask[0, 0]
        for q in range(query_len):
            assert m[q, q].item() == 0.0, (
                f"Position {q} cannot attend to itself: mask[{q},{q}]={m[q, q].item()}"
            )

    def test_causal_past_attention_allowed(self, small_mask_block0):
        """mask[q, k] == 0.0 for k < q — can attend to all past positions in block.

        Within the current block (P_past=0), every query q should be able to attend
        to every earlier position k < q with mask value 0.0.
        """
        mask, W, P = small_mask_block0
        query_len = W + P

        m = mask[0, 0]
        for q in range(query_len):
            for k in range(q):
                assert m[q, k].item() == 0.0, (
                    f"query {q} should attend to past position {k}, "
                    f"but mask[{q},{k}]={m[q, k].item()}"
                )

    def test_last_position_sees_everything(self, small_mask_block1):
        """The last query position can attend to all KV positions.

        In block 1: kv_len = P_past + W + P. The last query (q=W+P-1) should
        see compact_kv (with bias) and all current-block positions (with 0.0).
        """
        mask, W, P, P_past = small_mask_block1
        min_val = _min_val()
        last_q = W + P - 1

        m = mask[0, 0]  # (query_len, kv_len)
        kv_len = P_past + W + P

        # Compact_kv columns should NOT be -inf (they have bias value)
        for k in range(P_past):
            assert m[last_q, k].item() != min_val, (
                f"Last query should see compact_kv at position {k}"
            )

        # Current block columns: all should be 0.0 (last query sees everything)
        for k in range(P_past, kv_len):
            assert m[last_q, k].item() == 0.0, (
                f"Last query should see current block position {k}, "
                f"but mask value is {m[last_q, k].item()}"
            )


# ---------------------------------------------------------------------------
# Bias Tests
# ---------------------------------------------------------------------------

class TestBiasInjection:
    """Verify learnable bias is applied correctly to compact_kv columns."""

    def test_bias_on_compact_columns(self):
        """Columns 0:P_past carry the bias value, not 0.0 or -inf."""
        W, P, B = 8, 2, 1
        num_heads = 16
        bias_val = -2.0
        bias = _make_bias(num_heads, value=bias_val)
        P_past = P
        query_len = W + P
        kv_len = P_past + W + P

        mask = build_4d_attention_mask(B, query_len, kv_len, P_past, bias, torch.device("cpu"), torch.float32)

        # All compact_kv columns should have the bias value
        compact_cols = mask[0, :, :, :P_past]  # (num_heads, query_len, P_past)
        assert torch.allclose(compact_cols, torch.tensor(bias_val))

    def test_bias_value_matches_param(self):
        """The exact bias value from the parameter appears in compact_kv columns."""
        W, P, B = 8, 2, 1
        num_heads = 4
        bias_val = -3.5  # Non-default to verify it's actually used
        bias = _make_bias(num_heads, value=bias_val)
        P_past = P
        query_len = W + P
        kv_len = P_past + W + P

        mask = build_4d_attention_mask(B, query_len, kv_len, P_past, bias, torch.device("cpu"), torch.float32)

        for h in range(num_heads):
            for q in range(query_len):
                for k in range(P_past):
                    actual = mask[0, h, q, k].item()
                    assert actual == pytest.approx(bias_val, abs=1e-6), (
                        f"head={h}, q={q}, k={k}: expected {bias_val}, got {actual}"
                    )

    def test_no_bias_on_block_columns(self):
        """Current-block columns (P_past:) have only 0.0 or -inf, never the bias value."""
        W, P, B = 8, 2, 1
        num_heads = 16
        bias_val = -2.0
        bias = _make_bias(num_heads, value=bias_val)
        P_past = P
        query_len = W + P
        kv_len = P_past + W + P
        min_val = _min_val()

        mask = build_4d_attention_mask(B, query_len, kv_len, P_past, bias, torch.device("cpu"), torch.float32)

        block_cols = mask[0, :, :, P_past:]  # (num_heads, query_len, W+P)
        for h in range(block_cols.shape[0]):
            for q in range(block_cols.shape[1]):
                for k in range(block_cols.shape[2]):
                    val = block_cols[h, q, k].item()
                    assert val == 0.0 or val == min_val, (
                        f"Block column head={h}, q={q}, k={k}: expected 0.0 or -inf, got {val}"
                    )

    def test_bias_zero_when_no_past(self):
        """P_past=0 means no compact_kv columns, so no bias is applied.

        The mask should only contain 0.0 (attendable) and -inf (masked) values.
        """
        W, P, B = 8, 2, 1
        num_heads = 16
        bias_val = -2.0
        bias = _make_bias(num_heads, value=bias_val)
        query_len = W + P
        kv_len = W + P  # P_past=0

        mask = build_4d_attention_mask(B, query_len, kv_len, 0, bias, torch.device("cpu"), torch.float32)
        min_val = _min_val()

        # Every value should be exactly 0.0 or -inf
        unique_vals = set()
        for val in mask.flatten().tolist():
            unique_vals.add(val)

        for val in unique_vals:
            assert val == 0.0 or val == min_val, (
                f"With P_past=0, mask should only have 0.0 or -inf, found {val}"
            )


# ---------------------------------------------------------------------------
# Forward Layers Tests (GPU required for 2 of 3)
# ---------------------------------------------------------------------------

class TestForwardLayers:
    """Tests for forward_layers which bypasses HF's Qwen3Model.forward()."""

    @pytest.mark.gpu
    def test_forward_layers_output_shape(self):
        """Output hidden states are (B, T, 1024) where T = query_len.

        Requires GPU with the real Qwen3 model loaded.
        """
        from src.config import CompactionConfig, Qwen3Dims
        from src.model import setup_model, get_inner_model
        from transformers import DynamicCache

        config = CompactionConfig(W=16, P=4)
        dims = Qwen3Dims()
        model = setup_model(config)
        device = torch.device("cuda")
        model = model.to(device)
        model.eval()

        B = 1
        W, P = config.W, config.P
        query_len = W + P
        dtype = torch.bfloat16

        # Create random hidden states
        hidden = torch.randn(B, query_len, dims.hidden_size, device=device, dtype=dtype)

        # Build mask (block 0, no past)
        bias = model.compact_attn_bias
        mask = build_4d_attention_mask(B, query_len, query_len, 0, bias, device, dtype)

        # Position embeddings
        position_ids = torch.arange(query_len, device=device).unsqueeze(0)
        pos_emb = get_inner_model(model).rotary_emb(hidden, position_ids)

        # Cache
        cache = DynamicCache()
        cache_position = torch.arange(query_len, device=device)

        with torch.no_grad():
            out, _ = forward_layers(model, hidden, mask, pos_emb, cache, cache_position)

        assert out.shape == (B, query_len, dims.hidden_size)

    @pytest.mark.gpu
    def test_forward_layers_cache_populated(self):
        """After forward, the cache has entries for all layers.

        Each layer's cache should have seq_len == query_len KV entries.
        """
        from src.config import CompactionConfig, Qwen3Dims
        from src.model import setup_model, get_inner_model
        from transformers import DynamicCache

        config = CompactionConfig(W=16, P=4)
        dims = Qwen3Dims()
        model = setup_model(config)
        device = torch.device("cuda")
        model = model.to(device)
        model.eval()

        B = 1
        W, P = config.W, config.P
        query_len = W + P
        dtype = torch.bfloat16

        hidden = torch.randn(B, query_len, dims.hidden_size, device=device, dtype=dtype)
        bias = model.compact_attn_bias
        mask = build_4d_attention_mask(B, query_len, query_len, 0, bias, device, dtype)
        position_ids = torch.arange(query_len, device=device).unsqueeze(0)
        pos_emb = get_inner_model(model).rotary_emb(hidden, position_ids)
        cache = DynamicCache()
        cache_position = torch.arange(query_len, device=device)

        with torch.no_grad():
            _, updated_cache = forward_layers(model, hidden, mask, pos_emb, cache, cache_position)

        # Cache should have entries for every layer
        for layer_idx in range(dims.num_layers):
            k, v = updated_cache[layer_idx]
            assert k.shape == (B, dims.num_kv_heads, query_len, dims.head_dim), (
                f"Layer {layer_idx} key shape mismatch: {k.shape}"
            )
            assert v.shape == (B, dims.num_kv_heads, query_len, dims.head_dim), (
                f"Layer {layer_idx} value shape mismatch: {v.shape}"
            )

    def test_position_ids_2d_assertion(self):
        """get_rotary_embeddings raises AssertionError for 1D position_ids.

        position_ids must be 2D (B, seq_len). Passing a 1D tensor should fail
        with a clear error, not silently produce wrong RoPE embeddings.
        """
        from src.model import get_rotary_embeddings

        # We need a mock-like object to trigger the assertion.
        # The assertion happens in get_rotary_embeddings before any model call,
        # so we just need to pass a 1D tensor.
        class FakeModel:
            """Minimal fake model to reach the assertion."""
            class _base_model:
                class _model:
                    class _model:
                        class rotary_emb:
                            @staticmethod
                            def __call__(h, p):
                                pass  # pragma: no cover
                    model = _model()
                model = _model()
            base_model = _base_model()

        fake = FakeModel()
        hidden = torch.randn(1, 10, 1024)
        pos_1d = torch.arange(10)  # 1D — should fail

        with pytest.raises(AssertionError, match="position_ids must be 2D"):
            get_rotary_embeddings(fake, hidden, pos_1d)
