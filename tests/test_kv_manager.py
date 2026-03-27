"""Tests for kv_manager functions: extract, create, detach, random.

11 CPU-only tests using fake tensors — no real model needed.

DynamicCache API used:
  - cache[layer_idx] -> (keys, values) tuple
  - cache.update(k, v, layer_idx) -> adds KV to that layer
  - DynamicCache() creates empty cache
"""

import pytest
import torch
from transformers import DynamicCache

from src.kv_manager import (
    create_cache_with_compact_kv,
    detach_compact_kv,
    extract_compact_kv,
    random_compact_kv,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Dimensions matching Qwen3-0.6B (scaled down for speed where noted)
NUM_LAYERS = 4      # Smaller than real 28 for speed
NUM_KV_HEADS = 8
HEAD_DIM = 128
B = 2
P = 4
W = 16


def _make_compact_kv(
    B: int = B,
    P: int = P,
    num_layers: int = NUM_LAYERS,
    requires_grad: bool = False,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Create a compact_kv list with known values."""
    compact_kv = []
    for i in range(num_layers):
        k = torch.randn(B, NUM_KV_HEADS, P, HEAD_DIM, requires_grad=requires_grad)
        v = torch.randn(B, NUM_KV_HEADS, P, HEAD_DIM, requires_grad=requires_grad)
        compact_kv.append((k, v))
    return compact_kv


def _make_full_cache(
    B: int = B,
    seq_len: int = W + P,
    num_layers: int = NUM_LAYERS,
    requires_grad: bool = False,
) -> DynamicCache:
    """Create a DynamicCache with seq_len entries per layer, simulating a forward pass."""
    cache = DynamicCache()
    for layer_idx in range(num_layers):
        k = torch.randn(B, NUM_KV_HEADS, seq_len, HEAD_DIM, requires_grad=requires_grad)
        v = torch.randn(B, NUM_KV_HEADS, seq_len, HEAD_DIM, requires_grad=requires_grad)
        cache.update(k, v, layer_idx)
    return cache


# ---------------------------------------------------------------------------
# Construction Tests
# ---------------------------------------------------------------------------

class TestCacheConstruction:
    """Tests for creating DynamicCache objects."""

    def test_empty_cache_creation(self):
        """DynamicCache() creates an empty cache with no layers."""
        cache = DynamicCache()
        # An empty DynamicCache should have zero sequence length
        assert cache.get_seq_length() == 0

    def test_create_from_compact_kv_seq_len(self):
        """Pre-seeded cache has correct seq_len P per layer."""
        compact_kv = _make_compact_kv()
        cache = create_cache_with_compact_kv(compact_kv, NUM_LAYERS)

        assert cache.get_seq_length() == P, (
            f"Expected seq_len={P}, got {cache.get_seq_length()}"
        )

    def test_create_preserves_shapes(self):
        """KV entries in pre-seeded cache have shape (B, num_kv_heads, P, head_dim)."""
        compact_kv = _make_compact_kv()
        cache = create_cache_with_compact_kv(compact_kv, NUM_LAYERS)

        for layer_idx in range(NUM_LAYERS):
            k, v = cache[layer_idx]
            expected_shape = (B, NUM_KV_HEADS, P, HEAD_DIM)
            assert k.shape == expected_shape, (
                f"Layer {layer_idx} key shape: expected {expected_shape}, got {k.shape}"
            )
            assert v.shape == expected_shape, (
                f"Layer {layer_idx} value shape: expected {expected_shape}, got {v.shape}"
            )


# ---------------------------------------------------------------------------
# Extraction Tests
# ---------------------------------------------------------------------------

class TestExtraction:
    """Tests for extract_compact_kv — extracting last P entries from cache."""

    def test_extract_last_P_entries(self):
        """Extracts exactly the last P entries from each layer."""
        cache = _make_full_cache(seq_len=W + P)
        extracted = extract_compact_kv(cache, P, NUM_LAYERS)

        assert len(extracted) == NUM_LAYERS
        for layer_idx in range(NUM_LAYERS):
            k_ext, v_ext = extracted[layer_idx]
            k_full, v_full = cache[layer_idx]

            # Extracted should match the last P positions
            assert torch.equal(k_ext, k_full[:, :, -P:, :]), (
                f"Layer {layer_idx}: extracted keys don't match last P entries"
            )
            assert torch.equal(v_ext, v_full[:, :, -P:, :]), (
                f"Layer {layer_idx}: extracted values don't match last P entries"
            )

    def test_extract_shape_correct(self):
        """Extracted tensors have shape (B, num_kv_heads, P, head_dim)."""
        cache = _make_full_cache(seq_len=W + P)
        extracted = extract_compact_kv(cache, P, NUM_LAYERS)

        expected_shape = (B, NUM_KV_HEADS, P, HEAD_DIM)
        for layer_idx in range(NUM_LAYERS):
            k, v = extracted[layer_idx]
            assert k.shape == expected_shape, f"Key shape: {k.shape}"
            assert v.shape == expected_shape, f"Value shape: {v.shape}"

    def test_extract_uses_clone(self):
        """Modifying cache after extract does not change extracted values.

        extract_compact_kv uses .clone() to create independent tensors.
        If we modify the original cache tensor in-place, the extracted
        tensors should be unaffected.
        """
        cache = _make_full_cache(seq_len=W + P)
        extracted = extract_compact_kv(cache, P, NUM_LAYERS)

        # Save a copy of extracted values
        k_before = extracted[0][0].clone()
        v_before = extracted[0][1].clone()

        # Modify the original cache by adding more entries (simulates next block)
        new_k = torch.zeros(B, NUM_KV_HEADS, W, HEAD_DIM)
        new_v = torch.zeros(B, NUM_KV_HEADS, W, HEAD_DIM)
        cache.update(new_k, new_v, 0)

        # Extracted values should be unchanged
        assert torch.equal(extracted[0][0], k_before), (
            "Extracted keys changed after cache modification — clone() not used?"
        )
        assert torch.equal(extracted[0][1], v_before), (
            "Extracted values changed after cache modification — clone() not used?"
        )

    def test_extract_preserves_gradient(self):
        """Extracted tensors have grad_fn when cache was built with requires_grad.

        .clone() preserves the gradient graph (CloneBackward), which is
        essential for BPTT through compact_kv across blocks.
        """
        cache = _make_full_cache(seq_len=W + P, requires_grad=True)
        extracted = extract_compact_kv(cache, P, NUM_LAYERS)

        for layer_idx in range(NUM_LAYERS):
            k, v = extracted[layer_idx]
            assert k.grad_fn is not None, (
                f"Layer {layer_idx} key has no grad_fn — gradient chain broken"
            )
            assert v.grad_fn is not None, (
                f"Layer {layer_idx} value has no grad_fn — gradient chain broken"
            )


# ---------------------------------------------------------------------------
# Detach Tests
# ---------------------------------------------------------------------------

class TestDetach:
    """Tests for detach_compact_kv — breaking gradient chain at BPTT boundary."""

    def test_detach_removes_grad_fn(self):
        """Detached compact_kv tensors have no grad_fn."""
        # Start with tensors that have gradient history
        compact_kv = _make_compact_kv(requires_grad=True)
        # Apply an operation to create grad_fn (not just leaf tensors)
        compact_kv_with_grad = [(k * 2, v * 2) for k, v in compact_kv]

        detached = detach_compact_kv(compact_kv_with_grad)

        for layer_idx in range(NUM_LAYERS):
            k, v = detached[layer_idx]
            assert k.grad_fn is None, (
                f"Layer {layer_idx} key still has grad_fn after detach"
            )
            assert v.grad_fn is None, (
                f"Layer {layer_idx} value still has grad_fn after detach"
            )
            assert not k.requires_grad, (
                f"Layer {layer_idx} key still requires_grad after detach"
            )

    def test_detach_values_unchanged(self):
        """Detached values are numerically identical to originals."""
        compact_kv = _make_compact_kv()
        detached = detach_compact_kv(compact_kv)

        for layer_idx in range(NUM_LAYERS):
            k_orig, v_orig = compact_kv[layer_idx]
            k_det, v_det = detached[layer_idx]
            assert torch.equal(k_det, k_orig), (
                f"Layer {layer_idx}: detached key values differ from original"
            )
            assert torch.equal(v_det, v_orig), (
                f"Layer {layer_idx}: detached value values differ from original"
            )


# ---------------------------------------------------------------------------
# Random Compact KV Tests
# ---------------------------------------------------------------------------

class TestRandomCompactKV:
    """Tests for random_compact_kv — Condition D baseline."""

    def test_random_compact_kv_shape(self):
        """Generated tensors have correct shape (B, num_kv_heads, P, head_dim)."""
        result = random_compact_kv(
            B=B, P=P, num_layers=NUM_LAYERS,
            num_kv_heads=NUM_KV_HEADS, head_dim=HEAD_DIM,
            device=torch.device("cpu"), dtype=torch.float32,
        )

        assert len(result) == NUM_LAYERS
        expected_shape = (B, NUM_KV_HEADS, P, HEAD_DIM)
        for layer_idx in range(NUM_LAYERS):
            k, v = result[layer_idx]
            assert k.shape == expected_shape, f"Key shape: {k.shape}"
            assert v.shape == expected_shape, f"Value shape: {v.shape}"

    def test_random_compact_kv_independent(self):
        """Different calls produce different values (random, not deterministic)."""
        result1 = random_compact_kv(
            B=B, P=P, num_layers=NUM_LAYERS,
            num_kv_heads=NUM_KV_HEADS, head_dim=HEAD_DIM,
            device=torch.device("cpu"), dtype=torch.float32,
        )
        result2 = random_compact_kv(
            B=B, P=P, num_layers=NUM_LAYERS,
            num_kv_heads=NUM_KV_HEADS, head_dim=HEAD_DIM,
            device=torch.device("cpu"), dtype=torch.float32,
        )

        # At least one layer should differ (probability of all-same is ~0)
        any_different = False
        for layer_idx in range(NUM_LAYERS):
            k1, v1 = result1[layer_idx]
            k2, v2 = result2[layer_idx]
            if not torch.equal(k1, k2) or not torch.equal(v1, v2):
                any_different = True
                break

        assert any_different, "Two random_compact_kv calls produced identical results"
