"""Tests for model loading, LoRA application, and compaction parameters.

10 tests, ALL marked @pytest.mark.gpu because they require downloading and
loading Qwen/Qwen3-0.6B-Base weights.
"""

import pytest
import torch

from src.config import CompactionConfig, Qwen3Dims


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def gpu_model():
    """Load model once for the entire test module (expensive).

    Uses tiny compaction config but real Qwen3-0.6B-Base weights.
    """
    from src.model import setup_model

    config = CompactionConfig(
        W=16,
        P=4,
        K=2,
        max_seq_len=64,
        batch_size=1,
        bf16=True,
    )
    model = setup_model(config)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    return model


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.gpu
class TestModelLoading:
    """Tests for model setup: loading, LoRA, compaction params."""

    def test_model_loads_base(self, gpu_model):
        """Qwen/Qwen3-0.6B-Base loads without error and is on device."""
        assert gpu_model is not None
        # Verify it's a PeftModel wrapping the correct base
        assert hasattr(gpu_model, "base_model"), "Expected PeftModel wrapper"
        # Model config should be accessible
        assert gpu_model.config.num_hidden_layers == 28

    def test_attn_implementation_eager(self, gpu_model):
        """Model uses eager attention (required for custom 4D mask with bias)."""
        # Check the underlying model config
        config = gpu_model.config
        # Different transformers versions store this differently
        attn_impl = getattr(config, "_attn_implementation", None)
        if attn_impl is None:
            attn_impl = getattr(config, "attn_implementation", None)
        assert attn_impl == "eager", (
            f"Expected eager attention, got {attn_impl}. "
            "SDPA silently ignores our custom 4D mask."
        )

    def test_lora_applied(self, gpu_model):
        """LoRA modules exist on target layers (check 'lora' in named parameters)."""
        lora_param_names = [
            name for name, _ in gpu_model.named_parameters() if "lora" in name.lower()
        ]
        assert len(lora_param_names) > 0, "No LoRA parameters found"
        # Should have LoRA on q/k/v/o/gate/up/down_proj across 28 layers
        # Each layer has 7 target modules x 2 (lora_A, lora_B) = 14 LoRA params
        # 28 layers x 14 = 392 minimum (may vary with peft version)
        assert len(lora_param_names) >= 28 * 7 * 2, (
            f"Expected LoRA on all 7 target modules x 28 layers x 2 matrices, "
            f"got {len(lora_param_names)} LoRA parameters"
        )

    def test_lora_params_trainable(self, gpu_model):
        """All LoRA parameters have requires_grad=True."""
        for name, param in gpu_model.named_parameters():
            if "lora" in name.lower():
                assert param.requires_grad, (
                    f"LoRA param '{name}' has requires_grad=False"
                )

    def test_base_params_frozen(self, gpu_model):
        """Non-LoRA base model parameters have requires_grad=False."""
        frozen_count = 0
        for name, param in gpu_model.named_parameters():
            if "lora" not in name.lower() and name not in (
                "compaction_embeddings", "compact_attn_bias"
            ):
                assert not param.requires_grad, (
                    f"Base param '{name}' has requires_grad=True (should be frozen)"
                )
                frozen_count += 1
        assert frozen_count > 0, "No frozen base parameters found"

    def test_trainable_fraction(self, gpu_model):
        """Trainable params are < 5% of total parameters."""
        trainable = sum(p.numel() for p in gpu_model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in gpu_model.parameters())
        fraction = trainable / total
        assert fraction < 0.05, (
            f"Trainable fraction {fraction:.4f} ({trainable}/{total}) exceeds 5%. "
            "LoRA should keep trainable params small."
        )

    def test_compaction_embeddings_shape(self, gpu_model):
        """compaction_embeddings has shape (P, hidden_size=1024) and correct dtype."""
        dims = Qwen3Dims()
        # P=4 from our tiny config
        embed = gpu_model.compaction_embeddings
        assert embed.shape == (4, dims.hidden_size), (
            f"Expected shape (4, {dims.hidden_size}), got {embed.shape}"
        )
        assert embed.dtype == torch.bfloat16, (
            f"Expected bfloat16 dtype, got {embed.dtype}"
        )

    def test_compaction_embeddings_init(self, gpu_model):
        """compaction_embeddings initialization: mean approx 0, std approx 0.02."""
        embed = gpu_model.compaction_embeddings.data.float()
        mean = embed.mean().item()
        std = embed.std().item()
        assert abs(mean) < 0.1, (
            f"Embedding mean {mean:.4f} not within +/-0.1 of 0"
        )
        assert abs(std - 0.02) < 0.01, (
            f"Embedding std {std:.4f} not within +/-0.01 of 0.02"
        )

    def test_compact_attn_bias_shape(self, gpu_model):
        """compact_attn_bias has shape (num_q_heads=16,)."""
        dims = Qwen3Dims()
        bias = gpu_model.compact_attn_bias
        assert bias.shape == (dims.num_q_heads,), (
            f"Expected shape ({dims.num_q_heads},), got {bias.shape}"
        )

    def test_compact_attn_bias_init(self, gpu_model):
        """All compact_attn_bias values initialized to -2.0."""
        bias = gpu_model.compact_attn_bias.data
        expected = torch.full_like(bias, -2.0)
        assert torch.allclose(bias, expected, atol=1e-6), (
            f"Expected all values == -2.0, got {bias}"
        )
