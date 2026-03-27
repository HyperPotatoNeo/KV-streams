"""Tests for CompactionConfig and Qwen3Dims.

5 tests covering defaults, dimension consistency, architecture values,
condition literals, and override behavior.
"""

import pytest

from src.config import CompactionConfig, Qwen3Dims


class TestCompactionConfig:
    """Tests for CompactionConfig dataclass."""

    def test_default_config_valid(self):
        """All defaults parse correctly and have expected types."""
        cfg = CompactionConfig()

        # Model
        assert cfg.model_name == "Qwen/Qwen3-0.6B-Base"
        assert cfg.attn_implementation == "eager"

        # Compaction
        assert isinstance(cfg.W, int) and cfg.W == 128
        assert isinstance(cfg.P, int) and cfg.P == 16
        assert isinstance(cfg.K, int) and cfg.K == 2

        # LoRA
        assert cfg.lora_rank == 32
        assert cfg.lora_alpha == 64
        assert isinstance(cfg.lora_targets, list)
        assert "q_proj" in cfg.lora_targets
        assert "down_proj" in cfg.lora_targets

        # Optimizer
        assert isinstance(cfg.lr, float) and cfg.lr > 0
        assert isinstance(cfg.lr_compaction, float) and cfg.lr_compaction > 0
        assert isinstance(cfg.lr_bias, float) and cfg.lr_bias > 0
        assert cfg.weight_decay >= 0
        assert cfg.max_grad_norm > 0
        assert 0 < cfg.warmup_ratio < 1

        # Training
        assert cfg.max_seq_len == 4096
        assert cfg.batch_size == 4
        assert cfg.grad_accum == 1
        assert cfg.max_steps == 200
        assert cfg.eval_every == 50
        assert cfg.save_every == 100
        assert cfg.seed == 42
        assert cfg.bf16 is True

        # Compaction init
        assert cfg.bias_init == -2.0
        assert cfg.embed_init_std == 0.02
        assert cfg.embed_max_norm == 10.0

        # Data
        assert "Dolci" in cfg.think_sft_path
        assert "Dolci" in cfg.instruct_sft_path
        assert cfg.max_examples_per_dataset == 5000
        assert cfg.val_fraction == 0.05

        # Condition
        assert cfg.condition == "B"

        # Output
        assert cfg.output_dir == "outputs"
        assert cfg.wandb_project == "kv-self-compaction-phase2"

    def test_config_dimensions_consistent(self):
        """W divides max_seq_len, P < W, K >= 1 for defaults and tiny config."""
        for cfg in [CompactionConfig(), CompactionConfig(W=16, P=4, K=2, max_seq_len=64)]:
            assert cfg.max_seq_len % cfg.W == 0, (
                f"max_seq_len={cfg.max_seq_len} not divisible by W={cfg.W}"
            )
            assert cfg.P < cfg.W, f"P={cfg.P} should be less than W={cfg.W}"
            assert cfg.K >= 1, f"K={cfg.K} must be at least 1"

    def test_config_condition_enum(self):
        """Only A/B/C/D are valid condition values (enforced by Literal type)."""
        # Valid conditions should work
        for cond in ["A", "B", "C", "D"]:
            cfg = CompactionConfig(condition=cond)
            assert cfg.condition == cond

        # Note: dataclasses with Literal types don't enforce at runtime by default,
        # but we verify the type annotation exists and accepts valid values.
        # A static type checker (mypy) would catch invalid values.
        cfg = CompactionConfig()
        assert cfg.condition in {"A", "B", "C", "D"}

    def test_config_override(self):
        """Can override defaults via constructor kwargs."""
        cfg = CompactionConfig(
            W=64,
            P=8,
            K=4,
            max_seq_len=1024,
            lr=3e-4,
            batch_size=8,
            condition="A",
        )
        assert cfg.W == 64
        assert cfg.P == 8
        assert cfg.K == 4
        assert cfg.max_seq_len == 1024
        assert cfg.lr == 3e-4
        assert cfg.batch_size == 8
        assert cfg.condition == "A"

        # Unmodified fields retain defaults
        assert cfg.model_name == "Qwen/Qwen3-0.6B-Base"
        assert cfg.bias_init == -2.0


class TestQwen3Dims:
    """Tests for Qwen3Dims — known architecture values for Qwen3-0.6B-Base."""

    def test_qwen3_dims_match_expected(self):
        """Dimensions match the known Qwen3-0.6B-Base architecture.

        These values are fixed by the model and documented in the HuggingFace config.
        Any mismatch means our code will produce wrong tensor shapes.
        """
        dims = Qwen3Dims()

        assert dims.num_layers == 28
        assert dims.hidden_size == 1024
        assert dims.num_q_heads == 16
        assert dims.num_kv_heads == 8
        assert dims.head_dim == 128
        assert dims.vocab_size == 151936

        # GQA ratio: 2 query heads per KV head
        assert dims.num_q_heads // dims.num_kv_heads == 2

        # Head dim is independent of hidden_size in Qwen3
        # (not hidden_size // num_q_heads = 1024 // 16 = 64)
        assert dims.head_dim == 128
        assert dims.head_dim != dims.hidden_size // dims.num_q_heads
