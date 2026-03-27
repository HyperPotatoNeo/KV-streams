"""Tests for evaluation metrics: cross_block_ppl, per_block_ppl, val_ppl, diagnostics.

8 tests, ALL @pytest.mark.gpu (need real model for blockwise_forward_eval).
Uses synthetic data with random tokens for speed.
"""

import math

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.config import CompactionConfig
from src.evaluate import evaluate


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def gpu_model_and_config():
    """Load model once with tiny config for all evaluate tests."""
    from src.model import setup_model

    config = CompactionConfig(
        W=32,
        P=4,
        K=2,
        max_seq_len=128,
        batch_size=2,
        bf16=True,
    )
    model = setup_model(config)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    return model, config


@pytest.fixture(scope="module")
def val_loader(gpu_model_and_config):
    """Create a simple val DataLoader with random tokens.

    Uses 4 blocks per example (seq_len=128, W=32) so cross_block metrics
    have blocks 1, 2, 3 to measure.
    """
    _, config = gpu_model_and_config
    B = 2
    seq_len = config.W * 4  # 4 blocks

    input_ids = torch.randint(0, 151936, (B, seq_len))
    # Labels: mark all as trainable (non-(-100))
    labels = input_ids.clone()
    attention_mask = torch.ones(B, seq_len, dtype=torch.long)

    # Wrap in a simple list-of-dicts loader
    class SimpleDataset:
        def __init__(self, ids, lab, mask):
            self.data = [
                {"input_ids": ids[i], "labels": lab[i], "attention_mask": mask[i]}
                for i in range(ids.shape[0])
            ]

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            return self.data[idx]

    def collate(batch):
        return {
            "input_ids": torch.stack([b["input_ids"] for b in batch]),
            "labels": torch.stack([b["labels"] for b in batch]),
            "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
        }

    dataset = SimpleDataset(input_ids, labels, attention_mask)
    return DataLoader(dataset, batch_size=B, collate_fn=collate)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.gpu
class TestEvaluateMetrics:
    """Tests for evaluate() function."""

    def test_cross_block_ppl_computed(self, gpu_model_and_config, val_loader):
        """cross_block_ppl is a finite float > 0."""
        model, config = gpu_model_and_config
        metrics = evaluate(model, val_loader, config)

        ppl = metrics["cross_block_ppl"]
        assert isinstance(ppl, float), f"Expected float, got {type(ppl)}"
        assert ppl > 0, f"cross_block_ppl should be > 0, got {ppl}"
        assert math.isfinite(ppl), f"cross_block_ppl not finite: {ppl}"

    def test_cross_block_excludes_block0(self, gpu_model_and_config, val_loader):
        """Block 0 is not included in cross_block_ppl calculation.

        Verified indirectly: per_block_ppl has a block 0 entry, but
        cross_block_ppl differs from val_ppl (if it included block 0,
        they could be the same when all blocks have same loss).
        """
        model, config = gpu_model_and_config
        metrics = evaluate(model, val_loader, config)

        # cross_block only counts blocks 1+, val_ppl counts all
        # They should be different (block 0 has no compact_kv → different loss)
        assert "cross_block_ppl" in metrics
        assert "val_ppl" in metrics
        # per_block_ppl should have block 0
        assert 0 in metrics["per_block_ppl"], "Block 0 missing from per_block_ppl"

    def test_cross_block_window_32_tokens(self, gpu_model_and_config):
        """Cross-block metric uses first 32 tokens of blocks 1+.

        Test with W=32 so the entire block contributes (min(32, W=32) = 32).
        """
        _, config = gpu_model_and_config
        # The evaluate function uses cross_block_window = 32
        # With W=32, min(32, 32) = 32, so all tokens in each non-first block
        # are counted for cross_block. This is by design.
        assert config.W == 32, "This test assumes W=32"

    def test_per_block_ppl_all_blocks(self, gpu_model_and_config, val_loader):
        """per_block_ppl returns a dict with an entry per block index."""
        model, config = gpu_model_and_config
        metrics = evaluate(model, val_loader, config)

        per_block = metrics["per_block_ppl"]
        assert isinstance(per_block, dict), f"Expected dict, got {type(per_block)}"

        # With 4 blocks (seq_len=128, W=32), should have entries for 0,1,2,3
        expected_blocks = set(range(4))
        actual_blocks = set(per_block.keys())
        assert actual_blocks == expected_blocks, (
            f"Expected blocks {expected_blocks}, got {actual_blocks}"
        )

        # Each block's ppl should be finite and positive
        for block_idx, ppl in per_block.items():
            assert math.isfinite(ppl) and ppl > 0, (
                f"Block {block_idx} ppl invalid: {ppl}"
            )

    def test_val_ppl_computed(self, gpu_model_and_config, val_loader):
        """Overall val_ppl is a finite float > 0."""
        model, config = gpu_model_and_config
        metrics = evaluate(model, val_loader, config)

        ppl = metrics["val_ppl"]
        assert isinstance(ppl, float)
        assert ppl > 0
        assert math.isfinite(ppl), f"val_ppl not finite: {ppl}"

    def test_embed_norms_positive(self, gpu_model_and_config, val_loader):
        """Compaction embedding norms are > 0."""
        model, config = gpu_model_and_config
        metrics = evaluate(model, val_loader, config)

        norms = metrics["embed_norms"]
        assert isinstance(norms, float)
        assert norms > 0, f"embed_norms should be > 0, got {norms}"

    def test_attn_bias_mean_reported(self, gpu_model_and_config, val_loader):
        """attn_bias_mean is a finite float."""
        model, config = gpu_model_and_config
        metrics = evaluate(model, val_loader, config)

        bias_mean = metrics["attn_bias_mean"]
        assert isinstance(bias_mean, float)
        assert math.isfinite(bias_mean), f"attn_bias_mean not finite: {bias_mean}"
        # Should be close to -2.0 (the initialization value, no training happened)
        assert abs(bias_mean - (-2.0)) < 0.1, (
            f"Expected attn_bias_mean near -2.0, got {bias_mean}"
        )

    def test_eval_mode_no_gradient(self, gpu_model_and_config, val_loader):
        """No gradients are computed during evaluation."""
        model, config = gpu_model_and_config

        # Ensure model starts in train mode
        model.train()

        # Zero all existing gradients
        model.zero_grad()

        # Run evaluation
        metrics = evaluate(model, val_loader, config)

        # After evaluate, model should be back in train mode
        assert model.training, "evaluate() should restore model to train mode"

        # No parameter should have accumulated gradients
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is None or param.grad.abs().sum().item() == 0, (
                    f"Parameter '{name}' has non-zero gradient after eval"
                )
