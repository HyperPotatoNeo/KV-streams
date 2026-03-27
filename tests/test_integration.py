"""Integration tests for KV Self-Compaction Phase 2.

7 tests, ALL @pytest.mark.gpu. Test end-to-end training, conditions, checkpointing.
Uses tiny configs with random tokens for speed (no dataset download).
"""

import math
import os
import tempfile

import pytest
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.config import CompactionConfig, Qwen3Dims
from src.blockwise import blockwise_train_step, blockwise_forward_eval
from src.evaluate import evaluate


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_synthetic_data(B, seq_len, num_examples=16):
    """Create a list of dicts mimicking CompactionDataset output."""
    data = []
    for _ in range(num_examples):
        ids = torch.randint(0, 151936, (seq_len,))
        data.append({
            "input_ids": ids,
            "labels": ids.clone(),
            "attention_mask": torch.ones(seq_len, dtype=torch.long),
        })
    return data


def collate_fn(batch):
    return {
        "input_ids": torch.stack([b["input_ids"] for b in batch]),
        "labels": torch.stack([b["labels"] for b in batch]),
        "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
    }


def make_loader(data, batch_size):
    from torch.utils.data import Dataset

    class ListDataset(Dataset):
        def __init__(self, items):
            self.items = items
        def __len__(self):
            return len(self.items)
        def __getitem__(self, idx):
            return self.items[idx]

    return DataLoader(ListDataset(data), batch_size=batch_size,
                      shuffle=True, collate_fn=collate_fn, drop_last=True)


def run_mini_training(config, model=None, num_steps=10, return_losses=False):
    """Run a minimal training loop and return (final_loss, model).

    If return_losses=True, returns (losses_list, model).
    """
    from src.model import setup_model

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if model is None:
        model = setup_model(config)
        model = model.to(device)

    # Build optimizer
    lora_params = [p for n, p in model.named_parameters()
                   if p.requires_grad and "lora" in n.lower()]
    param_groups = [
        {"params": lora_params, "lr": config.lr, "weight_decay": config.weight_decay},
        {"params": [model.compaction_embeddings], "lr": config.lr_compaction, "weight_decay": 0.0},
        {"params": [model.compact_attn_bias], "lr": config.lr_bias, "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(param_groups)

    # Synthetic data
    seq_len = config.W * 4  # 4 blocks per sequence
    data = make_synthetic_data(B=1, seq_len=seq_len, num_examples=max(32, num_steps * 2))
    loader = make_loader(data, config.batch_size)

    model.train()
    losses = []
    step = 0

    for batch in loader:
        if step >= num_steps:
            break

        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        optimizer.zero_grad(set_to_none=True)

        if config.condition in ("B", "D"):
            loss_val = blockwise_train_step(model, input_ids, labels, attention_mask, config)
        elif config.condition == "A":
            outputs = model(input_ids=input_ids, labels=labels, attention_mask=attention_mask)
            loss_val = outputs.loss.item()
            outputs.loss.backward()
        elif config.condition == "C":
            W = config.W
            outputs = model(
                input_ids=input_ids[:, -W:],
                labels=labels[:, -W:],
                attention_mask=attention_mask[:, -W:],
            )
            loss_val = outputs.loss.item()
            outputs.loss.backward()

        # Gradient clipping and optimizer step
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
        optimizer.step()

        losses.append(loss_val)
        step += 1

    if return_losses:
        return losses, model
    return losses[-1] if losses else 0.0, model


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.gpu
class TestIntegration:
    """End-to-end integration tests."""

    def test_end_to_end_10_steps(self):
        """Full training: 10 steps on tiny data. Loss should decrease over time."""
        config = CompactionConfig(
            W=32, P=4, K=2,
            max_seq_len=128,
            batch_size=1,
            max_steps=10,
            lr=3e-4,
            lr_compaction=6e-4,
            lr_bias=0.1,
            bf16=True,
            condition="B",
        )
        losses, _ = run_mini_training(config, num_steps=10, return_losses=True)

        assert len(losses) == 10
        # Check that at least one of the later losses is less than the initial loss
        initial = losses[0]
        later = min(losses[5:])  # best loss in second half
        assert later < initial, (
            f"Loss did not decrease: initial={initial:.4f}, best_later={later:.4f}. "
            f"All losses: {[f'{l:.4f}' for l in losses]}"
        )

    def test_condition_B_produces_cross_block_metric(self):
        """Condition B training produces valid cross_block_ppl."""
        config = CompactionConfig(
            W=32, P=4, K=2,
            max_seq_len=128,
            batch_size=1,
            max_steps=5,
            bf16=True,
            condition="B",
        )
        _, model = run_mini_training(config, num_steps=5)

        # Build a val loader with synthetic data
        seq_len = config.W * 4
        data = make_synthetic_data(B=1, seq_len=seq_len, num_examples=4)
        val_loader = make_loader(data, config.batch_size)

        metrics = evaluate(model, val_loader, config)
        ppl = metrics["cross_block_ppl"]
        assert 1.0 < ppl < 1e6, f"cross_block_ppl={ppl} out of sane range"

    def test_condition_A_standard_forward(self):
        """Condition A (full context SFT) uses standard HF forward without blockwise."""
        config = CompactionConfig(
            W=32, P=4, K=2,
            max_seq_len=128,
            batch_size=1,
            max_steps=3,
            bf16=True,
            condition="A",
        )
        loss, _ = run_mini_training(config, num_steps=3)
        assert loss is not None and not math.isnan(loss) and loss > 0

    def test_blockwise_matches_full_context_block0(self):
        """Block 0 of blockwise (no compact_kv) should approximately match
        full-context forward for the same tokens. Validates bypass forward
        produces consistent results with HF's built-in forward."""
        from src.model import setup_model, embed_tokens, get_lm_head, get_rotary_embeddings
        from src.attention import build_4d_attention_mask, forward_layers
        from transformers.cache_utils import DynamicCache

        config = CompactionConfig(
            W=32, P=4, K=2,
            max_seq_len=128,
            batch_size=1,
            bf16=True,
        )
        model = setup_model(config)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        model.eval()

        # Create random input for just 1 block
        B = 1
        W = config.W
        input_ids = torch.randint(0, 151936, (B, W), device=device)

        with torch.no_grad():
            # Full-context forward via HF
            hf_out = model(input_ids=input_ids)
            hf_logits = hf_out.logits  # (B, W, vocab)

            # Blockwise block 0 forward
            text_embeds = embed_tokens(model, input_ids)
            P = config.P
            comp_embeds = model.compaction_embeddings.unsqueeze(0).expand(B, -1, -1)
            block_embeds = torch.cat([text_embeds, comp_embeds], dim=1)

            position_ids = torch.arange(0, W + P, device=device).unsqueeze(0).expand(B, -1)
            cache_position = torch.arange(0, W + P, device=device)
            past_cache = DynamicCache()
            attn_mask = build_4d_attention_mask(
                B, W + P, W + P, 0,
                model.compact_attn_bias, device, next(model.parameters()).dtype,
            )
            position_embeddings = get_rotary_embeddings(model, block_embeds, position_ids)
            hidden, _ = forward_layers(
                model, block_embeds, attn_mask, position_embeddings,
                past_cache, cache_position,
            )
            bw_logits = get_lm_head(model)(hidden[:, :W, :]).float()

        # Block 0 has no compact_kv, so the only difference is our mask construction
        # vs HF's internal mask. They should be close (eager attention is deterministic).
        hf_logits_f = hf_logits.float()
        # Compare using cosine similarity on the logit vectors
        cos_sim = F.cosine_similarity(
            hf_logits_f.reshape(-1, hf_logits_f.shape[-1]),
            bw_logits.reshape(-1, bw_logits.shape[-1]),
            dim=-1,
        ).mean().item()

        assert cos_sim > 0.95, (
            f"Block 0 logit cosine similarity {cos_sim:.4f} too low. "
            "Bypass forward may not match HF forward."
        )

    def test_checkpoint_save_load_roundtrip(self):
        """Save checkpoint, load into fresh model, verify parameters match."""
        from src.model import setup_model
        from src.train import save_checkpoint, load_checkpoint

        config = CompactionConfig(
            W=32, P=4, K=2,
            max_seq_len=128,
            batch_size=1,
            max_steps=5,
            bf16=True,
        )

        # Train for a few steps
        _, model = run_mini_training(config, num_steps=5)

        # Build optimizer + scheduler for save
        lora_params = [p for n, p in model.named_parameters()
                       if p.requires_grad and "lora" in n.lower()]
        optimizer = torch.optim.AdamW([
            {"params": lora_params, "lr": config.lr},
            {"params": [model.compaction_embeddings], "lr": config.lr_compaction},
            {"params": [model.compact_attn_bias], "lr": config.lr_bias},
        ])
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: 1.0)

        with tempfile.TemporaryDirectory() as tmpdir:
            # Save
            ckpt_path = save_checkpoint(model, optimizer, scheduler, 5, config, tmpdir)

            # Load into fresh model
            model2 = setup_model(config)
            device = next(model.parameters()).device
            model2 = model2.to(device)

            lora_params2 = [p for n, p in model2.named_parameters()
                            if p.requires_grad and "lora" in n.lower()]
            optimizer2 = torch.optim.AdamW([
                {"params": lora_params2, "lr": config.lr},
                {"params": [model2.compaction_embeddings], "lr": config.lr_compaction},
                {"params": [model2.compact_attn_bias], "lr": config.lr_bias},
            ])
            scheduler2 = torch.optim.lr_scheduler.LambdaLR(optimizer2, lambda s: 1.0)

            step = load_checkpoint(ckpt_path, model2, optimizer2, scheduler2)
            assert step == 5

            # Verify compaction params match
            assert torch.allclose(
                model.compaction_embeddings.data,
                model2.compaction_embeddings.data,
                atol=1e-6,
            ), "compaction_embeddings mismatch after load"

            assert torch.allclose(
                model.compact_attn_bias.data,
                model2.compact_attn_bias.data,
                atol=1e-6,
            ), "compact_attn_bias mismatch after load"

    def test_eval_during_training(self):
        """Evaluation runs without error during a training loop."""
        config = CompactionConfig(
            W=32, P=4, K=2,
            max_seq_len=128,
            batch_size=1,
            max_steps=6,
            eval_every=3,
            bf16=True,
            condition="B",
        )
        _, model = run_mini_training(config, num_steps=6)

        # Run evaluation
        seq_len = config.W * 4
        data = make_synthetic_data(B=1, seq_len=seq_len, num_examples=4)
        val_loader = make_loader(data, config.batch_size)

        metrics = evaluate(model, val_loader, config)

        # All keys should be present
        for key in ["cross_block_ppl", "per_block_ppl", "val_ppl", "embed_norms", "attn_bias_mean"]:
            assert key in metrics, f"Missing metric key: {key}"

    def test_compaction_params_change_during_training(self):
        """Compaction embeddings and bias values change after training steps."""
        from src.model import setup_model

        config = CompactionConfig(
            W=32, P=4, K=2,
            max_seq_len=128,
            batch_size=1,
            max_steps=10,
            lr=3e-4,
            lr_compaction=6e-4,
            lr_bias=0.1,
            bf16=True,
            condition="B",
        )
        model = setup_model(config)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = model.to(device)

        # Record initial values
        initial_embed = model.compaction_embeddings.data.clone()
        initial_bias = model.compact_attn_bias.data.clone()

        # Train
        _, model = run_mini_training(config, model=model, num_steps=10)

        # Verify params changed
        assert not torch.allclose(model.compaction_embeddings.data, initial_embed, atol=1e-6), (
            "compaction_embeddings did not change after 10 training steps"
        )
        assert not torch.allclose(model.compact_attn_bias.data, initial_bias, atol=1e-6), (
            "compact_attn_bias did not change after 10 training steps"
        )
