"""
Evaluation metrics for KV Self-Compaction Phase 2.

Computes compaction-specific metrics using blockwise_forward_eval:
  - cross_block_ppl: perplexity on first 32 tokens of blocks 1+ (PRIMARY metric)
  - per_block_ppl: {block_idx: ppl} loss breakdown by block
  - val_ppl: overall perplexity on all valid tokens
  - embed_norms: mean L2 norm of compaction embeddings
  - attn_bias_mean: mean attention bias value
"""

import math
from collections import defaultdict

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.config import CompactionConfig
from src.blockwise import blockwise_forward_eval


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    val_loader: DataLoader,
    config: CompactionConfig,
) -> dict:
    """Compute all evaluation metrics using blockwise forward.

    Args:
        model: PeftModel wrapping Qwen3ForCausalLM with compaction params.
        val_loader: DataLoader yielding dicts with input_ids, labels, attention_mask.
        config: CompactionConfig with W, P, etc.

    Returns:
        dict with keys:
          cross_block_ppl: float — perplexity on first 32 tokens of blocks 1+
          per_block_ppl: dict[int, float] — per-block perplexity
          val_ppl: float — overall perplexity on all valid tokens
          embed_norms: float — mean L2 norm of compaction embeddings
          attn_bias_mean: float — mean attention bias value
    """
    model.eval()

    W = config.W
    cross_block_window = 32  # first 32 tokens of each non-first block

    # Accumulators
    cross_block_loss = 0.0
    cross_block_tokens = 0
    per_block_loss: dict[int, float] = defaultdict(float)
    per_block_tokens: dict[int, int] = defaultdict(int)
    total_loss = 0.0
    total_tokens = 0

    for batch in val_loader:
        input_ids = batch["input_ids"]
        labels = batch["labels"]
        attention_mask = batch["attention_mask"]

        # Move to model device if needed
        device = next(model.parameters()).device
        input_ids = input_ids.to(device)
        labels = labels.to(device)
        attention_mask = attention_mask.to(device)

        B, seq_len = input_ids.shape
        num_blocks = seq_len // W

        # Get logits from blockwise forward — (B, seq_len, vocab_size)
        logits = blockwise_forward_eval(model, input_ids, attention_mask, config)

        # Per-block metric computation
        for block_idx in range(num_blocks):
            start = block_idx * W
            end = start + W

            block_logits = logits[:, start:end, :]   # (B, W, vocab_size)
            block_labels = labels[:, start:end]       # (B, W)

            valid = block_labels != -100
            num_valid = valid.sum().item()
            if num_valid == 0:
                continue

            # Compute loss on valid positions
            # Flatten for cross_entropy: select only valid positions
            loss = F.cross_entropy(
                block_logits[valid],          # (num_valid, vocab_size)
                block_labels[valid],          # (num_valid,)
                reduction="sum",
            )
            loss_val = loss.item()

            # Per-block accumulation
            per_block_loss[block_idx] += loss_val
            per_block_tokens[block_idx] += num_valid

            # Total accumulation
            total_loss += loss_val
            total_tokens += num_valid

            # Cross-block: first min(32, W) tokens of blocks 1+
            if block_idx > 0:
                cb_end = min(cross_block_window, W)
                cb_logits = block_logits[:, :cb_end, :]   # (B, cb_end, vocab_size)
                cb_labels = block_labels[:, :cb_end]       # (B, cb_end)
                cb_valid = cb_labels != -100
                cb_num_valid = cb_valid.sum().item()
                if cb_num_valid > 0:
                    cb_loss = F.cross_entropy(
                        cb_logits[cb_valid],
                        cb_labels[cb_valid],
                        reduction="sum",
                    )
                    cross_block_loss += cb_loss.item()
                    cross_block_tokens += cb_num_valid

    # Compute perplexities
    cross_block_ppl = (
        math.exp(cross_block_loss / cross_block_tokens)
        if cross_block_tokens > 0
        else float("nan")
    )

    per_block_ppl = {}
    for block_idx in sorted(per_block_loss.keys()):
        if per_block_tokens[block_idx] > 0:
            per_block_ppl[block_idx] = math.exp(
                per_block_loss[block_idx] / per_block_tokens[block_idx]
            )
        else:
            per_block_ppl[block_idx] = float("nan")

    val_ppl = (
        math.exp(total_loss / total_tokens)
        if total_tokens > 0
        else float("nan")
    )

    # Diagnostic metrics from compaction parameters
    embed_norms = model.compaction_embeddings.norm(dim=-1).mean().item()
    attn_bias_mean = model.compact_attn_bias.mean().item()

    model.train()

    return {
        "cross_block_ppl": cross_block_ppl,
        "per_block_ppl": per_block_ppl,
        "val_ppl": val_ppl,
        "embed_norms": embed_norms,
        "attn_bias_mean": attn_bias_mean,
    }


def print_metrics(metrics: dict, config: CompactionConfig) -> None:
    """Print evaluation metrics in a readable format.

    Args:
        metrics: dict returned by evaluate().
        config: CompactionConfig for display context.
    """
    print("=" * 60)
    print("Compaction Evaluation Metrics")
    print(f"  W={config.W}, P={config.P}, K={config.K}, condition={config.condition}")
    print("=" * 60)

    print(f"\nval_ppl:          {metrics['val_ppl']:.4f}")
    print(f"cross_block_ppl:  {metrics['cross_block_ppl']:.4f}")
    print(f"embed_norms:      {metrics['embed_norms']:.4f}")
    print(f"attn_bias_mean:   {metrics['attn_bias_mean']:.4f}")

    if metrics["per_block_ppl"]:
        print("\nper_block_ppl:")
        for block_idx in sorted(metrics["per_block_ppl"].keys()):
            ppl = metrics["per_block_ppl"][block_idx]
            marker = " *" if block_idx == 0 else ""
            print(f"  block {block_idx}: {ppl:.4f}{marker}")
        if 0 in metrics["per_block_ppl"]:
            print("  (* block 0 has no compact_kv from prior block)")

    print("=" * 60)
