"""Training loop for KV Self-Compaction Phase 2.

Main entry point for training Qwen3-0.6B-Base with compaction.
Supports all 4 experimental conditions:
  A: Full context SFT (standard HF forward, no compaction)
  B: KV Self-Compaction (learned compact_kv, the method under test)
  C: Truncation baseline (last W tokens only)
  D: Random compact_kv (like B but compact_kv is random noise, not learned)

Single GPU for Phase 2a. DDP support planned for Phase 2c.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.config import CompactionConfig
from src.model import setup_model
from src.data import load_data, analyze_sequence_lengths, CompactionDataset
from src.blockwise import blockwise_train_step
from src.evaluate import evaluate, print_metrics

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Collate function
# ---------------------------------------------------------------------------

def collate_fn(batch: list[dict[str, torch.Tensor]], W: int) -> dict[str, torch.Tensor]:
    """Collate variable-length sequences by right-padding to the max length in the batch.

    Each example is already padded to a multiple of W by data.py, but different
    examples may have different lengths (e.g., 512 vs 1024). This function finds
    the max length in the batch, right-pads shorter sequences to that length
    (also a multiple of W), and stacks into a single tensor.

    Args:
        batch: List of dicts, each with input_ids, labels, attention_mask tensors.
        W: Block size. The padded length is rounded up to the next multiple of W.

    Returns:
        Dict with stacked input_ids (B, max_len), labels (B, max_len),
        attention_mask (B, max_len) tensors.
    """
    # Find max length in batch, round up to multiple of W
    lengths = [ex["input_ids"].shape[0] for ex in batch]
    max_len = max(lengths)
    # Should already be a multiple of W from data.py, but ensure it
    max_len = ((max_len + W - 1) // W) * W

    input_ids_list = []
    labels_list = []
    attention_mask_list = []

    for ex in batch:
        seq_len = ex["input_ids"].shape[0]
        pad_amount = max_len - seq_len

        if pad_amount > 0:
            # Pad input_ids with 0 (will be masked out anyway)
            input_ids_list.append(
                torch.cat([ex["input_ids"], torch.zeros(pad_amount, dtype=torch.long)])
            )
            # Pad labels with -100 (ignored in loss)
            labels_list.append(
                torch.cat([ex["labels"], torch.full((pad_amount,), -100, dtype=torch.long)])
            )
            # Pad attention_mask with 0
            attention_mask_list.append(
                torch.cat([ex["attention_mask"], torch.zeros(pad_amount, dtype=torch.long)])
            )
        else:
            input_ids_list.append(ex["input_ids"])
            labels_list.append(ex["labels"])
            attention_mask_list.append(ex["attention_mask"])

    return {
        "input_ids": torch.stack(input_ids_list),
        "labels": torch.stack(labels_list),
        "attention_mask": torch.stack(attention_mask_list),
    }


# ---------------------------------------------------------------------------
# Checkpoint save / load
# ---------------------------------------------------------------------------

def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    step: int,
    config: CompactionConfig,
    output_dir: str,
) -> str:
    """Save a training checkpoint.

    Saves only the trainable parameters (LoRA + compaction) plus optimizer
    and scheduler state. The frozen base model is not saved.

    Args:
        model: PeftModelForCausalLM with compaction params.
        optimizer: AdamW optimizer with 3 param groups.
        scheduler: LR scheduler.
        step: Current training step.
        config: CompactionConfig (serialized as dict).
        output_dir: Directory to save checkpoint in.

    Returns:
        Path to the saved checkpoint file.
    """
    ckpt_dir = Path(output_dir) / f"checkpoint-{step}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Save trainable model state dict (LoRA + compaction params)
    trainable_state = {
        name: param.data.cpu()
        for name, param in model.named_parameters()
        if param.requires_grad
    }
    # Also save compaction_embeddings and compact_attn_bias explicitly
    # (they should be in trainable_state already, but ensure it)
    trainable_state["compaction_embeddings"] = model.compaction_embeddings.data.cpu()
    trainable_state["compact_attn_bias"] = model.compact_attn_bias.data.cpu()

    ckpt_path = ckpt_dir / "checkpoint.pt"
    torch.save(
        {
            "model_state_dict": trainable_state,
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "step": step,
            "config": vars(config),
        },
        ckpt_path,
    )

    logger.info("Saved checkpoint at step %d to %s", step, ckpt_path)
    return str(ckpt_path)


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
) -> int:
    """Load a training checkpoint and restore model/optimizer/scheduler state.

    Args:
        path: Path to checkpoint .pt file.
        model: PeftModelForCausalLM (must match architecture).
        optimizer: AdamW optimizer (must have same param groups).
        scheduler: LR scheduler.

    Returns:
        The training step to resume from.
    """
    ckpt = torch.load(path, map_location="cpu", weights_only=False)

    # Load trainable parameters
    model_state = ckpt["model_state_dict"]
    current_state = dict(model.named_parameters())
    for name, param in current_state.items():
        if param.requires_grad and name in model_state:
            param.data.copy_(model_state[name].to(param.device))

    # Restore compaction params explicitly (in case naming differs)
    if "compaction_embeddings" in model_state:
        model.compaction_embeddings.data.copy_(
            model_state["compaction_embeddings"].to(model.compaction_embeddings.device)
        )
    if "compact_attn_bias" in model_state:
        model.compact_attn_bias.data.copy_(
            model_state["compact_attn_bias"].to(model.compact_attn_bias.device)
        )

    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    scheduler.load_state_dict(ckpt["scheduler_state_dict"])

    step = ckpt["step"]
    logger.info("Loaded checkpoint from step %d at %s", step, path)
    return step


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train(config: CompactionConfig, resume_from: str | None = None) -> None:
    """Main training function for KV Self-Compaction.

    Sets up model, optimizer, data, and runs the training loop with:
    - 3 param groups (LoRA, compaction_embeddings, compact_attn_bias)
    - Cosine LR schedule with warmup
    - Condition routing (A/B/C/D)
    - Gradient accumulation
    - Embedding norm clamping
    - Periodic eval and checkpointing

    Args:
        config: CompactionConfig with all hyperparameters.
        resume_from: Optional path to checkpoint .pt file to resume training.
    """
    # ---- Setup logging ----
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    logger.info("=" * 60)
    logger.info("KV Self-Compaction Phase 2 — Training")
    logger.info("  Condition: %s", config.condition)
    logger.info("  W=%d, P=%d, K=%d", config.W, config.P, config.K)
    logger.info("  batch_size=%d, grad_accum=%d, max_steps=%d", config.batch_size, config.grad_accum, config.max_steps)
    logger.info("  lr=%g, lr_compaction=%g, lr_bias=%g", config.lr, config.lr_compaction, config.lr_bias)
    logger.info("=" * 60)

    # ---- Seed ----
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # ---- Tokenizer ----
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model_name, trust_remote_code=True)
    logger.info("Loaded tokenizer: %s", config.model_name)

    # ---- Model ----
    model = setup_model(config)
    model = model.to(device)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info("Model loaded. Trainable: %d / %d params (%.2f%%)",
                trainable_params, total_params, 100 * trainable_params / total_params)

    # ---- Optimizer: 3 param groups ----
    lora_params = []
    for name, param in model.named_parameters():
        if param.requires_grad and "lora" in name.lower():
            lora_params.append(param)

    param_groups = [
        {"params": lora_params, "lr": config.lr, "weight_decay": config.weight_decay},
        {"params": [model.compaction_embeddings], "lr": config.lr_compaction, "weight_decay": 0.0},
        {"params": [model.compact_attn_bias], "lr": config.lr_bias, "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(param_groups)
    logger.info("Optimizer: AdamW with 3 param groups (LoRA: %d params, embeddings: %d, bias: %d)",
                sum(p.numel() for p in lora_params),
                model.compaction_embeddings.numel(),
                model.compact_attn_bias.numel())

    # ---- LR Scheduler: cosine with warmup ----
    total_steps = config.max_steps
    warmup_steps = int(total_steps * config.warmup_ratio)

    # Use transformers scheduler if available, otherwise manual implementation
    try:
        from transformers import get_cosine_schedule_with_warmup
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )
    except ImportError:
        # Manual cosine schedule with warmup
        def lr_lambda(current_step: int) -> float:
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    logger.info("Scheduler: cosine with %d warmup steps / %d total steps", warmup_steps, total_steps)

    # ---- Resume from checkpoint ----
    start_step = 0
    if resume_from is not None:
        start_step = load_checkpoint(resume_from, model, optimizer, scheduler)
        logger.info("Resuming from step %d", start_step)

    # ---- Data ----
    logger.info("Loading data...")
    train_dataset, val_dataset = load_data(config, tokenizer)
    logger.info("Train: %d examples, Val: %d examples", len(train_dataset), len(val_dataset))

    # Build collate_fn with W bound
    W = config.W

    def _collate(batch):
        return collate_fn(batch, W)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=_collate,
        num_workers=0,  # data is already in memory
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=_collate,
        num_workers=0,
        drop_last=False,
    )

    # ---- Output directory ----
    output_dir = Path(config.output_dir) / f"condition_{config.condition}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    config_path = output_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(vars(config), f, indent=2)
    logger.info("Config saved to %s", config_path)

    # ---- Training loop ----
    model.train()
    step = start_step
    epoch = 0
    accum_loss = 0.0  # accumulated loss for logging across grad_accum micro-steps
    accum_tokens = 0  # count of micro-steps with valid tokens
    optimizer.zero_grad(set_to_none=True)

    logger.info("Starting training...")
    t_start = time.time()

    while step < config.max_steps:
        epoch += 1
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            # ---- Route to correct condition ----
            if config.condition == "A":
                # Full context SFT — standard HF forward through PeftModel
                outputs = model(
                    input_ids=input_ids,
                    labels=labels,
                    attention_mask=attention_mask,
                )
                micro_loss = outputs.loss
                # Scale loss for gradient accumulation
                (micro_loss / config.grad_accum).backward()
                accum_loss += micro_loss.detach().item()
                accum_tokens += 1

            elif config.condition in ("B", "D", "E"):
                # Blockwise compaction (B=learned, D=random compact_kv)
                # blockwise_train_step calls backward() internally (BPTT).
                # PyTorch backward() accumulates into .grad, so with grad_accum > 1,
                # gradients from multiple micro-batches add up naturally.
                # We scale all gradients once before the optimizer step (below).
                micro_loss_val = blockwise_train_step(
                    model, input_ids, labels, attention_mask, config,
                )
                accum_loss += micro_loss_val
                accum_tokens += 1

            elif config.condition == "C":
                # Truncation baseline — last W tokens only
                W_trunc = config.W
                trunc_input_ids = input_ids[:, -W_trunc:]
                trunc_labels = labels[:, -W_trunc:]
                trunc_attention_mask = attention_mask[:, -W_trunc:]
                outputs = model(
                    input_ids=trunc_input_ids,
                    labels=trunc_labels,
                    attention_mask=trunc_attention_mask,
                )
                micro_loss = outputs.loss
                (micro_loss / config.grad_accum).backward()
                accum_loss += micro_loss.detach().item()
                accum_tokens += 1

            # ---- Gradient accumulation: step every grad_accum micro-batches ----
            if accum_tokens >= config.grad_accum:
                # For conditions B/D, blockwise_train_step called backward() internally
                # without scaling. Scale accumulated gradients now.
                if config.condition in ("B", "D", "E") and config.grad_accum > 1:
                    for p in model.parameters():
                        if p.grad is not None:
                            p.grad.div_(config.grad_accum)

                # Max-norm clamp compaction embeddings BEFORE optimizer step
                # (clamp the parameter data, not gradients — prevents embedding drift)
                with torch.no_grad():
                    norms = model.compaction_embeddings.data.norm(dim=-1, keepdim=True)
                    scale = torch.clamp(config.embed_max_norm / norms, max=1.0)
                    model.compaction_embeddings.data.mul_(scale)

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)

                # Optimizer step
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

                step += 1
                avg_loss = accum_loss / accum_tokens

                # ---- Logging every 10 steps ----
                if step % 10 == 0:
                    elapsed = time.time() - t_start
                    bias_mean = model.compact_attn_bias.mean().item()
                    embed_norm = model.compaction_embeddings.norm(dim=-1).mean().item()
                    current_lr = scheduler.get_last_lr()[0]
                    logger.info(
                        "step=%d/%d | loss=%.4f | bias_mean=%.3f | embed_norm=%.3f | "
                        "lr=%.2e | elapsed=%.1fs",
                        step, config.max_steps, avg_loss, bias_mean, embed_norm,
                        current_lr, elapsed,
                    )

                # ---- Eval ----
                if step % config.eval_every == 0:
                    logger.info("Evaluating at step %d...", step)
                    metrics = evaluate(model, val_loader, config)
                    print_metrics(metrics, config)

                    # Log metrics to file
                    metrics_path = output_dir / "metrics.jsonl"
                    metrics_record = {
                        "step": step,
                        "train_loss": avg_loss,
                        **{k: v for k, v in metrics.items() if k != "per_block_ppl"},
                        "per_block_ppl": {str(k): v for k, v in metrics.get("per_block_ppl", {}).items()},
                    }
                    with open(metrics_path, "a") as f:
                        f.write(json.dumps(metrics_record) + "\n")

                    model.train()

                # ---- Checkpoint ----
                if step % config.save_every == 0:
                    save_checkpoint(model, optimizer, scheduler, step, config, str(output_dir))

                # Reset accumulators
                accum_loss = 0.0
                accum_tokens = 0

                # Check max_steps
                if step >= config.max_steps:
                    break

        if step >= config.max_steps:
            break

    # ---- Final eval + checkpoint ----
    elapsed = time.time() - t_start
    logger.info("Training complete. %d steps in %.1f seconds (%.1f sec/step)",
                step, elapsed, elapsed / max(step, 1))

    logger.info("Final evaluation...")
    metrics = evaluate(model, val_loader, config)
    print_metrics(metrics, config)

    # Save final checkpoint
    save_checkpoint(model, optimizer, scheduler, step, config, str(output_dir))

    # Save final metrics
    metrics_path = output_dir / "final_metrics.json"
    final_record = {
        "step": step,
        "elapsed_seconds": elapsed,
        **{k: v for k, v in metrics.items() if k != "per_block_ppl"},
        "per_block_ppl": {str(k): v for k, v in metrics.get("per_block_ppl", {}).items()},
    }
    with open(metrics_path, "w") as f:
        json.dump(final_record, f, indent=2)
    logger.info("Final metrics saved to %s", metrics_path)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments, overriding CompactionConfig defaults."""
    parser = argparse.ArgumentParser(
        description="KV Self-Compaction Phase 2 Training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Data analysis mode
    parser.add_argument(
        "--analyze_data", action="store_true",
        help="Run sequence length analysis and exit without training.",
    )

    # Condition
    parser.add_argument(
        "--condition", type=str, choices=["A", "B", "C", "D", "E"], default=None,
        help="Experimental condition.",
    )

    # Model
    parser.add_argument("--model_name", type=str, default=None)

    # Compaction
    parser.add_argument("--W", type=int, default=None, help="Block size in tokens.")
    parser.add_argument("--P", type=int, default=None, help="Number of compaction tokens per block.")
    parser.add_argument("--K", type=int, default=None, help="BPTT depth (blocks before truncation).")

    # LoRA
    parser.add_argument("--lora_rank", type=int, default=None)
    parser.add_argument("--lora_alpha", type=int, default=None)

    # Optimizer
    parser.add_argument("--lr", type=float, default=None, help="LoRA learning rate.")
    parser.add_argument("--lr_compaction", type=float, default=None, help="Compaction embeddings LR.")
    parser.add_argument("--lr_bias", type=float, default=None, help="Attention bias LR.")
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--max_grad_norm", type=float, default=None)
    parser.add_argument("--warmup_ratio", type=float, default=None)

    # Training
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--grad_accum", type=int, default=None, help="Gradient accumulation steps.")
    parser.add_argument("--max_seq_len", type=int, default=None)
    parser.add_argument("--eval_every", type=int, default=None)
    parser.add_argument("--save_every", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)

    # Compaction init
    parser.add_argument("--bias_init", type=float, default=None)
    parser.add_argument("--embed_init_std", type=float, default=None)
    parser.add_argument("--embed_max_norm", type=float, default=None)

    # Data
    parser.add_argument("--max_examples_per_dataset", type=int, default=None)
    parser.add_argument("--val_fraction", type=float, default=None)

    # Output
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--wandb_project", type=str, default=None)

    # Checkpoint resume
    parser.add_argument("--resume_from", type=str, default=None, help="Path to checkpoint .pt file to resume from.")

    return parser.parse_args()


def main() -> None:
    """Entry point: parse CLI args, build config, and train."""
    args = parse_args()

    # Build config with defaults, then override from CLI
    config = CompactionConfig()

    # Override config fields from CLI args (only non-None values)
    for field_name in vars(config):
        cli_val = getattr(args, field_name, None)
        if cli_val is not None:
            setattr(config, field_name, cli_val)

    # ---- Analyze data mode ----
    if args.analyze_data:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        )
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(config.model_name, trust_remote_code=True)
        analyze_sequence_lengths(config, tokenizer)
        return

    # ---- Train ----
    train(config, resume_from=args.resume_from)


if __name__ == "__main__":
    main()
