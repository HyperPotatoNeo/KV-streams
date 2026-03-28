"""Training loop for KV Self-Compaction Phase 2.

Main entry point for training Qwen3-0.6B-Base with compaction.
Supports conditions A/B/C/D/E. Supports DDP via torchrun (manual gradient all-reduce).

DDP approach: No DDP wrapper. Each rank runs forward/backward independently on its data
shard, then we manually all-reduce gradients before the optimizer step. This avoids
complications with DDP's .module indirection and blockwise_train_step's internal backward()
calls (BPTT every K blocks triggers backward multiple times per micro-batch).
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from src.config import CompactionConfig
from src.model import setup_model
from src.data import load_data, analyze_sequence_lengths, CompactionDataset
from src.blockwise import blockwise_train_step
from src.evaluate import evaluate, print_metrics

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# DDP helpers
# ---------------------------------------------------------------------------

def setup_distributed() -> tuple[int, int, int]:
    """Initialize distributed training if launched via torchrun.

    Returns (rank, world_size, local_rank). If not distributed, returns (0, 1, 0).
    """
    if "RANK" not in os.environ:
        return 0, 1, 0

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    from datetime import timedelta
    dist.init_process_group(backend="nccl", timeout=timedelta(hours=2))
    torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank


def all_reduce_gradients(model: nn.Module, world_size: int) -> None:
    """Manually all-reduce gradients across ranks (average)."""
    for p in model.parameters():
        if p.requires_grad and p.grad is not None:
            dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
            p.grad.div_(world_size)


# ---------------------------------------------------------------------------
# Collate function
# ---------------------------------------------------------------------------

def collate_fn(batch: list[dict[str, torch.Tensor]], W: int, pad_token_id: int = 151643) -> dict[str, torch.Tensor]:
    """Collate variable-length sequences by right-padding to the max length in the batch.

    Each example is already padded to a multiple of W by data.py, but different
    examples may have different lengths (e.g., 512 vs 1024). This function finds
    the max length in the batch, right-pads shorter sequences to that length
    (also a multiple of W), and stacks into a single tensor.

    Args:
        batch: List of dicts, each with input_ids, labels, attention_mask tensors.
        W: Block size. The padded length is rounded up to the next multiple of W.
        pad_token_id: Token ID to use for padding (default: Qwen3 EOS/pad = 151643).

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
            # Pad input_ids with pad_token_id (not 0 which is a real vocab token)
            input_ids_list.append(
                torch.cat([ex["input_ids"], torch.full((pad_amount,), pad_token_id, dtype=torch.long)])
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
    # ---- Distributed setup ----
    rank, world_size, local_rank = setup_distributed()

    # ---- Setup logging (rank 0 only for INFO) ----
    logging.basicConfig(
        level=logging.INFO if rank == 0 else logging.WARNING,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    logger.info("=" * 60)
    logger.info("KV Self-Compaction Phase 2 — Training")
    logger.info("  Condition: %s", config.condition)
    logger.info("  W=%d, P=%d, K=%d", config.W, config.P, config.K)
    logger.info("  batch_size=%d, grad_accum=%d, max_steps=%d", config.batch_size, config.grad_accum, config.max_steps)
    logger.info("  lr=%g, lr_compaction=%g, lr_bias=%g", config.lr, config.lr_compaction, config.lr_bias)
    logger.info("  DDP: rank=%d, world_size=%d, local_rank=%d", rank, world_size, local_rank)
    logger.info("=" * 60)

    # ---- Seed (same seed for model init, different data order via DistributedSampler) ----
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

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

    # ---- W&B (rank 0 only) ----
    use_wandb = False
    if rank == 0:
        try:
            import wandb
            wandb_run_id = os.environ.get("WANDB_RUN_ID", None)
            wandb.init(
                project=config.wandb_project,
                id=wandb_run_id,
                name=f"cond_{config.condition}_W{config.W}_P{config.P}_K{config.K}",
                config=vars(config),
                resume="allow" if wandb_run_id else "never",
            )
            use_wandb = True
            logger.info("W&B initialized: %s", wandb.run.url)
        except Exception as e:
            logger.warning("W&B init failed: %s (continuing without)", e)

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

    # ---- LR Scheduler: WSD (Warmup-Stable-Decay) ----
    total_steps = config.max_steps
    warmup_steps = 50  # Quick warmup
    decay_start = int(total_steps * 0.8)  # Last 20% is decay
    decay_steps = total_steps - decay_start

    def wsd_lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            # Linear warmup
            return float(current_step) / float(max(1, warmup_steps))
        elif current_step < decay_start:
            # Stable phase — full LR
            return 1.0
        else:
            # Cosine decay to 0
            progress = float(current_step - decay_start) / float(max(1, decay_steps))
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, wsd_lr_lambda)

    logger.info("Scheduler: WSD (warmup=%d, stable=%d-%d, decay=%d-%d)",
                warmup_steps, warmup_steps, decay_start, decay_start, total_steps)

    # ---- Resume from checkpoint ----
    start_step = 0
    if resume_from is not None:
        start_step = load_checkpoint(resume_from, model, optimizer, scheduler)
        logger.info("Resuming from step %d", start_step)

    # ---- Data ----
    logger.info("Loading data...")
    train_dataset, val_dataset = load_data(config, tokenizer)
    logger.info("Train: %d examples, Val: %d examples", len(train_dataset), len(val_dataset))

    # Build collate_fn with W and pad_token_id bound
    W = config.W
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    def _collate(batch):
        return collate_fn(batch, W, pad_token_id=pad_id)

    train_sampler = (
        DistributedSampler(train_dataset, num_replicas=world_size, rank=rank,
                           shuffle=True, seed=config.seed)
        if world_size > 1 else None
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        collate_fn=_collate,
        num_workers=0,  # data is already in memory
        drop_last=True,
    )
    # Eval batch size capped at 4 to avoid OOM from logit concatenation
    # (batch_size=8 with max_seq_len=4096 and vocab=151936 → 18.5 GB logits tensor)
    eval_batch_size = min(config.batch_size, 4)
    val_loader = DataLoader(
        val_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        collate_fn=_collate,
        num_workers=0,
        drop_last=False,
    )

    # ---- Output directory (rank 0 only) ----
    output_dir = Path(config.output_dir) / f"condition_{config.condition}"
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        config_path = output_dir / "config.json"
        with open(config_path, "w") as f:
            json.dump(vars(config), f, indent=2)
        logger.info("Config saved to %s", config_path)
    if world_size > 1:
        dist.barrier()

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
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            # ---- Route to correct condition ----
            if config.condition == "A":
                # Full context SFT — manual loss to match blockwise normalization.
                # Labels are pre-shifted (labels[j] = token_ids[j+1]), so we compute
                # CE(logits[j], labels[j]) directly, NOT via HF's model(labels=...)
                # which would double-shift.
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
                logits = outputs.logits.float()  # (B, seq_len, vocab)
                valid_mask = labels != -100
                num_valid = valid_mask.sum().item()
                if num_valid > 0:
                    micro_loss = torch.nn.functional.cross_entropy(
                        logits[valid_mask], labels[valid_mask],
                        reduction="sum",
                    )
                    (micro_loss / num_valid / config.grad_accum).backward()
                    accum_loss += micro_loss.detach().item() / num_valid
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
                # Truncation baseline — last W tokens only, manual loss (no double-shift)
                W_trunc = config.W
                trunc_input_ids = input_ids[:, -W_trunc:]
                trunc_labels = labels[:, -W_trunc:]
                trunc_attention_mask = attention_mask[:, -W_trunc:]
                outputs = model(
                    input_ids=trunc_input_ids,
                    attention_mask=trunc_attention_mask,
                )
                logits = outputs.logits.float()
                valid_mask = trunc_labels != -100
                num_valid = valid_mask.sum().item()
                if num_valid > 0:
                    micro_loss = torch.nn.functional.cross_entropy(
                        logits[valid_mask], trunc_labels[valid_mask],
                        reduction="sum",
                    )
                    (micro_loss / num_valid / config.grad_accum).backward()
                    accum_loss += micro_loss.detach().item() / num_valid
                accum_tokens += 1

            # ---- Gradient accumulation: step every grad_accum micro-batches ----
            if accum_tokens >= config.grad_accum:
                # For conditions B/D/E, blockwise_train_step called backward() internally
                # without scaling. Scale accumulated gradients now.
                if config.condition in ("B", "D", "E") and config.grad_accum > 1:
                    for p in model.parameters():
                        if p.grad is not None:
                            p.grad.div_(config.grad_accum)

                # ---- DDP: all-reduce gradients across ranks ----
                if world_size > 1:
                    all_reduce_gradients(model, world_size)

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)

                # Optimizer step
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

                # Max-norm clamp compaction embeddings AFTER optimizer step
                with torch.no_grad():
                    norms = model.compaction_embeddings.data.norm(dim=-1, keepdim=True)
                    scale = torch.clamp(config.embed_max_norm / norms, max=1.0)
                    model.compaction_embeddings.data.mul_(scale)

                step += 1
                avg_loss = accum_loss / accum_tokens

                # ---- Logging every step (rank 0 only) ----
                if rank == 0:
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
                    # Write training loss to file for plotting
                    train_log_path = output_dir / "train_loss.jsonl"
                    train_record = {
                        "step": step, "loss": avg_loss, "bias_mean": bias_mean,
                        "embed_norm": embed_norm, "lr": current_lr, "elapsed": elapsed,
                    }
                    with open(train_log_path, "a") as f:
                        f.write(json.dumps(train_record) + "\n")
                    if use_wandb:
                        wandb.log(train_record, step=step)

                # ---- Eval (rank 0 only) ----
                if rank == 0 and step % config.eval_every == 0:
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
                    if use_wandb:
                        wandb.log({k: v for k, v in metrics_record.items()
                                   if k != "per_block_ppl"}, step=step)

                    model.train()

                # ---- Checkpoint (rank 0 only) ----
                if rank == 0 and step % config.save_every == 0:
                    save_checkpoint(model, optimizer, scheduler, step, config, str(output_dir))

                # Barrier after eval/checkpoint to keep ranks in sync
                if world_size > 1 and (step % config.eval_every == 0 or step % config.save_every == 0):
                    dist.barrier()

                # Reset accumulators
                accum_loss = 0.0
                accum_tokens = 0

                # Check max_steps
                if step >= config.max_steps:
                    break

        epoch += 1
        if step >= config.max_steps:
            break

    # ---- Final eval + checkpoint (rank 0 only) ----
    elapsed = time.time() - t_start
    if rank == 0:
        logger.info("Training complete. %d steps in %.1f seconds (%.1f sec/step)",
                    step, elapsed, elapsed / max(step, 1))

        logger.info("Final evaluation...")
        metrics = evaluate(model, val_loader, config)
        print_metrics(metrics, config)

        save_checkpoint(model, optimizer, scheduler, step, config, str(output_dir))

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

    # Cleanup
    if rank == 0 and use_wandb:
        wandb.finish()
    if world_size > 1:
        dist.destroy_process_group()


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
