#!/usr/bin/env python3
"""Collect all experiment metrics and generate LaTeX report."""
import json, os, math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

OUT = Path("/pscratch/sd/s/siddart2/kv-self-compaction-phase2/report")
OUT.mkdir(exist_ok=True)
RESULTS = Path("/pscratch/sd/s/siddart2/kv-self-compaction-phase2/results")
RESULTS.mkdir(exist_ok=True)

plt.rcParams.update({'font.size': 11, 'axes.titlesize': 13, 'axes.labelsize': 12})

# ============================================================
# Collect all experiment data
# ============================================================

experiments = {}

# Scan all output directories
base = Path("/pscratch/sd/s/siddart2/kv-self-compaction-phase2/outputs")
for exp_dir in sorted(base.glob("**/condition_*")):
    # Parse experiment name
    parts = str(exp_dir.relative_to(base)).split("/")
    run_name = parts[0].replace("ddp_scaleup_", "").replace("ddp_scaleup", "main")
    cond = parts[1].replace("condition_", "")
    exp_name = f"{cond}_{run_name}" if run_name != "main" else cond

    # Load train losses
    train_path = exp_dir / "train_loss.jsonl"
    train_data = []
    if train_path.exists():
        with open(train_path) as f:
            train_data = [json.loads(l) for l in f if l.strip()]

    # Load eval metrics
    metrics_path = exp_dir / "metrics.jsonl"
    eval_data = []
    if metrics_path.exists():
        with open(metrics_path) as f:
            eval_data = [json.loads(l) for l in f if l.strip()]

    # Load final metrics
    final_path = exp_dir / "final_metrics.json"
    final_data = None
    if final_path.exists():
        with open(final_path) as f:
            final_data = json.load(f)

    # Load config
    config_path = exp_dir / "config.json"
    config = None
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)

    if train_data or eval_data or final_data:
        experiments[exp_name] = {
            "dir": str(exp_dir),
            "config": config,
            "train": train_data,
            "eval": eval_data,
            "final": final_data,
        }
        n_train = len(train_data)
        n_eval = len(eval_data)
        last_step = train_data[-1]["step"] if train_data else 0
        print(f"  {exp_name}: {n_train} train steps, {n_eval} evals, last_step={last_step}")

# Save collected data
with open(RESULTS / "all_experiments.json", "w") as f:
    json.dump(experiments, f, indent=2, default=str)
print(f"\nSaved {len(experiments)} experiments to {RESULTS / 'all_experiments.json'}")

# ============================================================
# Generate plots
# ============================================================

# Color scheme
COLORS = {
    "B": "#2196F3", "A": "#4CAF50", "D": "#FF9800", "E": "#F44336", "C": "#9E9E9E",
    "B_W512": "#2196F3", "A_W512": "#4CAF50",
    "B_W512_K1": "#9C27B0", "E_W512": "#F44336", "E_W512_E": "#F44336",
}

def get_color(name):
    for key, col in COLORS.items():
        if key in name:
            return col
    return "#333333"

# Plot 1: Training loss curves
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

for name, exp in sorted(experiments.items()):
    if not exp["train"]:
        continue
    steps = [d["step"] for d in exp["train"]]
    losses = [d["loss"] for d in exp["train"]]
    # Smooth with rolling average
    if len(losses) > 20:
        window = min(20, len(losses)//5)
        smoothed = np.convolve(losses, np.ones(window)/window, mode='valid')
        smooth_steps = steps[window-1:]
    else:
        smoothed = losses
        smooth_steps = steps
    ax1.plot(smooth_steps, smoothed, label=name, color=get_color(name), linewidth=1.5, alpha=0.8)

ax1.set_xlabel('Training Step')
ax1.set_ylabel('Training Loss')
ax1.set_title('Training Loss (smoothed)')
ax1.legend(fontsize=8)
ax1.grid(True, alpha=0.3)

# Plot 2: Eval val_ppl and full_context_ppl
for name, exp in sorted(experiments.items()):
    if not exp["eval"]:
        continue
    steps = [d["step"] for d in exp["eval"]]
    # Use full_context_ppl for A, val_ppl for others
    if "A" in name and any("full_context_ppl" in d for d in exp["eval"]):
        raw_ppls = [d.get("full_context_ppl", float("nan")) for d in exp["eval"]]
        label = f"{name} (full ctx)"
    else:
        raw_ppls = [d.get("val_ppl", float("nan")) for d in exp["eval"]]
        label = f"{name} (blockwise)"
    # Filter (step, ppl) pairs together to maintain alignment
    paired = [(s, p) for s, p in zip(steps, raw_ppls) if isinstance(p, (int, float)) and not math.isnan(p)]
    if paired:
        plot_steps, plot_ppls = zip(*paired)
        ax2.plot(plot_steps, plot_ppls, 'o-', label=label, color=get_color(name), linewidth=2, markersize=6)

ax2.set_xlabel('Training Step')
ax2.set_ylabel('Validation PPL')
ax2.set_title('Validation Perplexity')
ax2.legend(fontsize=8)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUT / "fig1_training.pdf", dpi=150, bbox_inches='tight')
plt.savefig(OUT / "fig1_training.png", dpi=150, bbox_inches='tight')
print("Saved fig1_training")

# Plot 3: Bias trajectory (for compaction conditions)
fig, ax = plt.subplots(figsize=(8, 5))
for name, exp in sorted(experiments.items()):
    if not exp["train"] or "bias_mean" not in exp["train"][0]:
        continue
    steps = [d["step"] for d in exp["train"]]
    biases = [d["bias_mean"] for d in exp["train"]]
    if len(biases) > 20:
        window = min(20, len(biases)//5)
        smoothed = np.convolve(biases, np.ones(window)/window, mode='valid')
        smooth_steps = steps[window-1:]
    else:
        smoothed = biases
        smooth_steps = steps
    ax.plot(smooth_steps, smoothed, label=name, color=get_color(name), linewidth=1.5)

ax.axhline(y=-2.0, color='gray', linestyle=':', linewidth=1, label='init=-2.0')
ax.axhline(y=0.0, color='gray', linestyle='-', alpha=0.3, linewidth=0.5)
ax.set_xlabel('Training Step')
ax.set_ylabel('compact_attn_bias mean')
ax.set_title('Attention Bias Trajectory')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUT / "fig2_bias.pdf", dpi=150, bbox_inches='tight')
plt.savefig(OUT / "fig2_bias.png", dpi=150, bbox_inches='tight')
print("Saved fig2_bias")

# Plot 4: Per-block PPL (if available)
fig, ax = plt.subplots(figsize=(10, 5))
has_per_block = False
for name, exp in sorted(experiments.items()):
    if not exp["eval"]:
        continue
    last_eval = exp["eval"][-1]
    if "per_block_ppl" not in last_eval:
        continue
    pb = last_eval["per_block_ppl"]
    blocks = sorted([int(k) for k in pb.keys() if int(k) > 0])
    vals = [pb[str(b)] for b in blocks]
    if vals:
        ax.plot(blocks, vals, 'o-', label=f'{name} (step {last_eval["step"]})',
                color=get_color(name), linewidth=1.5, markersize=4)
        has_per_block = True

if has_per_block:
    ax.set_xlabel('Block Index')
    ax.set_ylabel('Per-Block PPL')
    ax.set_title('Per-Block Perplexity (blocks 1+)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT / "fig3_per_block.pdf", dpi=150, bbox_inches='tight')
    plt.savefig(OUT / "fig3_per_block.png", dpi=150, bbox_inches='tight')
    print("Saved fig3_per_block")
else:
    plt.close()

# ============================================================
# Summary table
# ============================================================
print("\n" + "=" * 80)
print("EXPERIMENT SUMMARY")
print("=" * 80)
print(f"{'Experiment':<20} {'Steps':>6} {'Train Loss':>11} {'Val PPL':>9} {'CB PPL':>8} {'FC PPL':>8} {'Bias':>7}")
print("-" * 80)
for name, exp in sorted(experiments.items()):
    steps = exp["train"][-1]["step"] if exp["train"] else 0
    loss = exp["train"][-1]["loss"] if exp["train"] else float("nan")
    if exp["eval"]:
        last_e = exp["eval"][-1]
        val_ppl = last_e.get("val_ppl", float("nan"))
        cb_ppl = last_e.get("cross_block_ppl", float("nan"))
        fc_ppl = last_e.get("full_context_ppl", float("nan"))
        bias = last_e.get("attn_bias_mean", float("nan"))
    else:
        val_ppl = cb_ppl = fc_ppl = bias = float("nan")
    fc_str = f"{fc_ppl:.3f}" if not math.isnan(fc_ppl) else "—"
    bias_str = f"{bias:.3f}" if not math.isnan(bias) else "—"
    print(f"{name:<20} {steps:>6} {loss:>11.3f} {val_ppl:>9.3f} {cb_ppl:>8.2f} {fc_str:>8} {bias_str:>7}")
print("=" * 80)

# Save summary
summary = {}
for name, exp in experiments.items():
    last_eval = exp["eval"][-1] if exp["eval"] else {}
    summary[name] = {
        "steps": exp["train"][-1]["step"] if exp["train"] else 0,
        "train_loss": exp["train"][-1]["loss"] if exp["train"] else None,
        "val_ppl": last_eval.get("val_ppl"),
        "cross_block_ppl": last_eval.get("cross_block_ppl"),
        "full_context_ppl": last_eval.get("full_context_ppl"),
        "attn_bias_mean": last_eval.get("attn_bias_mean"),
        "config": exp["config"],
    }
with open(RESULTS / "summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print(f"\nSaved summary to {RESULTS / 'summary.json'}")
print(f"Plots in {OUT}")
