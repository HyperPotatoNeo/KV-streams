#!/usr/bin/env python3
"""Generate LaTeX report with plots for KV Self-Compaction Phase 2."""
import json, re, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

OUT = Path("/pscratch/sd/s/siddart2/kv-self-compaction-phase2/report")
OUT.mkdir(exist_ok=True)
phase2a_dir = Path("/pscratch/sd/s/siddart2/kv-self-compaction-phase2/outputs/phase2a_v2")

def load_metrics(path):
    if not path.exists(): return []
    with open(path) as f: return [json.loads(l) for l in f if l.strip()]

def load_final(path):
    if not path.exists(): return None
    with open(path) as f: return json.load(f)

# Phase 2a v2 training logs (from stdout captures)
# B (200 steps, correct labels)
b_train = [
    (10, 1.8517, -1.945), (20, 2.0861, -1.672), (30, 1.7606, -1.430),
    (40, 1.8837, -1.281), (50, 2.2901, -1.164), (60, 1.8348, -1.078),
    (70, 1.8463, -1.047), (80, 1.9933, -1.031), (90, 1.8780, -1.000),
    (100, 1.7887, -0.984), (110, 1.8483, -0.961), (120, 1.9814, -0.957),
    (130, 1.8443, -0.953), (140, 4.1848, -0.949), (150, 1.7349, -0.938),
    (160, 1.8773, -0.934), (170, 1.5152, -0.934), (180, 2.2830, -0.934),
    (190, 1.6588, -0.934), (200, 1.9594, -0.934),
]
# D (200 steps, correct labels)
d_train = [
    (10, 2.1202, -2.016), (20, 2.1449, -2.047), (30, 1.8511, -2.062),
    (40, 1.9896, -2.062), (50, 2.4290, -2.094), (60, 1.9954, -2.109),
    (70, 2.0525, -2.109), (80, 2.1476, -2.109), (90, 2.0282, -2.125),
    (100, 1.9624, -2.125), (110, 1.9917, -2.125), (120, 2.1749, -2.125),
    (130, 2.0130, -2.125), (140, 4.3460, -2.125), (150, 1.9839, -2.125),
    (160, 2.0236, -2.125), (170, 1.6948, -2.125), (180, 2.4854, -2.125),
    (190, 1.8561, -2.125), (200, 2.1388, -2.125),
]
# A (200 steps, correct labels - uses HF internal shift)
a_train = [
    (10, 6.5218, -2.0), (20, 4.2594, -2.0), (30, 4.3928, -2.0),
    (40, 4.3936, -2.0), (50, 4.4525, -2.0), (60, 3.6228, -2.0),
    (70, 4.3411, -2.0), (80, 3.4438, -2.0), (90, 2.9255, -2.0),
    (100, 3.2480, -2.0), (110, 3.5781, -2.0), (120, 3.5295, -2.0),
    (130, 3.2293, -2.0), (140, 3.8256, -2.0), (150, 2.7892, -2.0),
    (160, 2.6045, -2.0), (170, 2.8180, -2.0), (180, 3.6662, -2.0),
    (190, 2.7005, -2.0), (200, 3.9035, -2.0),
]
# C (200 steps)
c_train = [
    (10, 5.8284, -2.0), (20, 4.6343, -2.0), (30, 3.7498, -2.0),
    (40, 5.1566, -2.0), (50, 5.6179, -2.0), (60, 4.4713, -2.0),
    (70, 7.1762, -2.0), (80, 5.0936, -2.0), (90, 3.7990, -2.0),
    (100, 4.8683, -2.0), (110, 8.3961, -2.0), (120, 3.2895, -2.0),
    (130, 4.5158, -2.0), (140, 8.0264, -2.0), (150, 5.0587, -2.0),
    (160, 3.6524, -2.0), (170, 3.8567, -2.0), (180, 5.6408, -2.0),
    (190, 4.2750, -2.0), (200, 4.1592, -2.0),
]

b_metrics = load_metrics(phase2a_dir / "condition_B" / "metrics.jsonl")
d_final = load_final(phase2a_dir / "condition_D" / "final_metrics.json")
b_final = load_final(phase2a_dir / "condition_B" / "final_metrics.json")
a_final = load_final(phase2a_dir / "condition_A" / "final_metrics.json")
c_metrics = load_metrics(phase2a_dir / "condition_C" / "metrics.jsonl")

plt.rcParams.update({'font.size': 11, 'axes.titlesize': 13, 'axes.labelsize': 12})

# ============================================================
# Plot 1: Training loss curves (all conditions)
# ============================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

# Left: Training losses
for data, label, color, ls in [
    (b_train, 'B (learned compaction)', '#2196F3', '-'),
    (d_train, 'D (random compact_kv)', '#FF9800', '--'),
    (a_train, 'A (full context)', '#9E9E9E', ':'),
    (c_train, 'C (truncation W=128)', '#F44336', '-.'),
]:
    steps = [d[0] for d in data]
    losses = [d[1] for d in data]
    ax1.plot(steps, losses, ls, color=color, label=label, linewidth=1.8, alpha=0.85)

ax1.set_xlabel('Training Step')
ax1.set_ylabel('Training Loss (per-token CE)')
ax1.set_title('Training Loss — Phase 2a (200 steps, 5K examples)')
ax1.legend(fontsize=9, loc='upper right')
ax1.grid(True, alpha=0.3)
ax1.set_ylim(0, 10)

# Right: Attention bias trajectories
for data, label, color, ls in [
    (b_train, 'B (learned)', '#2196F3', '-'),
    (d_train, 'D (random)', '#FF9800', '--'),
]:
    steps = [d[0] for d in data]
    biases = [d[2] for d in data]
    ax2.plot(steps, biases, ls, color=color, label=label, linewidth=2, marker='o', markersize=3)

ax2.axhline(y=-2.0, color='gray', linestyle=':', label='init = −2.0', linewidth=1)
ax2.axhline(y=0.0, color='gray', linestyle='-', alpha=0.3, linewidth=0.5)
ax2.set_xlabel('Training Step')
ax2.set_ylabel('compact_attn_bias mean')
ax2.set_title('Attention Bias Trajectory')
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)
ax2.annotate('B: opens attention\nto compact_kv', xy=(100, -0.98), fontsize=8,
             ha='center', color='#2196F3')
ax2.annotate('D: suppresses\nrandom noise', xy=(100, -2.12), fontsize=8,
             ha='center', color='#FF9800')

plt.tight_layout()
plt.savefig(OUT / "fig1_training.pdf", dpi=150, bbox_inches='tight')
plt.savefig(OUT / "fig1_training.png", dpi=150, bbox_inches='tight')
print("Saved fig1_training")

# ============================================================
# Plot 2: Eval metrics (cross_block_ppl and val_ppl over training)
# ============================================================
if b_metrics:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    steps = [m["step"] for m in b_metrics]
    cb = [m["cross_block_ppl"] for m in b_metrics]
    vp = [m["val_ppl"] for m in b_metrics]

    ax1.plot(steps, cb, 'o-', color='#2196F3', label='B cross_block_ppl', linewidth=2, markersize=8)
    ax1.plot(steps, vp, 's-', color='#4CAF50', label='B val_ppl', linewidth=2, markersize=8)
    if d_final:
        ax1.axhline(y=d_final["cross_block_ppl"], color='#FF9800', linestyle='--',
                     label=f'D cross_block_ppl = {d_final["cross_block_ppl"]:.1f}', linewidth=1.5)
    ax1.set_xlabel('Training Step')
    ax1.set_ylabel('Perplexity')
    ax1.set_title('Evaluation Perplexity Over Training')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(bottom=0)

    # Per-block PPL comparison at final step
    if b_final and d_final:
        b_blk = b_final["per_block_ppl"]
        d_blk = d_final["per_block_ppl"]
        ids = sorted([int(k) for k in b_blk.keys() if int(k) > 0])
        bv = [b_blk[str(i)] for i in ids]
        dv = [d_blk[str(i)] for i in ids]
        x = np.arange(len(ids))
        w = 0.35
        ax2.bar(x - w/2, bv, w, label='B (learned)', color='#2196F3', alpha=0.8)
        ax2.bar(x + w/2, dv, w, label='D (random)', color='#FF9800', alpha=0.8)
        ax2.set_xlabel('Block Index')
        ax2.set_ylabel('Per-Block PPL')
        ax2.set_title('Per-Block PPL at Step 200 (blocks 1–31)')
        ax2.set_xticks(x[::3])
        ax2.set_xticklabels([str(i) for i in ids[::3]])
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(OUT / "fig2_eval.pdf", dpi=150, bbox_inches='tight')
    plt.savefig(OUT / "fig2_eval.png", dpi=150, bbox_inches='tight')
    print("Saved fig2_eval")

# ============================================================
# Plot 3: Condition comparison bar chart
# ============================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))

# B vs D (main)
names = ['B\n(learned)', 'D\n(random)']
vals = [b_final["cross_block_ppl"], d_final["cross_block_ppl"]]
cols = ['#2196F3', '#FF9800']
bars = ax1.bar(names, vals, color=cols, width=0.5, edgecolor='black', linewidth=0.5)
for bar, val in zip(bars, vals):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
             f'{val:.1f}', ha='center', fontsize=13, fontweight='bold')
ax1.set_ylabel('Cross-Block PPL')
ax1.set_title('B vs D: 45% Improvement\n(learned compaction works)')
ax1.grid(True, alpha=0.3, axis='y')
ax1.set_ylim(0, 20)

# All conditions (log scale)
all_names = ['B\n(learned)', 'D\n(random)', 'C\n(truncation)', 'A\n(full ctx)']
all_vals = [b_final["cross_block_ppl"], d_final["cross_block_ppl"],
            c_metrics[-1]["cross_block_ppl"], a_final["cross_block_ppl"]]
all_cols = ['#2196F3', '#FF9800', '#F44336', '#9E9E9E']
bars2 = ax2.bar(all_names, all_vals, color=all_cols, width=0.5, edgecolor='black', linewidth=0.5)
ax2.set_yscale('log')
for bar, val in zip(bars2, all_vals):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.2,
             f'{val:.0f}', ha='center', fontsize=10, fontweight='bold')
ax2.set_ylabel('Cross-Block PPL (log)')
ax2.set_title('All Conditions (log scale)')
ax2.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(OUT / "fig3_comparison.pdf", dpi=150, bbox_inches='tight')
plt.savefig(OUT / "fig3_comparison.png", dpi=150, bbox_inches='tight')
print("Saved fig3_comparison")

print(f"\nAll plots in {OUT}")
