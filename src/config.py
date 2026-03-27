"""Configuration dataclasses for KV Self-Compaction Phase 2.

All hyperparameters for model, compaction mechanism, training, and data loading.
Phase 2a defaults: 1 GPU, Condition B, 5K examples per dataset, 200 steps.
"""

from dataclasses import dataclass, field
from typing import Literal, Optional


@dataclass
class CompactionConfig:
    """Full configuration for KV Self-Compaction training.

    Organized into sections: model, compaction, LoRA, optimizer, training,
    compaction initialization, data, condition, and output.
    """

    # --- Model ---
    model_name: str = "Qwen/Qwen3-0.6B-Base"
    """HuggingFace model ID. Must be the BASE model (not instruct) for SFT."""

    attn_implementation: str = "eager"
    """Attention implementation. MUST be 'eager' — SDPA silently ignores our custom
    4D attention mask with learnable bias on compact_kv columns."""

    # --- Compaction mechanism ---
    W: int = 128
    """Block size in tokens. Input sequences are divided into blocks of W tokens,
    each processed with compaction tokens appended. Changed from 256 to 128 based on
    sequence length analysis: median=997 tokens, W=128 gives 7 blocks at median vs 3."""

    P: int = 16
    """Number of compaction tokens per block (W/8 compression ratio). Changed from
    32 to 16 to match W=128. Phase 1 showed monotonically better with more P
    (when using bias), but 28-layer GQA model has 14x more state per token than
    Phase 1's 8-layer model, so P=16 should be sufficient."""

    K: int = 2
    """BPTT depth — number of blocks before truncating gradients. After K blocks,
    we call backward() and detach compact_kv. K=2 is a conservative start;
    higher K allows longer-range gradient flow but uses more memory."""

    # --- LoRA ---
    lora_rank: int = 32
    """LoRA rank for all target linear layers."""

    lora_alpha: int = 64
    """LoRA alpha scaling factor. Effective scaling = alpha / rank = 2.0."""

    lora_targets: list[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"
    ])
    """Target modules for LoRA. All linear layers in each transformer block."""

    # --- Optimizer ---
    lr: float = 1.5e-4
    """Learning rate for LoRA parameters (AdamW)."""

    lr_compaction: float = 3e-4
    """Learning rate for compaction_embeddings. Higher than LoRA LR because these
    are new parameters that must learn from scratch."""

    lr_bias: float = 0.05
    """Learning rate for compact_attn_bias. High scalar LR because these are
    single per-head values that need to move quickly (init -2.0)."""

    weight_decay: float = 0.01
    """Weight decay for LoRA parameters. Compaction params use 0.0."""

    max_grad_norm: float = 1.0
    """Gradient clipping norm. Applied to all parameters."""

    warmup_ratio: float = 0.05
    """Fraction of total steps used for linear warmup."""

    # --- Training ---
    max_seq_len: int = 4096
    """Maximum sequence length after tokenization. Sequences longer than this
    are truncated. Must be divisible by W. Changed from 2048 to 4096 to cover
    p90 of the combined distribution (3886 tokens)."""

    batch_size: int = 4
    """Per-GPU batch size. Phase 2a: 4 (single GPU). Phase 2c: 4 per GPU x 16 GPUs."""

    grad_accum: int = 1
    """Gradient accumulation steps. Phase 2a: 1. Phase 2c: 8."""

    max_steps: int = 200
    """Maximum training steps. Phase 2a: 200. Phase 2c: 660."""

    eval_every: int = 50
    """Evaluate every N steps."""

    save_every: int = 100
    """Save checkpoint every N steps."""

    seed: int = 42
    """Random seed for reproducibility."""

    bf16: bool = True
    """Use bfloat16 precision. Required for A100 GPUs."""

    # --- Compaction initialization ---
    bias_init: float = -2.0
    """Initial value for per-head compact_attn_bias. exp(-2) ~ 0.14, so compact_kv
    attention weights start suppressed. This was the Phase 1 breakthrough: without
    this bias, P>1 compaction fails because attention gets diluted over too many
    compact_kv positions."""

    embed_init_std: float = 0.02
    """Standard deviation for compaction_embeddings initialization (normal dist).
    Matches Phase 1 pattern."""

    embed_max_norm: float = 10.0
    """Max norm for compaction_embeddings. Clamped after each optimizer step to
    prevent embedding drift."""

    # --- Data ---
    think_sft_path: str = "allenai/Dolci-Think-SFT-7B"
    """HuggingFace path for Think-SFT dataset. Contains ~2.27M examples with
    <think>...</think> reasoning traces in assistant responses."""

    instruct_sft_path: str = "allenai/Dolci-Instruct-SFT"
    """HuggingFace path for Instruct-SFT dataset. Contains ~2.15M examples,
    standard instruction-following without reasoning traces."""

    max_examples_per_dataset: Optional[int] = 5000
    """Maximum examples to sample from each dataset. Phase 2a: 5000 per dataset
    (10K total). Set to None for full dataset in Phase 2c."""

    val_fraction: float = 0.05
    """Fraction of data reserved for validation (5%)."""

    # --- Condition ---
    condition: Literal["A", "B", "C", "D", "E"] = "B"
    """Experimental condition:
    A = Full context SFT (standard HF forward, no compaction)
    B = KV Self-Compaction (learned compact_kv, the method under test)
    C = Truncation baseline (last W tokens only)
    D = Random compact_kv (like B but compact_kv is random noise, not learned)
    E = Blockwise, no compact_kv (within-block only, fairest baseline for B)"""

    # --- Output ---
    output_dir: str = "outputs"
    """Directory for checkpoints, logs, and eval results."""

    wandb_project: str = "kv-self-compaction-phase2"
    """Weights & Biases project name."""


@dataclass
class Qwen3Dims:
    """Pre-computed architectural dimensions for Qwen3-0.6B-Base.

    These are fixed by the model architecture and used throughout the codebase
    for tensor shape computations. Extracted from the HuggingFace config to
    avoid repeated config lookups.
    """

    num_layers: int = 28
    """Number of transformer decoder layers."""

    hidden_size: int = 1024
    """Hidden dimension (embedding size)."""

    num_q_heads: int = 16
    """Number of query attention heads."""

    num_kv_heads: int = 8
    """Number of key/value attention heads (GQA with 2:1 ratio)."""

    head_dim: int = 128
    """Dimension per attention head. Independent of hidden_size for Qwen3
    (not hidden_size // num_q_heads)."""

    vocab_size: int = 151936
    """Vocabulary size including special tokens."""
