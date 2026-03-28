"""Data loading and preprocessing for KV Self-Compaction Phase 2.

Loads Dolci Think-SFT + Instruct-SFT datasets, formats with chat template,
creates labels with -100 masking for non-assistant tokens, and pads to
multiples of the block size W.

Dataset details:
- Think-SFT (allenai/Dolci-Think-SFT-7B): ~2.27M examples, assistant responses
  contain <think>...</think> reasoning traces that must be preserved.
- Instruct-SFT (allenai/Dolci-Instruct-SFT): ~2.15M examples, standard
  instruction-following. Has optional function_calls/functions fields (ignored).
- Both have a `messages` column: list of {role, content} dicts.

Tokenizer details (Qwen3-0.6B-Base):
- Built-in Jinja2 chat template with <|im_start|>/<|im_end|> delimiters
- No BOS token (add_bos_token=false)
- EOS = <|endoftext|> (ID 151643)
- Think tokens: <think> (ID 151667), </think> (ID 151668)
"""

import logging
from typing import Optional

import torch
from torch.utils.data import Dataset
from datasets import load_dataset

from .config import CompactionConfig

logger = logging.getLogger(__name__)


class CompactionDataset(Dataset):
    """Torch Dataset wrapping preprocessed examples for blockwise training.

    Each example has:
        input_ids: (seq_len,) — token IDs, padded to multiple of W
        labels: (seq_len,) — same as input_ids for assistant tokens, -100 elsewhere
        attention_mask: (seq_len,) — 1 for real tokens, 0 for padding
    """

    def __init__(
        self,
        input_ids: list[torch.Tensor],
        labels: list[torch.Tensor],
        attention_masks: list[torch.Tensor],
    ):
        assert len(input_ids) == len(labels) == len(attention_masks)
        self.input_ids = input_ids
        self.labels = labels
        self.attention_masks = attention_masks

    def __len__(self) -> int:
        return len(self.input_ids)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "input_ids": self.input_ids[idx],
            "labels": self.labels[idx],
            "attention_mask": self.attention_masks[idx],
        }


def _find_assistant_token_ranges(
    token_ids: list[int],
    tokenizer,
) -> list[tuple[int, int]]:
    """Find (start, end) ranges of assistant response tokens in a tokenized conversation.

    The chat template produces:
        <|im_start|>system\n...<|im_end|>\n
        <|im_start|>user\n...<|im_end|>\n
        <|im_start|>assistant\n...<|im_end|>\n
        ...

    We find each "<|im_start|>assistant\n" marker and mark everything after it
    (including the content) until and including the next <|im_end|> as assistant tokens.
    The label includes the <|im_end|> token so the model learns when to stop.

    Args:
        token_ids: Full tokenized conversation as a list of ints.
        tokenizer: The tokenizer (needed for special token IDs).

    Returns:
        List of (start, end) index pairs. tokens[start:end] are assistant tokens.
    """
    # Get special token IDs
    im_start_id = tokenizer.convert_tokens_to_ids("<|im_start|>")
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")

    # Encode "assistant\n" to get the token sequence that follows <|im_start|>
    assistant_marker_ids = tokenizer.encode("assistant\n", add_special_tokens=False)
    marker_len = len(assistant_marker_ids)

    ranges = []
    i = 0
    while i < len(token_ids):
        # Look for <|im_start|> followed by "assistant\n" tokens
        if token_ids[i] == im_start_id:
            # Check if followed by "assistant\n"
            marker_end = i + 1 + marker_len
            if (marker_end <= len(token_ids) and
                    token_ids[i + 1 : marker_end] == assistant_marker_ids):
                # Assistant content starts after the marker
                content_start = marker_end
                # Find the closing <|im_end|>
                content_end = content_start
                while content_end < len(token_ids) and token_ids[content_end] != im_end_id:
                    content_end += 1
                # Include the <|im_end|> token in the label range
                if content_end < len(token_ids):
                    content_end += 1
                ranges.append((content_start, content_end))
                i = content_end
                continue
        i += 1

    return ranges


def _process_example(
    messages: list[dict[str, str]],
    tokenizer,
    max_seq_len: int,
    W: int,
) -> Optional[dict[str, torch.Tensor]]:
    """Process a single conversation into padded input_ids, labels, attention_mask.

    Pipeline:
    1. Format with chat template (preserves <think>...</think> tags)
    2. Tokenize (no special tokens added — template handles them)
    3. Truncate to max_seq_len
    4. Create labels: -100 for non-assistant positions, token_ids for assistant
    5. Pad to nearest multiple of W

    Args:
        messages: List of {role, content} message dicts.
        tokenizer: Qwen3-0.6B-Base tokenizer.
        max_seq_len: Maximum sequence length before padding.
        W: Block size — sequences are padded to multiples of this.

    Returns:
        Dict with input_ids, labels, attention_mask tensors, or None if empty.
    """
    # Sanitize messages: keep only role+content, skip messages with None content
    # (Dolci-Instruct-SFT has function_calls/functions fields that break the template)
    clean_messages = []
    for msg in messages:
        content = msg.get("content")
        if content is None:
            continue
        clean_messages.append({"role": msg["role"], "content": content})
    if not clean_messages:
        return None

    # Format conversation using built-in chat template
    text = tokenizer.apply_chat_template(clean_messages, tokenize=False)

    # Tokenize without adding special tokens (template already added them)
    encoding = tokenizer(text, add_special_tokens=False, return_attention_mask=False)
    token_ids = encoding["input_ids"]

    if len(token_ids) == 0:
        return None

    # Truncate to max_seq_len
    token_ids = token_ids[:max_seq_len]

    # Find assistant response ranges for label masking
    assistant_ranges = _find_assistant_token_ranges(token_ids, tokenizer)

    # Create labels: -100 everywhere, then fill in assistant tokens with SHIFTED targets.
    # labels[j] = token_ids[j+1] (the NEXT token), so blockwise CE(logits[j], labels[j])
    # trains the model to predict the next token, not the current one.
    labels = [-100] * len(token_ids)
    for start, end in assistant_ranges:
        for j in range(start, min(end, len(token_ids))):
            if j + 1 < len(token_ids):
                labels[j] = token_ids[j + 1]
            # else: last position has no next token, stays -100

    # Skip examples with no trainable tokens
    if all(l == -100 for l in labels):
        return None

    # Pad to nearest multiple of W
    seq_len = len(token_ids)
    padded_len = ((seq_len + W - 1) // W) * W

    # Use pad_token_id, falling back to eos_token_id
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id

    pad_amount = padded_len - seq_len
    input_ids = torch.tensor(token_ids + [pad_id] * pad_amount, dtype=torch.long)
    labels = torch.tensor(labels + [-100] * pad_amount, dtype=torch.long)
    attention_mask = torch.tensor(
        [1] * seq_len + [0] * pad_amount, dtype=torch.long
    )

    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask,
    }


_CACHE_VERSION = "v3"  # Bump when changing _process_example logic (label shift, sanitization, etc.)


def _cache_key(config: CompactionConfig) -> str:
    """Deterministic cache key based on data-affecting config fields."""
    import hashlib
    key_parts = (
        _CACHE_VERSION,
        config.model_name,
        config.think_sft_path,
        config.instruct_sft_path,
        str(config.max_examples_per_dataset),
        str(config.seed),
        str(config.W),
        str(config.max_seq_len),
        str(config.val_fraction),
    )
    h = hashlib.md5("|".join(key_parts).encode()).hexdigest()[:12]
    return f"processed_{_CACHE_VERSION}_W{config.W}_N{config.max_examples_per_dataset}_{h}"


def load_data(
    config: CompactionConfig,
    tokenizer,
    cache_dir: str = "/pscratch/sd/s/siddart2/kv-self-compaction-phase2/data_cache",
) -> tuple[CompactionDataset, CompactionDataset]:
    """Load tokenized dataset, using cache if available.

    If a cached .pt file exists for this config, loads from it (~2s).
    Otherwise falls back to full tokenization (~20 min for 100K examples).

    Args:
        config: CompactionConfig with data paths, max_examples, W, max_seq_len.
        cache_dir: Directory for cached .pt files.
        tokenizer: Qwen3-0.6B-Base tokenizer with built-in chat template.

    Returns:
        (train_dataset, val_dataset) — CompactionDataset instances with
        input_ids, labels, attention_mask fields.
    """
    import os

    # Try loading from cache first
    key = _cache_key(config)
    cache_path = os.path.join(cache_dir, f"{key}.pt")
    if os.path.exists(cache_path):
        logger.info("Loading cached dataset from %s", cache_path)
        data = torch.load(cache_path, weights_only=False)
        assert data["config_key"] == key, f"Cache key mismatch: {data['config_key']} != {key}"
        train_ds = CompactionDataset(
            input_ids=data["train_input_ids"],
            labels=data["train_labels"],
            attention_masks=data["train_attention_masks"],
        )
        val_ds = CompactionDataset(
            input_ids=data["val_input_ids"],
            labels=data["val_labels"],
            attention_masks=data["val_attention_masks"],
        )
        logger.info("Loaded %d train + %d val from cache", len(train_ds), len(val_ds))
        return train_ds, val_ds

    # No cache — do full tokenization
    logger.info("No cache found, running full tokenization...")
    train_ds, val_ds = _load_data_raw(config, tokenizer)

    # Save cache for next time (atomic write to avoid DDP race)
    # All ranks tokenize (needed for memory), but only rank 0 writes cache
    rank = int(os.environ.get("RANK", "0"))
    if rank == 0:
        os.makedirs(cache_dir, exist_ok=True)
        tmp_path = cache_path + ".tmp"
        torch.save({
            "train_input_ids": train_ds.input_ids,
            "train_labels": train_ds.labels,
            "train_attention_masks": train_ds.attention_masks,
            "val_input_ids": val_ds.input_ids,
            "val_labels": val_ds.labels,
            "val_attention_masks": val_ds.attention_masks,
            "config_key": key,
        }, tmp_path)
        os.rename(tmp_path, cache_path)  # atomic on POSIX
        logger.info("Cached to %s (%.1f MB)", cache_path, os.path.getsize(cache_path) / 1e6)

    return train_ds, val_ds


def _load_data_raw(
    config: CompactionConfig,
    tokenizer,
) -> tuple[CompactionDataset, CompactionDataset]:
    """Raw data loading without cache. Called by load_data and save_processed_data."""
    logger.info("Loading Think-SFT dataset: %s", config.think_sft_path)
    think_ds = load_dataset(config.think_sft_path, split="train")

    logger.info("Loading Instruct-SFT dataset: %s", config.instruct_sft_path)
    instruct_ds = load_dataset(config.instruct_sft_path, split="train")

    # Subsample for Phase 2a
    if config.max_examples_per_dataset is not None:
        n_think = min(config.max_examples_per_dataset, len(think_ds))
        n_instruct = min(config.max_examples_per_dataset, len(instruct_ds))
        think_ds = think_ds.shuffle(seed=config.seed).select(range(n_think))
        instruct_ds = instruct_ds.shuffle(seed=config.seed).select(range(n_instruct))
        logger.info("Subsampled to %d Think-SFT + %d Instruct-SFT examples",
                     n_think, n_instruct)

    # Interleave: alternate rows from each dataset for balanced mixing
    interleaved_messages = []
    think_iter = iter(think_ds)
    instruct_iter = iter(instruct_ds)
    think_exhausted = False
    instruct_exhausted = False

    while not (think_exhausted and instruct_exhausted):
        if not think_exhausted:
            example = next(think_iter, None)
            if example is not None:
                interleaved_messages.append(example["messages"])
            else:
                think_exhausted = True

        if not instruct_exhausted:
            example = next(instruct_iter, None)
            if example is not None:
                interleaved_messages.append(example["messages"])
            else:
                instruct_exhausted = True

    logger.info("Interleaved %d total examples", len(interleaved_messages))

    # Process each example
    all_input_ids = []
    all_labels = []
    all_attention_masks = []
    skipped = 0

    for messages in interleaved_messages:
        result = _process_example(messages, tokenizer, config.max_seq_len, config.W)
        if result is None:
            skipped += 1
            continue
        all_input_ids.append(result["input_ids"])
        all_labels.append(result["labels"])
        all_attention_masks.append(result["attention_mask"])

    logger.info("Processed %d examples, skipped %d (empty/no assistant tokens)",
                len(all_input_ids), skipped)

    # Split into train/val
    n_total = len(all_input_ids)
    n_val = max(1, int(n_total * config.val_fraction))
    n_train = n_total - n_val

    # Deterministic split (data is already shuffled via dataset.shuffle)
    train_dataset = CompactionDataset(
        input_ids=all_input_ids[:n_train],
        labels=all_labels[:n_train],
        attention_masks=all_attention_masks[:n_train],
    )
    val_dataset = CompactionDataset(
        input_ids=all_input_ids[n_train:],
        labels=all_labels[n_train:],
        attention_masks=all_attention_masks[n_train:],
    )

    logger.info("Split: %d train, %d val", len(train_dataset), len(val_dataset))

    return train_dataset, val_dataset


def analyze_sequence_lengths(
    config: CompactionConfig,
    tokenizer,
    num_samples: int = 2000,
) -> dict:
    """Analyze tokenized sequence lengths to recommend W/P/max_seq_len settings.

    Compaction training requires MULTIPLE block boundaries per sequence to be effective.
    If most sequences have only 1-2 blocks, compaction can't learn much. This function
    samples data, computes length statistics, and recommends configurations.

    Args:
        config: CompactionConfig with dataset paths.
        tokenizer: Qwen3-0.6B-Base tokenizer.
        num_samples: Number of examples to sample per dataset for analysis.

    Returns:
        dict with statistics and recommendations:
          think_lengths: list of token counts (Think-SFT sample)
          instruct_lengths: list of token counts (Instruct-SFT sample)
          combined_lengths: list of all token counts
          percentiles: {10, 25, 50, 75, 90, 95, 99} percentile lengths
          recommended_W: block size that gives >= 4 blocks for median sequence
          recommended_max_seq_len: covers 90th percentile
          blocks_per_seq: {W: {percentile: num_blocks}} for W in [64, 128, 256, 512]
    """
    import numpy as np

    think_ds = load_dataset(config.think_sft_path, split="train", streaming=True)
    instruct_ds = load_dataset(config.instruct_sft_path, split="train", streaming=True)

    think_lengths = []
    for i, ex in enumerate(think_ds):
        if i >= num_samples:
            break
        text = tokenizer.apply_chat_template(ex["messages"], tokenize=False)
        tokens = tokenizer(text, add_special_tokens=False)["input_ids"]
        think_lengths.append(len(tokens))

    instruct_lengths = []
    for i, ex in enumerate(instruct_ds):
        if i >= num_samples:
            break
        text = tokenizer.apply_chat_template(ex["messages"], tokenize=False)
        tokens = tokenizer(text, add_special_tokens=False)["input_ids"]
        instruct_lengths.append(len(tokens))

    combined = think_lengths + instruct_lengths
    combined_arr = np.array(combined)

    pcts = {p: int(np.percentile(combined_arr, p)) for p in [10, 25, 50, 75, 90, 95, 99]}

    # Compute blocks per sequence for different W values
    blocks_per_seq = {}
    for W_candidate in [64, 128, 256, 512]:
        blocks = {}
        for p_name, length in pcts.items():
            blocks[p_name] = length // W_candidate
        blocks_per_seq[W_candidate] = blocks

    # Recommend W: want >= 4 blocks at the median
    median_len = pcts[50]
    recommended_W = 256  # default
    for W_candidate in [512, 256, 128, 64]:
        if median_len // W_candidate >= 4:
            recommended_W = W_candidate
            break

    # Recommend max_seq_len: covers 90th percentile, rounded to multiple of recommended_W
    p90 = pcts[90]
    recommended_max_seq_len = ((p90 + recommended_W - 1) // recommended_W) * recommended_W
    recommended_max_seq_len = min(recommended_max_seq_len, 8192)  # cap for memory

    print("=" * 60)
    print("Sequence Length Analysis")
    print("=" * 60)
    print(f"\nThink-SFT ({len(think_lengths)} samples):")
    think_arr = np.array(think_lengths)
    print(f"  min={think_arr.min()}, median={int(np.median(think_arr))}, "
          f"mean={think_arr.mean():.0f}, max={think_arr.max()}")
    print(f"\nInstruct-SFT ({len(instruct_lengths)} samples):")
    inst_arr = np.array(instruct_lengths)
    print(f"  min={inst_arr.min()}, median={int(np.median(inst_arr))}, "
          f"mean={inst_arr.mean():.0f}, max={inst_arr.max()}")
    print(f"\nCombined percentiles:")
    for p, v in sorted(pcts.items()):
        print(f"  p{p}: {v} tokens")
    print(f"\nBlocks per sequence at each percentile:")
    print(f"  {'W':>6} | " + " | ".join(f"p{p:2d}" for p in sorted(pcts.keys())))
    print(f"  {'-'*6}-+-" + "-+-".join("-" * 4 for _ in pcts))
    for W_candidate in [64, 128, 256, 512]:
        row = f"  {W_candidate:>6} | "
        row += " | ".join(f"{blocks_per_seq[W_candidate][p]:>4}" for p in sorted(pcts.keys()))
        print(row)
    print(f"\nRecommendation:")
    print(f"  W = {recommended_W} (>= 4 blocks for median sequence)")
    print(f"  max_seq_len = {recommended_max_seq_len} (covers p90)")
    print(f"  P = {recommended_W // 8} (W/8 compression ratio)")
    print("=" * 60)

    return {
        "think_lengths": think_lengths,
        "instruct_lengths": instruct_lengths,
        "combined_lengths": combined,
        "percentiles": pcts,
        "recommended_W": recommended_W,
        "recommended_max_seq_len": recommended_max_seq_len,
        "blocks_per_seq": blocks_per_seq,
    }
