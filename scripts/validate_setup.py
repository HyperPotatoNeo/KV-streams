#!/usr/bin/env python3
"""Pre-flight validation for KV Self-Compaction Phase 2.

Run ONCE before any training to verify all assumptions hold:
1. Qwen3-0.6B-Base loads with eager attention
2. Dolci datasets are accessible and have expected columns
3. Eager vs SDPA attention overhead is measured
4. Sequence length analysis recommends W/P/max_seq_len

Usage:
    python scripts/validate_setup.py
"""

import sys
import time

import torch


def check_model():
    """Verify Qwen3-0.6B-Base loads with eager attention."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("Loading Qwen3-0.6B-Base with eager attention...")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-0.6B-Base",
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen3-0.6B-Base", trust_remote_code=True
    )

    print(f"  Model type: {model.config.model_type}")
    print(f"  Layers: {model.config.num_hidden_layers}")
    print(f"  Hidden: {model.config.hidden_size}")
    print(f"  Q heads: {model.config.num_attention_heads}")
    print(f"  KV heads: {model.config.num_key_value_heads}")
    print(f"  Head dim: {getattr(model.config, 'head_dim', 'NOT SET')}")
    print(f"  Vocab: {model.config.vocab_size}")
    print(f"  BOS: {tokenizer.bos_token_id}, EOS: {tokenizer.eos_token_id}")
    print(f"  Pad: {tokenizer.pad_token_id}")

    # Test chat template
    messages = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi there"},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False)
    print(f"  Chat template output (first 200 chars): {text[:200]}")

    # Test forward
    if torch.cuda.is_available():
        model = model.cuda()
        x = torch.randint(0, 1000, (1, 32)).cuda()
        out = model(input_ids=x)
        print(f"  Forward OK, logits shape: {out.logits.shape}")
    else:
        print("  No GPU available, skipping forward pass test")

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("  CHECK PASSED: Model loads correctly\n")
    return True


def check_datasets():
    """Verify Dolci datasets accessible and have expected columns."""
    from datasets import load_dataset

    for name in ["allenai/Dolci-Think-SFT-7B", "allenai/Dolci-Instruct-SFT"]:
        print(f"Checking dataset: {name}")
        ds = load_dataset(name, split="train", streaming=True)
        sample = next(iter(ds))
        print(f"  Columns: {list(sample.keys())}")
        assert "messages" in sample, f"Missing 'messages' column in {name}"
        print(f"  messages[0]: {sample['messages'][0]}")
        print(f"  Num turns: {len(sample['messages'])}")
        print(f"  CHECK PASSED: {name}\n")

    return True


def profile_eager_vs_sdpa():
    """Profile eager attention vs SDPA to measure overhead."""
    if not torch.cuda.is_available():
        print("No GPU available, skipping attention profiling")
        return None

    from transformers import AutoModelForCausalLM

    results = {}
    for impl in ["eager", "sdpa"]:
        print(f"Profiling {impl} attention...")
        model = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen3-0.6B-Base",
            torch_dtype=torch.bfloat16,
            attn_implementation=impl,
        ).cuda()
        x = torch.randint(0, 1000, (4, 512)).cuda()

        # Warmup
        for _ in range(3):
            model(input_ids=x)
        torch.cuda.synchronize()

        # Time
        start = time.time()
        for _ in range(20):
            model(input_ids=x)
        torch.cuda.synchronize()
        elapsed = (time.time() - start) / 20
        results[impl] = elapsed
        print(f"  {impl}: {elapsed*1000:.1f}ms per forward")
        del model
        torch.cuda.empty_cache()

    ratio = results["eager"] / results["sdpa"]
    print(f"\nEager/SDPA ratio: {ratio:.2f}x")
    if ratio > 2.0:
        print("WARNING: Eager is >2x slower than SDPA.")
        print("Consider alternative bias injection strategy.")
    else:
        print("OK: Eager overhead acceptable.")
    print()
    return ratio


def analyze_lengths():
    """Run sequence length analysis to recommend W/P/max_seq_len."""
    from transformers import AutoTokenizer
    from src.config import CompactionConfig
    from src.data import analyze_sequence_lengths

    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen3-0.6B-Base", trust_remote_code=True
    )
    config = CompactionConfig()
    return analyze_sequence_lengths(config, tokenizer, num_samples=2000)


if __name__ == "__main__":
    print("=" * 60)
    print("KV Self-Compaction Phase 2: Pre-Flight Validation")
    print("=" * 60)

    print("\n[1/4] Checking model...")
    try:
        check_model()
    except Exception as e:
        print(f"  FAILED: {e}")
        sys.exit(1)

    print("[2/4] Checking datasets...")
    try:
        check_datasets()
    except Exception as e:
        print(f"  FAILED: {e}")
        sys.exit(1)

    print("[3/4] Profiling attention implementations...")
    try:
        profile_eager_vs_sdpa()
    except Exception as e:
        print(f"  FAILED: {e}")
        sys.exit(1)

    print("[4/4] Analyzing sequence lengths...")
    try:
        stats = analyze_lengths()
    except Exception as e:
        print(f"  FAILED: {e}")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("ALL CHECKS PASSED")
    print("=" * 60)
