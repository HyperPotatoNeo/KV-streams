#!/usr/bin/env python3
"""Pre-compute tokenized dataset caches for all W values we use.
Just calls load_data which auto-caches on first call."""
import logging
import time
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

from transformers import AutoTokenizer
from src.config import CompactionConfig
from src.data import load_data, _cache_key

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B-Base", trust_remote_code=True)

for W in [128, 512]:
    config = CompactionConfig(W=W, P=W//8, max_examples_per_dataset=50000)
    key = _cache_key(config)
    print(f"\nW={W}: cache key = {key}")
    t0 = time.time()
    train_ds, val_ds = load_data(config, tokenizer)
    dt = time.time() - t0
    print(f"  {len(train_ds)} train + {len(val_ds)} val in {dt:.1f}s")

    # Verify a sample
    ex = train_ds[0]
    ids = ex["input_ids"]
    labels = ex["labels"]
    valid = labels != -100
    if valid.sum() > 0:
        # Check labels are shifted (labels[i] == input_ids[i+1])
        shifted_match = (labels[valid] == ids[1:][valid[:-1]]).float().mean().item()
        identity_match = (labels[valid] == ids[valid]).float().mean().item()
        print(f"  Label check: shifted_match={shifted_match:.3f} identity_match={identity_match:.3f}")
        assert shifted_match > 0.9, f"Labels not properly shifted! shifted={shifted_match}"
        assert identity_match < 0.2, f"Labels look like identity! identity={identity_match}"
        print(f"  ✓ Labels correctly shifted (next-token prediction)")

    # Check padding
    print(f"  Seq length: {ids.shape[0]}, padded to multiple of W={W}: {ids.shape[0] % W == 0}")
    assert ids.shape[0] % W == 0, "Not padded to multiple of W"

print("\nAll caches verified ✓")
