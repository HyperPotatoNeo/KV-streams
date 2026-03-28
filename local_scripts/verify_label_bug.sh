#!/bin/bash
set -e
source /pscratch/sd/s/siddart2/prime-rl/.venv/bin/activate
cd /pscratch/sd/s/siddart2/kv-self-compaction-phase2
export PYTHONPATH=/pscratch/sd/s/siddart2/kv-self-compaction-phase2:$PYTHONPATH
export HF_DATASETS_CACHE=/pscratch/sd/s/siddart2/hf_cache
export HF_HOME=/pscratch/sd/s/siddart2/hf_cache

python -c "
from transformers import AutoTokenizer
from src.config import CompactionConfig
from src.data import load_data
import torch

config = CompactionConfig(max_examples_per_dataset=100)
tokenizer = AutoTokenizer.from_pretrained(config.model_name, trust_remote_code=True)
train_ds, val_ds = load_data(config, tokenizer)

# Check if labels == input_ids for valid positions
total_match = 0
total_valid = 0
total_shifted_match = 0

for i in range(min(10, len(val_ds))):
    example = val_ds[i]
    ids = example['input_ids']
    labels = example['labels']
    valid = labels != -100

    # Check identity: labels[j] == input_ids[j] ?
    match = (labels[valid] == ids[valid]).sum().item()
    total_match += match
    total_valid += valid.sum().item()

    # Check shifted: labels[j] == input_ids[j+1] ?
    shifted_valid = valid.clone()
    shifted_valid[-1] = False  # last position has no next token
    if shifted_valid.sum() > 0:
        shifted_match = (labels[shifted_valid] == ids[1:][shifted_valid[:-1]]).sum().item()
        total_shifted_match += shifted_match

print(f'Labels == input_ids (identity): {total_match}/{total_valid} = {total_match/total_valid:.3f}')
print(f'Labels == input_ids[+1] (shifted): {total_shifted_match}/{total_valid} = {total_shifted_match/total_valid:.3f}')
print()
print('If identity is ~1.0 and shifted is ~0.0, the labels are NOT shifted (BUG)')
print('If identity is ~0.0 and shifted is ~1.0, the labels ARE shifted (correct)')
"
