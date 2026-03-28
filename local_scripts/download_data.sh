#!/bin/bash
set -e
source /pscratch/sd/s/siddart2/prime-rl/.venv/bin/activate
cd /pscratch/sd/s/siddart2/kv-self-compaction-phase2
export PYTHONPATH=/pscratch/sd/s/siddart2/kv-self-compaction-phase2:$PYTHONPATH
export HF_DATASETS_CACHE=/pscratch/sd/s/siddart2/hf_cache

python -c "
from datasets import load_dataset
import time

t0 = time.time()
print('Downloading Think-SFT...')
ds1 = load_dataset('allenai/Dolci-Think-SFT-7B', split='train')
print(f'Think-SFT: {len(ds1)} examples in {time.time()-t0:.0f}s')

t1 = time.time()
print('Downloading Instruct-SFT...')
ds2 = load_dataset('allenai/Dolci-Instruct-SFT', split='train')
print(f'Instruct-SFT: {len(ds2)} examples in {time.time()-t1:.0f}s')

print(f'Total: {time.time()-t0:.0f}s')
" 2>&1
