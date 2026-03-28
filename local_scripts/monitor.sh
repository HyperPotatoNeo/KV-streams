#!/bin/bash
# Quick status check for all running experiments
echo "=== $(date) ==="
for dir in ddp_scaleup/condition_B ddp_scaleup/condition_D ddp_scaleup/condition_E ddp_scaleup_K4/condition_B ddp_scaleup_P32/condition_B ddp_scaleup/condition_A; do
  base="/pscratch/sd/s/siddart2/kv-self-compaction-phase2/outputs/$dir"
  name=$(echo $dir | sed 's|ddp_scaleup/condition_||;s|ddp_scaleup_||;s|/condition_B||')
  ckpts=$(ls $base/checkpoint-*/checkpoint.pt 2>/dev/null | wc -l)
  metrics=$(wc -l < $base/metrics.jsonl 2>/dev/null || echo 0)
  final=$(test -f $base/final_metrics.json && echo "DONE" || echo "running")
  echo "  $name: checkpoints=$ckpts metrics=$metrics status=$final"
done
echo "---"
squeue --me --format="%.10i %.8j %.6D %.20R" 2>&1
