#!/bin/bash
# Orchestrate Phase 2b: wait for B+A, collect, report, launch K1+E
set -e
cd /pscratch/sd/s/siddart2/kv-self-compaction-phase2
source /pscratch/sd/s/siddart2/prime-rl/.venv/bin/activate
export PYTHONPATH=$PWD:$PYTHONPATH
export HF_DATASETS_CACHE=/pscratch/sd/s/siddart2/hf_cache
export HF_HOME=/pscratch/sd/s/siddart2/hf_cache

echo "=== Phase 2b Orchestrator ==="
echo "Waiting for B and A to produce final_metrics.json..."

# Wait for both runs to finish
while true; do
    b_done=$(test -f outputs/ddp_scaleup_W512/condition_B/final_metrics.json && echo 1 || echo 0)
    a_done=$(test -f outputs/ddp_scaleup_W512/condition_A/final_metrics.json && echo 1 || echo 0)

    b_step=$(tail -1 outputs/ddp_scaleup_W512/condition_B/train_loss.jsonl 2>/dev/null | python3 -c "import sys,json;print(json.loads(sys.stdin.read())['step'])" 2>/dev/null || echo "?")
    a_step=$(tail -1 outputs/ddp_scaleup_W512/condition_A/train_loss.jsonl 2>/dev/null | python3 -c "import sys,json;print(json.loads(sys.stdin.read())['step'])" 2>/dev/null || echo "?")

    echo "$(date +%H:%M) | B: step=$b_step done=$b_done | A: step=$a_step done=$a_done"

    if [ "$b_done" = "1" ] && [ "$a_done" = "1" ]; then
        echo "Both done!"
        break
    fi
    sleep 300
done

echo ""
echo "=== Collecting metrics and generating report ==="
python collect_and_report.py

echo ""
echo "=== Report Phase 1 complete ==="
echo "Results in: results/ and report/"
