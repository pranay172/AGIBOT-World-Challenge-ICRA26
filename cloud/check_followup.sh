#!/usr/bin/env bash
# Pre-flight checks for the follow-up run (run on the GPU box after setup_gesim2.sh, from the project dir).
#  A. Regression: --init-lora best.pt with --band orig --grip-weight 0 must reproduce the first long run's
#     logged step-3000 val x0-MSE [0.0260, 0.0176, 0.0132, 0.0102] (CPU-seeded noise, same GPU type).
#  B. Speed / memory of the follow-up config (--band clip --grip-weight 3) and its step-0 val.
#  C. Resume: continue B's 20-step run to 30 steps from last.pt.
# usage: bash cloud/check_followup.sh [extra train args, e.g. --keep-blocks 22] > runs/check_followup.log 2>&1
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"
source cloud/env.sh
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=.venv-gesim2/bin/python
rm -rf runs/check_a runs/check_b
echo "=== A regression"
$PY -u train_gesim2.py --out runs/check_a --init-lora runs/lora_r64_long/best.pt --band orig --grip-weight 0 \
    --steps 1 --val-n 32 --val-every 1000 --save-every 1000 --workers 12 "$@" 2>&1 | grep -E "initialised|step 0 val|Error|error"
echo "=== B follow-up config, 20 steps"
$PY -u train_gesim2.py --out runs/check_b --init-lora runs/lora_r64_long/best.pt --band clip --grip-weight 3 \
    --steps 20 --val-n 32 --val-every 1000 --save-every 20 --workers 12 --accum 4 --lr 5e-5 --warmup 50 "$@" 2>&1 \
    | grep -vE "Warning|warn"
echo "=== C resume to 30"
$PY -u train_gesim2.py --out runs/check_b --resume auto --band clip --grip-weight 3 \
    --steps 30 --val-n 32 --val-every 1000 --save-every 20 --workers 12 --accum 4 --lr 5e-5 --warmup 50 "$@" 2>&1 \
    | grep -E "resumed|step 30 |done|Error|error"
echo "=== CHECKS_DONE"
