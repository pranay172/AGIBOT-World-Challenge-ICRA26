#!/usr/bin/env bash
# One command on a fresh GPU pod (after sync_to_cloud.sh push): setup -> follow-up training -> eval -> stop pod.
# Pre-flight checks (cloud/check_followup.sh) passed on an RTX PRO 5000 on 2026-10-12, so they are skipped here.
# On GPUs with < 45 GB (e.g. RTX 5090 32 GB) it first probes the largest --keep-blocks whose 10-step peak stays <= 29.5 GB;
# KEEP_BLOCKS=<n> skips the probe. On Vast it also creates .venv-vast (vastai CLI) for the self-stop.
# usage: setsid nohup bash cloud/run_followup.sh > runs/run_followup.log 2>&1 < /dev/null &
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"
mkdir -p runs
RUN=runs/lora_r64_grip
bash cloud/setup_gesim2.sh > setup.log 2>&1 || { echo "setup failed"; tail -20 setup.log; exit 1; }
until grep -q "tar exit=0" dl_train.log 2>/dev/null; do
    grep -q "tar exit=[1-9]" dl_train.log 2>/dev/null && { echo "download failed"; exit 1; }
    sleep 20
done
echo "data ready: $(ls WorldModel_Dataset/train | wc -l) clips $(date -u)"
source cloud/env.sh
if [ -n "${CONTAINER_ID:-}" ] && [ ! -x .venv-vast/bin/vastai ]; then
    uv venv .venv-vast --python 3.10 -q && uv pip install --python .venv-vast/bin/python -q vastai
    echo "vastai cli: $(.venv-vast/bin/vastai --version 2>&1 | tail -1)"
fi
GPU_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
if [ -z "${KEEP_BLOCKS:-}" ]; then
    KEEP_BLOCKS=22
    if [ "$GPU_MB" -lt 45000 ]; then
        KEEP_BLOCKS=0
        for kb in 16 13 10 6 3; do
            rm -rf runs/probe_kb
            PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True .venv-gesim2/bin/python -u train_gesim2.py \
                --out runs/probe_kb --init-lora runs/lora_r64_long/best.pt --band clip --grip-weight 3 \
                --steps 10 --val-n 1 --val-every 1000 --save-every 1000 --workers 12 --keep-blocks $kb \
                > runs/probe_kb_$kb.log 2>&1
            rc=$?; line=$(grep -E "^step [0-9]+ loss" runs/probe_kb_$kb.log | tail -1)
            echo "probe keep-blocks $kb: exit $rc ${line:-no step line}"
            mem=$(grep -oE "mem [0-9.]+G" <<<"$line" | grep -oE "[0-9.]+")
            if [ $rc -eq 0 ] && [ -n "$mem" ] && awk "BEGIN{exit !($mem <= 29.5)}"; then KEEP_BLOCKS=$kb; break; fi
        done
    fi
fi
echo "using --keep-blocks $KEEP_BLOCKS (GPU $GPU_MB MiB)"
# Evaluate best.pt on all 189 val clips after training (comparable with the first run's 0.7936).
CKPTS="$RUN/best.pt" EPISODES="" setsid nohup bash cloud/after_train.sh "$RUN" > runs/after_train.log 2>&1 < /dev/null &
bash cloud/train_long.sh "$RUN" --init-lora runs/lora_r64_long/best.pt --band clip --grip-weight 3 \
    --steps 2000 --lr 5e-5 --warmup 50 --patience 4 --save-every 100 --keep-blocks "$KEEP_BLOCKS"
echo "training script returned $(date -u)"
