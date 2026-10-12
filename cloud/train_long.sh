#!/usr/bin/env bash
# Long GE-Sim 2.0 LoRA run (head-only), tuned on an RTX PRO 5000 48 GB:
#   batch 1 (one sample = ~8.4k tokens already saturates the GPU; larger batches were slower per sample),
#   --keep-blocks 22 (partial grad checkpointing, ~39 GB, 18% faster than full checkpointing).
# Restarts from <out>/last.pt up to 3 times if the process dies.
# usage: bash cloud/train_long.sh [out_dir] [extra train_gesim2.py args...]
#   (later args override the defaults below). Follow-up run from the first long run's best.pt:
#   bash cloud/train_long.sh runs/lora_r64_grip --init-lora runs/lora_r64_long/best.pt --band clip \
#       --grip-weight 3 --steps 2000 --lr 5e-5 --warmup 50 --patience 4 --save-every 100   # 100: Spot-safe
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"
OUT="${1:-runs/lora_r64_long}"; shift || true
export HF_HOME="$ROOT/.cache/hf" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$OUT"
for attempt in 1 2 3 4; do
    echo "=== attempt $attempt $(date -u)" >> "$OUT/train.log"
    .venv-gesim2/bin/python -u train_gesim2.py --out "$OUT" --resume auto \
        --steps 4000 --batch 1 --accum 4 --lr 1e-4 --warmup 100 --rank 64 --alpha 64 \
        --val-n 32 --val-every 250 --save-every 250 --patience 6 --min-delta 0.0001 \
        --workers 12 --keep-blocks 22 "$@" >> "$OUT/train.log" 2>&1 && break
    echo "=== exited with $? $(date -u), retrying in 30 s" >> "$OUT/train.log"
    sleep 30
done
echo "=== finished $(date -u)" >> "$OUT/train.log"
