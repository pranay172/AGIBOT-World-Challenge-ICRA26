#!/usr/bin/env bash
# Copy the code + the small input splits (no train videos) to a cloud box, or pull predictions back.
# Usage:
#   bash cloud/sync_to_cloud.sh push user@host[:port] [remote_dir]
#   bash cloud/sync_to_cloud.sh pull user@host[:port] [remote_dir]   # fetch preds/ back for local scoring
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ACTION="$1"; HOSTPORT="$2"; REMOTE_DIR="${3:-AGIBOT}"
HOST="${HOSTPORT%%:*}"; PORT=22
[[ "$HOSTPORT" == *:* ]] && PORT="${HOSTPORT##*:}"
SSH="ssh -p $PORT"

if [ "$ACTION" = push ]; then
    rsync -az --info=progress2 -e "$SSH" \
        --exclude '.git' --exclude '__pycache__' --exclude '.venv*' --exclude 'outputs' \
        --exclude 'WorldModel_Dataset/train' --exclude 'WorldModel_Dataset/val_raw' \
        --exclude 'WorldModel_Dataset/validation/gt_dataset' --exclude 'WorldModel' \
        --exclude 'EWMBench' --exclude 'preds' --exclude 'runs' --exclude 'weights' --exclude 'archive' --exclude '.cache' \
        "$ROOT/" "$HOST:$REMOTE_DIR/"
    # LoRA checkpoints to use on the box (--init-lora / --lora); PUSH_CKPTS="a.pt b.pt" overrides
    (cd "$ROOT" && rsync -az --relative -e "$SSH" ${PUSH_CKPTS:-runs/lora_r64_long/best.pt} "$HOST:$REMOTE_DIR/")
elif [ "$ACTION" = pull ]; then
    rsync -az --info=progress2 -e "$SSH" "$HOST:$REMOTE_DIR/preds/" "$ROOT/preds/"
else
    echo "unknown action $ACTION"; exit 1
fi
