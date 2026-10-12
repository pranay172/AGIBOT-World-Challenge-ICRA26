#!/usr/bin/env bash
# Runs on the Vast instance next to a training run. When training ends (finished, early stop, or out
# of retries) it evaluates checkpoints on the 30-clip val subset, then stops this instance:
# right away once the laptop has pulled the results (it touches $RUN/PULLED), or after a grace period.
# Vast: stops the instance with its instance-scoped key. RunPod: `runpodctl stop pod` with the pod-scoped key
# every pod ships with; if the pod cannot be stopped (pods on a network volume may only be terminable) it is
# terminated only when RUNPOD_TERMINATE=1 was set when launching this script (all results are on the volume).
# usage: setsid nohup bash cloud/after_train.sh runs/lora_r64_long > runs/after_train.log 2>&1 &
# env: CKPTS="a.pt b.pt" overrides the checkpoint list; EPISODES=<file> (default eval/val_subset30.txt,
#      empty = all 189 validation clips).
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"
RUN="${1:-runs/lora_r64_long}"; GRACE_S="${GRACE_S:-10800}"
source cloud/env.sh
log() { echo "[$(date -u +%F' '%T)] $*"; }

log "waiting for $RUN to finish"
until grep -q "=== finished" "$RUN/train.log" 2>/dev/null; do sleep 60; done
log "training ended: $(grep -E 'early stop|^done|=== exited' "$RUN/train.log" | tail -2 | tr '\n' ' ')"

# best.pt, steps 1000/2000, and the final step checkpoint (deduplicated).
final=$(ls "$RUN"/step*.pt 2>/dev/null | sort | tail -1)
mkdir -p preds/val
cks=()
EPISODES="${EPISODES-eval/val_subset30.txt}"; EP_ARGS=(); [ -n "$EPISODES" ] && EP_ARGS=(--episodes-file "$EPISODES")
for c in ${CKPTS:-"$RUN/best.pt" "$RUN/step001000.pt" "$RUN/step002000.pt" "$final"}; do
    [ -n "$c" ] && [ -f "$c" ] && [[ ! " ${cks[*]-} " == *" $c "* ]] && cks+=("$c")
done
for c in "${cks[@]}"; do
    tag="$(basename "$(dirname "$c")")_$(basename "$c" .pt)"
    log "eval $c -> preds/val/$tag"
    .venv-gesim2/bin/python -u run_challenge_gesim2.py --input-root WorldModel_Dataset/validation/info_dataset \
        --output-root "preds/val/$tag" ${EP_ARGS[@]+"${EP_ARGS[@]}"} --fps-mode upsample30 \
        --skip-existing --lora "$c" > "preds/val/$tag.log" 2>&1
    log "  exit $? frames $(find "preds/val/$tag" -name '*.jpg' | wc -l)"
done
touch "$RUN/EVAL_DONE"
log "EVAL_DONE; waiting up to $GRACE_S s for the laptop to pull results"

t0=$(date +%s)
until [ -f "$RUN/PULLED" ] || [ $(( $(date +%s) - t0 )) -ge "$GRACE_S" ]; do sleep 30; done
log "stopping instance (pulled: $([ -f "$RUN/PULLED" ] && echo yes || echo no))"
if [ -n "${RUNPOD_POD_ID:-}" ]; then
    if runpodctl stop pod "$RUNPOD_POD_ID"; then
        log "stop requested"
    elif [ "${RUNPOD_TERMINATE:-0}" = 1 ]; then
        log "stop failed; terminating pod (RUNPOD_TERMINATE=1)"; runpodctl remove pod "$RUNPOD_POD_ID"
    else
        log "stop failed and RUNPOD_TERMINATE is not 1: pod left running"
    fi
else
    .venv-vast/bin/vastai stop instance "$CONTAINER_ID" --api-key "$CONTAINER_API_KEY"
    log "stop requested"
fi
