#!/usr/bin/env bash
# Run a list of "ckpt|episodes_file|tag" eval jobs, touch $DONE_MARKER, then stop the instance once the laptop
# touched $PULLED_MARKER (or after GRACE_S). usage: bash cloud/eval_jobs.sh <done_marker> <pulled_marker> job...
# An empty episodes_file means all episodes. Env: INPUT_ROOT (default validation), OUT_BASE (default preds/val),
# N_PRED (default 1; the test submission needs 3).
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"
source cloud/env.sh
DONE="$1"; PULLED="$2"; shift 2; GRACE_S="${GRACE_S:-7200}"
log() { echo "[$(date -u +%F' '%T)] $*"; }
INPUT_ROOT="${INPUT_ROOT:-WorldModel_Dataset/validation/info_dataset}"; OUT_BASE="${OUT_BASE:-preds/val}"
mkdir -p "$OUT_BASE"
for job in "$@"; do
    IFS='|' read -r ck eps tag <<<"$job"
    log "eval $ck on ${eps:-all of $INPUT_ROOT} -> $OUT_BASE/$tag"
    .venv-gesim2/bin/python -u run_challenge_gesim2.py --input-root "$INPUT_ROOT" \
        --output-root "$OUT_BASE/$tag" ${eps:+--episodes-file "$eps"} --n-pred "${N_PRED:-1}" \
        --fps-mode upsample30 --skip-existing --lora "$ck" > "$OUT_BASE/$tag.log" 2>&1
    log "  exit $? frames $(find "$OUT_BASE/$tag" -name '*.jpg' | wc -l)"
done
touch "$DONE"; log "done; waiting up to $GRACE_S s for $PULLED"
t0=$(date +%s)
until [ -f "$PULLED" ] || [ $(( $(date +%s) - t0 )) -ge "$GRACE_S" ]; do sleep 30; done
log "stopping instance (pulled: $([ -f "$PULLED" ] && echo yes || echo no))"
if [ -n "${RUNPOD_POD_ID:-}" ]; then runpodctl stop pod "$RUNPOD_POD_ID"
else .venv-vast/bin/vastai stop instance "$CONTAINER_ID" --api-key "$CONTAINER_API_KEY"; fi
