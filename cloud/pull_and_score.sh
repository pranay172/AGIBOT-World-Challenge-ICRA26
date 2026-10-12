#!/usr/bin/env bash
# Laptop side of cloud/after_train.sh. Polls the instance; exits early (to get attention) if training
# crashed or the instance is unreachable for 30 min. Once the instance has written EVAL_DONE it pulls
# predictions + checkpoints, touches PULLED (the instance then stops itself), and scores locally.
# usage: bash cloud/pull_and_score.sh "ssh -p PORT root@HOST" /workspace/AGIBOT runs/lora_r64_long
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"
SSH="$1"; RDIR="$2"; RUN="$3"; NAME="$(basename "$RUN")"
host="${SSH##* }"; port="$(echo "$SSH" | grep -oE -- '-p [0-9]+' | cut -d' ' -f2)"
RSH="ssh -p $port"
q() { timeout 60 $SSH -o ConnectTimeout=20 "$@" 2>/dev/null | grep -vE "^Welcome|^Have fun"; }

fails=0; seen_exit=$(q "grep -c '=== exited' $RDIR/$RUN/train.log; true" | tail -1); seen_exit=${seen_exit:-0}
while true; do
    st=$(q "cd $RDIR/$RUN && echo ok; test -f EVAL_DONE && echo EVAL_DONE; grep -c '=== exited' train.log; true")
    if ! grep -q ok <<<"$st"; then
        fails=$((fails + 1)); [ $fails -ge 6 ] && { echo "UNREACHABLE for 30 min"; exit 2; }
    else
        fails=0
        grep -q EVAL_DONE <<<"$st" && break
        n_exit=$(tail -1 <<<"$st"); [[ "$n_exit" =~ ^[0-9]+$ ]] || n_exit=0
        [ "$n_exit" -gt "$seen_exit" ] && { echo "TRAINING CRASHED (auto-retrying):"; q "grep -B5 '=== exited' $RDIR/$RUN/train.log | tail -12"; exit 3; }
    fi
    sleep 300
done

echo "EVAL_DONE at $(date -u)"
mkdir -p "runs/$NAME" preds/val
rsync -az -e "$RSH" "$host:$RDIR/preds/val/${NAME}_*" preds/val/
# Only best.pt + logs by default: hosts bill transfer (step checkpoints 367 MB each, last.pt 1.1 GB).
# PULL_CKPTS="best.pt step001000.pt last.pt" to fetch more.
files="${PULL_CKPTS:-best.pt} train.log log.jsonl args.json"
rsync -az -e "$RSH" $(for f in $files; do echo "$host:$RDIR/$RUN/$f"; done) "runs/$NAME/"
rsync -az -e "$RSH" "$host:$RDIR/runs/after_train.log" "runs/$NAME/"
ls -la "runs/$NAME"
q "touch $RDIR/$RUN/PULLED" ; echo "PULLED marker set; instance will stop itself within ~30 s"

for d in preds/val/${NAME}_*/; do
    d="${d%/}"
    [ -d "${d}_scores" ] || .venv/bin/python -u eval/score.py --pred "$d" > "${d}_score.log" 2>&1
done
.venv/bin/python eval/compare.py
grep -E "val x0|new best|early stop" "runs/$NAME/train.log" | tail -20
sleep 60; q "echo still-reachable" || echo "instance no longer reachable (stopped)"
