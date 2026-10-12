#!/usr/bin/env bash
# Laptop side of run_eval_only.sh: wait for runs/EVAL_ONLY_DONE, pull preds/val/<tag>, touch EVAL_ONLY_PULLED (the
# pod stops itself; check the log line after it), score, and combine with the hard-19 result into a 189-clip score.
# usage: bash cloud/pull_eval_only.sh "ssh -p PORT root@HOST" /workspace/AGIBOT <tag>
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"
SSH="$1"; RDIR="$2"; TAG="$3"
host="${SSH##* }"; port="$(echo "$SSH" | grep -oE -- '-p [0-9]+' | cut -d' ' -f2)"
q() { timeout 60 $SSH -o ConnectTimeout=20 "$@" 2>/dev/null | grep -vE "^Welcome|^Have fun"; }
fails=0
while true; do
    st=$(q "test -f $RDIR/runs/EVAL_ONLY_DONE && echo DONE; echo ok")
    grep -q DONE <<<"$st" && break
    if grep -q ok <<<"$st"; then fails=0; else fails=$((fails + 1)); [ $fails -ge 10 ] && { echo UNREACHABLE; exit 2; }; fi
    sleep 180
done
echo "EVAL_ONLY_DONE at $(date -u)"
rsync -az -e "ssh -p $port" "$host:$RDIR/preds/val/$TAG" "$host:$RDIR/preds/val/$TAG.log" preds/val/
n=$(find "preds/val/$TAG" -name '*.jpg' | wc -l); echo "pulled $n frames"
[ "$n" -gt 0 ] && q "touch $RDIR/runs/EVAL_ONLY_PULLED && echo touched" | grep -q touched && echo "EVAL_ONLY_PULLED set"
[ -d "preds/val/${TAG}_scores" ] || .venv/bin/python -u eval/score.py --pred "preds/val/$TAG" > "preds/val/${TAG}_score.log" 2>&1
.venv/bin/python - "$TAG" <<'PY'
import sys, json, pandas as pd
def load(name):
    d = pd.read_csv(f"preds/val/{name}_scores/per_trial.csv"); d["ep"] = d.task_id.astype(str) + "/" + d.episode_id.astype(str); return d
def summ(d):
    s = ((d.psnr.clip(0, 35) / 35) + d.scene_consistency + d.ndtw) / 3
    return f"n={len(d):3d} PSNR {d.psnr.mean():.2f} SC {d.scene_consistency.mean():.3f} nDTW {d.ndtw.mean():.3f} zeros {(d.ndtw == 0).sum():2d} score {s.mean():.4f}"
base = load("lora_best_full189"); hard = [l.strip() for l in open("eval/val_hard.txt") if l.strip()]
rest = load(sys.argv[1])
print("rest170 baseline:     ", summ(base[~base.ep.isin(hard)]))
print("rest170 new step1000: ", summ(rest))
print("FULL189 baseline:     ", summ(base))
new = pd.concat([load("lora_r64_grip_step001000_hard"), rest])
print("FULL189 new step1000: ", summ(new))
for t, g in new.groupby("task_id"):
    b = base[base.task_id == t]
    print(f"  task {t}: score {(((g.psnr.clip(0,35)/35)+g.scene_consistency+g.ndtw)/3).mean():.3f} (was {(((b.psnr.clip(0,35)/35)+b.scene_consistency+b.ndtw)/3).mean():.3f})  nDTW {g.ndtw.mean():.3f} (was {b.ndtw.mean():.3f})")
PY
sleep 90; q "echo still-reachable" | grep -q still && echo "WARNING: pod still running after self-stop request" || echo "pod no longer reachable (stopped)"
