#!/usr/bin/env bash
# Eval-only box: setup without the train set, then run eval_jobs.sh (stops the pod once the laptop pulled results).
# usage: setsid nohup bash cloud/run_eval_only.sh "ckpt|episodes|tag" ... > runs/run_eval_only.log 2>&1 < /dev/null &
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"
mkdir -p runs
SKIP_TRAIN=1 bash cloud/setup_gesim2.sh > setup.log 2>&1 || { echo "setup failed"; tail -20 setup.log; exit 1; }
echo "setup done $(date -u)"
bash cloud/eval_jobs.sh runs/EVAL_ONLY_DONE runs/EVAL_ONLY_PULLED "$@"
