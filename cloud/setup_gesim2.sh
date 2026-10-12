#!/usr/bin/env bash
# One-shot GE-Sim 2.0 setup on a fresh CUDA 12.8+ Linux box (RunPod / GCP).
# First push the code from the laptop:   bash cloud/sync_to_cloud.sh push root@HOST:PORT /workspace/AGIBOT
# Then on the box, from that dir:        bash cloud/setup_gesim2.sh
# Starts the 38 GB train download in the background (log: dl_train.log) while it builds the venv.
# Everything (uv, its managed CPython, caches, venv, data, weights) stays inside the project dir, so on a
# RunPod network volume a later pod (CPU for setup, GPU for training) reuses it as is. Re-runnable.
# Works without a GPU (CPU pod): only the final CUDA check is skipped.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"
TRAIN_URL=https://huggingface.co/datasets/agibot-world/AgiBotWorldChallenge-2026/resolve/main/WorldModel/train.tar.gz

source cloud/env.sh
df -h "$ROOT" | tail -1
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || echo "no GPU on this box"

# 1. Train data: parallel HTTP ranges streamed into tar (one HF stream can be < 1 MB/s).
# SKIP_TRAIN=1 for an inference/eval-only box (needs only the pushed validation inputs).
if [ "${SKIP_TRAIN:-0}" != 1 ] && { [ ! -d WorldModel_Dataset/train ] || [ "$(ls WorldModel_Dataset/train | wc -l)" -lt 32000 ]; }; then
    rm -rf WorldModel_Dataset/train WorldModel_Dataset/train.spool
    nohup python3 -u cloud/dl_parallel_tar.py "$TRAIN_URL" WorldModel_Dataset/train --strip 6 \
        --workers 32 --chunk-mb 64 --window 96 > dl_train.log 2>&1 &
    echo "train download started (pid $!), tail -f dl_train.log"
fi

# 2. venv. uv-managed CPython ships Python.h, which Triton's JIT (liger kernels) needs.
[ -x "$UV_INSTALL_DIR/uv" ] || curl -LsSf https://astral.sh/uv/install.sh | sh  # installs into $UV_INSTALL_DIR
[ -x .venv-gesim2/bin/python ] || UV_PYTHON_PREFERENCE=only-managed uv venv .venv-gesim2 --python 3.10
VPY="$ROOT/.venv-gesim2/bin/python"
uv pip install --python "$VPY" torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
uv pip install --python "$VPY" -e "GE-Sim-V2[server]" diffusers==0.41.0 liger-kernel h5py "huggingface_hub[cli]"
"$VPY" -c "import torch; ok = torch.cuda.is_available(); print('torch', torch.__version__, 'cuda ok:', ok, torch.cuda.get_device_name(0) if ok else '')"

# 3. GE-Sim 2.0 weights (ungated, ~4.5 GB).
CK=weights/gesim2/checkpoints/gesim_community_v2.0.1_g01op_distill_2B
if [ ! -f "$CK/model.safetensors" ]; then
    "$ROOT/.venv-gesim2/bin/hf" download agibot-world/Genie-Envisioner-Sim-v2.0 --local-dir weights/gesim2
fi
ls "$CK"
echo "SETUP_DONE"
