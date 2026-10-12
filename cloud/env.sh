# Source on any box (fresh pod on the same network volume included): everything lives under the project dir.
#   source cloud/env.sh
_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export UV_INSTALL_DIR="$_ROOT/.cache/bin" UV_NO_MODIFY_PATH=1
export UV_CACHE_DIR="$_ROOT/.cache/uv" UV_PYTHON_INSTALL_DIR="$_ROOT/.cache/uv-python"
export UV_LINK_MODE=copy  # network volumes do not support hard links into the cache
export HF_HOME="$_ROOT/.cache/hf"
export PATH="$_ROOT/.cache/bin:$PATH"
unset _ROOT
# RunPod passes RUNPOD_* (pod id, pod-scoped API key for runpodctl) only to the container's PID 1, not to sshd
# sessions; import them so runpodctl and the self-stop work from scripts started over SSH.
if [ -z "${RUNPOD_POD_ID:-}" ] && [ -r /proc/1/environ ] && tr '\0' '\n' < /proc/1/environ | grep -q '^RUNPOD_POD_ID='; then
    while IFS= read -r _kv; do export "$_kv"; done < <(tr '\0' '\n' < /proc/1/environ | grep -E '^RUNPOD_[A-Z_]+=')
    unset _kv
fi
