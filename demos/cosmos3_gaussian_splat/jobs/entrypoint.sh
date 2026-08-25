#!/usr/bin/env bash
set -euo pipefail

IMAGE=""
PROMPT_FILE=""
OUTPUT_DIR=""
PROFILE="full"
MASK=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --image) IMAGE="$2"; shift 2 ;;
    --prompt-file) PROMPT_FILE="$2"; shift 2 ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    --profile) PROFILE="$2"; shift 2 ;;
    --mask) MASK="$2"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

if [[ -z "$IMAGE" || -z "$PROMPT_FILE" || -z "$OUTPUT_DIR" ]]; then
  echo "--image, --prompt-file, and --output-dir are required" >&2
  exit 2
fi

export HF_HOME="${HF_HOME:-/tmp/huggingface}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/uv-cache}"
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-/tmp/cosmos3-gsplat-venv}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
mkdir -p "$HF_HOME" "$UV_CACHE_DIR" "$OUTPUT_DIR"

if ! command -v uv >/dev/null 2>&1; then
  python -m pip install --no-cache-dir uv
fi

echo "GPU:"
nvidia-smi
echo "Installing locked project dependencies..."
uv sync --project /workspace --extra gpu --torch-backend=auto

PROMPT="$(<"$PROMPT_FILE")"
ARGS=(
  run
  --image "$IMAGE"
  --prompt "$PROMPT"
  --output-dir "$OUTPUT_DIR"
  --profile "$PROFILE"
)
if [[ -n "$MASK" ]]; then
  ARGS+=(--mask "$MASK")
fi

uv run --project /workspace cosmos3-gsplat "${ARGS[@]}" 2>&1 | tee "$OUTPUT_DIR/job.log"
