#!/usr/bin/env bash
#SBATCH -J memweave-qwen35-locomo
#SBATCH -o /home/jwcao/xingzhuo/slm_memory_agent/code/MemWeaveMine/hpc/logs/memweave-qwen35-locomo-%j.out
#SBATCH -e /home/jwcao/xingzhuo/slm_memory_agent/code/MemWeaveMine/hpc/logs/memweave-qwen35-locomo-%j.err
#SBATCH -p compute
#SBATCH -N 1
#SBATCH -t 50:00:00
#SBATCH --gres=gpu:a100-pcie-40gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32gb

# Evaluate the current MemWeaveMine LoCoMo QA pipeline with Qwen3.5-4B.

set -euo pipefail

PROJECT_ROOT="/home/jwcao/xingzhuo/slm_memory_agent/code/MemWeaveMine"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_ROOT="$PROJECT_ROOT"

source ~/.bashrc || true
source /home/jwcao/miniconda3/etc/profile.d/conda.sh
conda activate "${MEMWEAVE_CONDA_ENV:-${LOCOMEM_CONDA_ENV:-${AMEM_CONDA_ENV:-premem-qwen35}}}"

ROOT="${MEMWEAVE_ROOT:-$DEFAULT_ROOT}"
cd "$ROOT"
mkdir -p "$ROOT/hpc/logs" "$ROOT/hpc/results" "$ROOT/hpc/cache" "$ROOT/.hf_cache"

export HF_HOME="${HF_HOME:-$ROOT/.hf_cache}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
for proxy_key in http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY; do
  unset "$proxy_key" || true
done

MODEL_DIR="${MODEL_DIR:-/home/jwcao/xingzhuo/mimic_dataset_building/models/Qwen3.5-4B}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3.5-4B}"
EMBED_DIR="${EMBED_DIR:-/home/jwcao/xingzhuo/mimic_dataset_building/models/stella_en_400M_v5}"
DATASET_PATH="${DATASET_PATH:-$ROOT/data/locomo10.json}"
PRECOMPRESS="1"
LLMLINGUA_MODEL_DIR="${LLMLINGUA_MODEL_DIR:-/home/jwcao/xingzhuo/mimic_dataset_building/models/llmlingua-2-bert-base-multilingual-cased-meetingbank}"
LLMLINGUA_DEVICE="${LLMLINGUA_DEVICE:-cpu}"
LLMLINGUA_RATE="${LLMLINGUA_RATE:-0.90}"
LLMLINGUA_TARGET_TOKEN="${LLMLINGUA_TARGET_TOKEN:--1}"
LLMLINGUA_MIN_TOKENS="${LLMLINGUA_MIN_TOKENS:-24}"


FORCE_REBUILD_CACHE="${FORCE_REBUILD_CACHE:-1}"
COST_PROFILING="${COST_PROFILING:-on}"
PORT="${PORT:-30000}"
RUN_NAME="${RUN_NAME:-memweave_qwen35_4b_locomo_eval}"
CACHE_DIR="${CACHE_DIR:-$ROOT/hpc/cache/memweave_qwen35_4b_locomo_eval}"
PYTHON_ENTRY="${PYTHON_ENTRY:-$ROOT/hpc/memweave_qwen25_3b_locomo_eval.py}"
if [[ "$PRECOMPRESS" == "1" ]]; then
  RUN_VARIANT="${RUN_VARIANT:-precompress}"
else
  RUN_VARIANT="${RUN_VARIANT:-plain}"
fi
JOB_ID="${SLURM_JOB_ID:-manual-$(date +%Y%m%d-%H%M%S)-$$}"
RESULT_DIR="${RESULT_DIR:-$ROOT/hpc/results/${RUN_NAME}_${RUN_VARIANT}/job_${JOB_ID}}"
mkdir -p "$RESULT_DIR"
RESULT_JSON="${RESULT_JSON:-$RESULT_DIR/results.json}"
VLLM_LOG="${VLLM_LOG:-$ROOT/hpc/logs/${RUN_NAME}_vllm_${RUN_VARIANT}_${JOB_ID}.log}"

vllm_pid=""
cleanup() {
  if [[ -n "${vllm_pid:-}" ]] && kill -0 "$vllm_pid" 2>/dev/null; then
    kill "$vllm_pid" 2>/dev/null || true
    wait "$vllm_pid" 2>/dev/null || true
  fi
}
trap cleanup EXIT

echo "Run name: $RUN_NAME"
echo "Variant: $RUN_VARIANT"
echo "Dataset: $DATASET_PATH"
echo "Model dir: $MODEL_DIR"
echo "Embedding dir: $EMBED_DIR"
echo "LLMLingua dir: $LLMLINGUA_MODEL_DIR"
echo "Force rebuild cache: $FORCE_REBUILD_CACHE"
echo "Cache dir: $CACHE_DIR"
echo "Result dir: $RESULT_DIR"
echo "Result JSON: $RESULT_JSON"
echo "Cost statistics: $RESULT_DIR/cost_statistics.jsonl (enabled=$COST_PROFILING)"
echo "vLLM log: $VLLM_LOG"

wait_for_vllm() {
  local port="$1"
  local tries="${2:-180}"
  for _ in $(seq 1 "$tries"); do
    if python - "$port" <<'PY' >/dev/null 2>&1
import json
import sys
import urllib.request

port = sys.argv[1]
with urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=2) as resp:
    payload = json.loads(resp.read().decode("utf-8"))
assert payload.get("data") is not None
PY
    then
      return 0
    fi
    sleep 5
  done
  return 1
}

CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_DIR" \
  --served-model-name "$MODEL_NAME" \
  --host 127.0.0.1 \
  --port "$PORT" \
  --dtype float16 \
  --gpu-memory-utilization 0.90 \
  --max-model-len 8192 \
  --trust-remote-code \
  > "$VLLM_LOG" 2>&1 &
vllm_pid=$!

wait_for_vllm "$PORT"

EXTRA_ARGS=()
if [[ "$FORCE_REBUILD_CACHE" == "1" ]]; then
  EXTRA_ARGS+=(--force-rebuild-cache)
fi
if [[ "$PRECOMPRESS" == "1" ]]; then
  if [[ -n "$LLMLINGUA_MODEL_DIR" ]]; then
    EXTRA_ARGS+=(
      --enable-precompress
      --llmlingua-model-dir "$LLMLINGUA_MODEL_DIR"
      --llmlingua-device "$LLMLINGUA_DEVICE"
      --llmlingua-rate "$LLMLINGUA_RATE"
      --llmlingua-target-token "$LLMLINGUA_TARGET_TOKEN"
      --llmlingua-min-tokens "$LLMLINGUA_MIN_TOKENS"
    )
  else
    echo "PRECOMPRESS=1 but LLMLINGUA_MODEL_DIR is empty; falling back to original text."
  fi
fi

CUDA_VISIBLE_DEVICES="" python "$PYTHON_ENTRY" \
  --repo-root "$ROOT" \
  --dataset "$DATASET_PATH" \
  --model "$MODEL_NAME" \
  --backend vllm \
  --vllm-host http://127.0.0.1 \
  --vllm-port "$PORT" \
  --embedding-model-dir "$EMBED_DIR" \
  --cache-dir "$CACHE_DIR" \
  --output "$RESULT_JSON" \
  --cost-profiling "$COST_PROFILING" \
  "${EXTRA_ARGS[@]}"

echo "Result file: $RESULT_JSON"
