#!/usr/bin/env bash
#SBATCH -J mem125weave-qwen25-locomo
#SBATCH -o /home/jwcao/xingzhuo/slm_memory_agent/code/MemWeaveMine/hpc/logs/memweave-qwen25-locomo-%j.out
#SBATCH -e /home/jwcao/xingzhuo/slm_memory_agent/code/MemWeaveMine/hpc/logs/memweave-qwen25-locomo-%j.err
#SBATCH -p compute
#SBATCH -N 1
#SBATCH -t 50:00:00
#SBATCH --gres=gpu:a100-pcie-40gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32gb

# 检查名字、端口和脚本内消融开关。

# Evaluate the current MemWeaveMine LoCoMo QA pipeline with Qwen2.5-3B.
# Inputs:
# - data/locomo10.json via DATASET_PATH or the default below
# - local Qwen2.5-3B-Instruct weights via MODEL_DIR
# - local embedding model via EMBED_DIR
# - local LLMLingua-2 model via LLMLINGUA_MODEL_DIR when its shell switch is true
# - vLLM backend on one local GPU, plus optional compression knobs:
#   LLMLINGUA_DEVICE, LLMLINGUA_RATE,
#   LLMLINGUA_TARGET_TOKEN, LLMLINGUA_MIN_TOKENS
# Outputs:
# - result JSON: hpc/results/memweave_qwen25_3b_locomo_eval_<variant>/job_<jobid>/results.json
# - unified incremental cost JSONL in the same directory: cost_statistics.jsonl
# - vLLM server log: hpc/logs/memweave_qwen25_3b_locomo_eval_vllm_<variant>_<jobid>.log
# - Slurm stdout/stderr:
#   hpc/logs/memweave-qwen25-locomo-<jobid>.out/.err
# - memory/retriever caches under hpc/cache/memweave_qwen25_3b_locomo_eval/<cache_tag>/

set -euo pipefail


source ~/.bashrc || true
source /home/jwcao/miniconda3/etc/profile.d/conda.sh
conda activate "${MEMWEAVE_CONDA_ENV:-${LOCOMEM_CONDA_ENV:-${AMEM_CONDA_ENV:-premem}}}"

ROOT="/home/jwcao/xingzhuo/slm_memory_agent/code/MemWeaveMine"
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

MODEL_DIR="${MODEL_DIR:-/home/jwcao/xingzhuo/mimic_dataset_building/models/Qwen2.5-3B-Instruct}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-3B-Instruct}"
EMBED_DIR="${EMBED_DIR:-/home/jwcao/xingzhuo/mimic_dataset_building/models/stella_en_400M_v5}"
DATASET_PATH="${DATASET_PATH:-$ROOT/data/locomo10.json}"

# Ablation switches. Edit only these true/false values before submitting the job.
# Slurm stores these assignments in the submitted batch-script snapshot.
LLMLINGUA_COMPONENT="llmlingua_pre_step1_compression"
AMEM_STEP1_COMPONENT="amem_after_step1_update_and_evolution"
AMEM_STEP2_COMPONENT="amem_after_step2_update_and_evolution"
CATEGORY3_PROMPT_COMPONENT="locomo_category3_use_existing_prompt"
STEP1_NOTE_PROMPT_COMPONENT="prestorage_step1_use_existing_note_prompt"
LLMLINGUA_PRE_STEP1_COMPRESSION=false
AMEM_AFTER_STEP1_UPDATE_AND_EVOLUTION=false
AMEM_AFTER_STEP2_UPDATE_AND_EVOLUTION=true
LOCOMO_CATEGORY3_USE_EXISTING_PROMPT=true
PRESTORAGE_STEP1_USE_EXISTING_NOTE_PROMPT=false

validate_ablation_boolean() {
  local component_name="$1"
  local component_value="$2"

  case "$component_value" in
    true|false)
      ;;
    *)
      echo "Invalid ablation component: $component_name must be true or false" >&2
      return 2
      ;;
  esac
}

validate_ablation_boolean "$LLMLINGUA_COMPONENT" "$LLMLINGUA_PRE_STEP1_COMPRESSION"
validate_ablation_boolean "$AMEM_STEP1_COMPONENT" "$AMEM_AFTER_STEP1_UPDATE_AND_EVOLUTION"
validate_ablation_boolean "$AMEM_STEP2_COMPONENT" "$AMEM_AFTER_STEP2_UPDATE_AND_EVOLUTION"
validate_ablation_boolean "$CATEGORY3_PROMPT_COMPONENT" "$LOCOMO_CATEGORY3_USE_EXISTING_PROMPT"
validate_ablation_boolean "$STEP1_NOTE_PROMPT_COMPONENT" "$PRESTORAGE_STEP1_USE_EXISTING_NOTE_PROMPT"

case "$LLMLINGUA_PRE_STEP1_COMPRESSION" in
  true)
    PRECOMPRESS="1"
    ;;
  false)
    PRECOMPRESS="0"
    ;;
  *)
    echo "Invalid ablation component: components.$LLMLINGUA_COMPONENT must be true or false" >&2
    exit 2
    ;;
esac

LLMLINGUA_MODEL_DIR="${LLMLINGUA_MODEL_DIR:-/home/jwcao/xingzhuo/mimic_dataset_building/models/llmlingua-2-bert-base-multilingual-cased-meetingbank}"
LLMLINGUA_DEVICE="${LLMLINGUA_DEVICE:-cpu}"
LLMLINGUA_RATE="${LLMLINGUA_RATE:-0.90}"
LLMLINGUA_TARGET_TOKEN="${LLMLINGUA_TARGET_TOKEN:--1}"
LLMLINGUA_MIN_TOKENS="${LLMLINGUA_MIN_TOKENS:-24}"

append_llmlingua_args() {
  local enabled="$1"

  case "$enabled" in
    true)
      if [[ -z "$LLMLINGUA_MODEL_DIR" ]]; then
        echo "components.$LLMLINGUA_COMPONENT=true requires LLMLINGUA_MODEL_DIR" >&2
        return 2
      fi
      if [[ ! -d "$LLMLINGUA_MODEL_DIR" ]]; then
        echo "LLMLingua model directory not found: $LLMLINGUA_MODEL_DIR" >&2
        return 2
      fi
      EXTRA_ARGS+=(
        --llmlingua-model-dir "$LLMLINGUA_MODEL_DIR"
        --llmlingua-device "$LLMLINGUA_DEVICE"
        --llmlingua-rate "$LLMLINGUA_RATE"
        --llmlingua-target-token "$LLMLINGUA_TARGET_TOKEN"
        --llmlingua-min-tokens "$LLMLINGUA_MIN_TOKENS"
      )
      ;;
    false)
      ;;
    *)
      echo "Invalid LLMLingua compression switch: $enabled" >&2
      return 2
      ;;
  esac
}


FORCE_REBUILD_CACHE="${FORCE_REBUILD_CACHE:-1}"
COST_PROFILING="${COST_PROFILING:-on}"
PORT="${PORT:-30125}"
RUN_NAME="${RUN_NAME:-memweave_qwen25_3b_locomo_eval}"
CACHE_DIR="${CACHE_DIR:-$ROOT/hpc/cache/memweave_qwen25_3b_locomo_eval}"
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

EXTRA_ARGS=()
if [[ "$FORCE_REBUILD_CACHE" == "1" ]]; then
  EXTRA_ARGS+=(--force-rebuild-cache)
fi
append_llmlingua_args "$LLMLINGUA_PRE_STEP1_COMPRESSION"

vllm_pid=""
cleanup() {
  local exit_status=$?
  # Compare runs need an unambiguous shell-level exit marker when Python/Slurm stops.
  echo "[DEBUG_QWEN_COMPARE] SHELL_EXIT job_id=${SLURM_JOB_ID:-unset} exit_status=$exit_status vllm_pid=${vllm_pid:-unset}"
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
echo "Ablation component: $LLMLINGUA_COMPONENT=$LLMLINGUA_PRE_STEP1_COMPRESSION"
echo "Ablation component: $AMEM_STEP1_COMPONENT=$AMEM_AFTER_STEP1_UPDATE_AND_EVOLUTION"
echo "Ablation component: $AMEM_STEP2_COMPONENT=$AMEM_AFTER_STEP2_UPDATE_AND_EVOLUTION"
echo "Ablation component: $CATEGORY3_PROMPT_COMPONENT=$LOCOMO_CATEGORY3_USE_EXISTING_PROMPT"
echo "Ablation component: $STEP1_NOTE_PROMPT_COMPONENT=$PRESTORAGE_STEP1_USE_EXISTING_NOTE_PROMPT"
echo "LLMLingua pre-Step-1 compression enabled: $LLMLINGUA_PRE_STEP1_COMPRESSION"
echo "LLMLingua dir: $LLMLINGUA_MODEL_DIR"
echo "Force rebuild cache: $FORCE_REBUILD_CACHE"
echo "Cache dir: $CACHE_DIR"
echo "Result dir: $RESULT_DIR"
echo "Result JSON: $RESULT_JSON"
echo "Cost statistics: $RESULT_DIR/cost_statistics.jsonl (enabled=$COST_PROFILING)"
echo "vLLM log: $VLLM_LOG"
# Log only resolved configuration; these values are not consumed by the pipeline.
echo "[DEBUG_QWEN_COMPARE] HPC_CONFIG job_id=${SLURM_JOB_ID:-unset} host=$(hostname) pwd=$PWD conda_env=${CONDA_DEFAULT_ENV:-unset} python=$(command -v python)"
echo "[DEBUG_QWEN_COMPARE] HPC_PATHS model_dir=$MODEL_DIR model_name=$MODEL_NAME dataset=$DATASET_PATH embedding_model=$EMBED_DIR llmlingua_model=$LLMLINGUA_MODEL_DIR cache_dir=$CACHE_DIR output=$RESULT_JSON"
echo "[DEBUG_QWEN_COMPARE] HPC_RUNTIME port=$PORT max_model_len=8192 vllm_generation_config=auto eval_cuda_visible_devices=empty llmlingua_device=$LLMLINGUA_DEVICE force_rebuild_cache=$FORCE_REBUILD_CACHE llmlingua_pre_step1_compression=$LLMLINGUA_PRE_STEP1_COMPRESSION"
echo "[DEBUG_QWEN_COMPARE] HPC_MAX_TOKENS default=${MEMWEAVE_MAX_TOKENS:-unset} step1=${MEMWEAVE_MAX_TOKENS_PRESTORAGE_STEP1_METADATA:-unset} step2=${MEMWEAVE_MAX_TOKENS_PRESTORAGE_STEP2_REASONING:-unset} memory=${MEMWEAVE_MAX_TOKENS_MEMORY_METADATA:-unset} evolution_decision=${MEMWEAVE_MAX_TOKENS_MEMORY_EVOLUTION_DECISION:-unset} evolution_strengthen=${MEMWEAVE_MAX_TOKENS_MEMORY_EVOLUTION_STRENGTHEN:-unset} evolution_update=${MEMWEAVE_MAX_TOKENS_MEMORY_EVOLUTION_UPDATE_NEIGHBOR:-unset} qa=${MEMWEAVE_MAX_TOKENS_QA_ANSWER_GENERATION:-unset}"

wait_for_vllm() {
  local port="$1"
  local tries="${2:-180}"
  for _ in $(seq 1 "$tries"); do
    if python - "$port" <<'PY' >/dev/null 2>&1
import sys
import urllib.request

port = sys.argv[1]
with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as resp:
    assert resp.status == 200
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

CUDA_VISIBLE_DEVICES="" python "$PYTHON_ENTRY" \
  --repo-root "$ROOT" \
  --dataset "$DATASET_PATH" \
  --model "$MODEL_NAME" \
  --backend vllm \
  --vllm-host http://127.0.0.1 \
  --vllm-port "$PORT" \
  --embedding-model-dir "$EMBED_DIR" \
  --llmlingua-pre-step1-compression "$LLMLINGUA_PRE_STEP1_COMPRESSION" \
  --amem-after-step1-update-and-evolution "$AMEM_AFTER_STEP1_UPDATE_AND_EVOLUTION" \
  --amem-after-step2-update-and-evolution "$AMEM_AFTER_STEP2_UPDATE_AND_EVOLUTION" \
  --locomo-category3-use-existing-prompt "$LOCOMO_CATEGORY3_USE_EXISTING_PROMPT" \
  --prestorage-step1-use-existing-note-prompt "$PRESTORAGE_STEP1_USE_EXISTING_NOTE_PROMPT" \
  --cache-dir "$CACHE_DIR" \
  --output "$RESULT_JSON" \
  --cost-profiling "$COST_PROFILING" \
  "${EXTRA_ARGS[@]}"

echo "Result file: $RESULT_JSON"
