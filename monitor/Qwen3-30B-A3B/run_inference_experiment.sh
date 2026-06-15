#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
KTRANSFORMERS_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-/mnt/data/wbw/miniconda3/envs/kt-kernel/bin/python}"
MODEL_PATH="${MODEL_PATH:-/mnt/data/models/Qwen3-30B-A3B}"
KT_WEIGHT_PATH="${KT_WEIGHT_PATH:-/mnt/data/wbw/models/Qwen3-30B-A3B-CPU}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
MONITOR_INTERVAL="${MONITOR_INTERVAL:-2}"
BENCHMARK_INPUT_LENGTHS="${BENCHMARK_INPUT_LENGTHS:-128,512,2048,4096}"
BENCHMARK_OUTPUT_LENGTHS="${BENCHMARK_OUTPUT_LENGTHS:-128,512,1024}"
BENCHMARK_REPETITIONS="${BENCHMARK_REPETITIONS:-1}"
BENCHMARK_WARMUP="${BENCHMARK_WARMUP:-1}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"

export CUDA_VISIBLE_DEVICES
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

cd "${KTRANSFORMERS_DIR}"

"${PYTHON_BIN}" -c "import sglang, kt_kernel" >/dev/null

exec "${PYTHON_BIN}" "${SCRIPT_DIR}/launch_with_monitor.py" \
  --monitor-interval "${MONITOR_INTERVAL}" \
  --benchmark \
  --benchmark-input-lengths "${BENCHMARK_INPUT_LENGTHS}" \
  --benchmark-output-lengths "${BENCHMARK_OUTPUT_LENGTHS}" \
  --benchmark-repetitions "${BENCHMARK_REPETITIONS}" \
  --benchmark-warmup "${BENCHMARK_WARMUP}" \
  --benchmark-host "127.0.0.1" \
  --benchmark-port "${PORT}" \
  --benchmark-model "Qwen3-30B-A3B" \
  --benchmark-tokenizer "${MODEL_PATH}" \
  --exit-after-benchmark \
  --host "${HOST}" \
  --port "${PORT}" \
  --model "${MODEL_PATH}" \
  --trust-remote-code \
  --mem-fraction-static 0.92 \
  --chunked-prefill-size 4096 \
  --served-model-name Qwen3-30B-A3B \
  --enable-mixed-chunk \
  --kt-method AMXINT8 \
  --kt-weight-path "${KT_WEIGHT_PATH}" \
  --kt-cpuinfer 64 \
  --kt-threadpool-count 2 \
  --kt-num-gpu-experts 32 \
  --kt-max-deferred-experts-per-token 2 \
  "$@"
