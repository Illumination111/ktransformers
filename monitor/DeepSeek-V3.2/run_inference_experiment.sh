#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
KTRANSFORMERS_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-/mnt/data/wbw/miniconda3/envs/kt-kernel/bin/python}"
MODEL_PATH="${MODEL_PATH:-/mnt/data/models/DeepSeek-V3.2}"
KT_WEIGHT_PATH="${KT_WEIGHT_PATH:-/mnt/data/wbw/models/DeepSeek-V3.2_CPU}"
KT_METHOD="${KT_METHOD:-AMXINT8}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-30000}"
MONITOR_INTERVAL="${MONITOR_INTERVAL:-2}"
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.92}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-2}"
MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-32}"
MAX_TOTAL_TOKENS="${MAX_TOTAL_TOKENS:-40000}"
KT_NUM_GPU_EXPERTS="${KT_NUM_GPU_EXPERTS:-2}"
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
  --benchmark-model "DeepSeek-V3.2" \
  --benchmark-tokenizer "${MODEL_PATH}" \
  --exit-after-benchmark \
  --host "${HOST}" \
  --port "${PORT}" \
  --model "${MODEL_PATH}" \
  --trust-remote-code \
  --mem-fraction-static "${MEM_FRACTION_STATIC}" \
  --chunked-prefill-size 4096 \
  --max-running-requests "${MAX_RUNNING_REQUESTS}" \
  --max-total-tokens "${MAX_TOTAL_TOKENS}" \
  --served-model-name DeepSeek-V3.2 \
  --enable-mixed-chunk \
  --attention-backend triton \
  --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}" \
  --enable-p2p-check \
  --disable-shared-experts-fusion \
  --skip-server-warmup \
  --kt-method "${KT_METHOD}" \
  --kt-weight-path "${KT_WEIGHT_PATH}" \
  --kt-cpuinfer 64 \
  --kt-threadpool-count 2 \
  --kt-num-gpu-experts "${KT_NUM_GPU_EXPERTS}" \
  --kt-max-deferred-experts-per-token 2 \
  "$@"
