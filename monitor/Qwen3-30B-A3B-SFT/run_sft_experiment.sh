#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LLAMA_FACTORY_DIR="${LLAMA_FACTORY_DIR:-/mnt/data/wbw/LLaMA-Factory}"
PYTHON_BIN="${PYTHON_BIN:-/mnt/data/wbw/miniconda3/envs/Kllama/bin/python}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
ACCELERATE_CONFIG="${ACCELERATE_CONFIG:-examples/ktransformers/accelerate/fsdp2_kt_int8.yaml}"
TRAIN_YAML="${TRAIN_YAML:-examples/ktransformers/train_lora/qwen3_30b_a3b_lora_sft_kt.yaml}"
MONITOR_INTERVAL="${MONITOR_INTERVAL:-5}"

export CUDA_VISIBLE_DEVICES
export USE_KT="${USE_KT:-1}"
export ACCELERATE_USE_KT="${ACCELERATE_USE_KT:-true}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export PYTHONUNBUFFERED=1

extra_args=()
if [[ -n "${PRETRAIN_METRICS_JSON:-}" ]]; then
  extra_args+=(--pretrain-metrics-json "${PRETRAIN_METRICS_JSON}")
fi
if [[ -n "${FINETUNED_METRICS_JSON:-}" ]]; then
  extra_args+=(--finetuned-metrics-json "${FINETUNED_METRICS_JSON}")
fi

cd "${LLAMA_FACTORY_DIR}"

exec "${PYTHON_BIN}" "${SCRIPT_DIR}/launch_sft_with_monitor.py" \
  --offline \
  --monitor-interval "${MONITOR_INTERVAL}" \
  --workdir "${LLAMA_FACTORY_DIR}" \
  --config-file "${ACCELERATE_CONFIG}" \
  --train-yaml "${TRAIN_YAML}" \
  "${extra_args[@]}" \
  "$@"

