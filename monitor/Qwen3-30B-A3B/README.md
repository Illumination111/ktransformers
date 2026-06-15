# Qwen3-30B-A3B 推理实验监控

每次使用 sglang 启动 Qwen3-30B-A3B MoE 异构推理时，自动创建以 **日期时间命名** 的实验子目录，记录：

- LLM 请求的 IO 参数（input/output token 长度、延迟等）
- 服务器 GPU 显存和 CPU 内存的初始状态及推理过程中的变化

---

## 快速开始

推荐直接运行本目录的一键推理实验脚本，它会启动 Qwen3-30B-A3B kt-kernel 异构推理服务，等待服务就绪，记录模型加载后内存，然后自动完成输入/输出长度矩阵 benchmark 并退出服务：

```bash
cd /mnt/data/wbw/ktransformers

monitor/Qwen3-30B-A3B/run_inference_experiment.sh
```

默认测试矩阵：

| 项目 | 默认值 |
| --- | --- |
| 输入 token 长度 | `128,512,2048,4096` |
| 输出 token 长度 | `128,512,1024` |
| 每组重复次数 | `1` |
| 服务端口 | `30001` |
| CPU 权重目录 | `/mnt/data/wbw/models/Qwen3-30B-A3B-CPU` |

常用覆盖方式：

```bash
PORT=31001 \
MODEL_PATH=/mnt/data/models/Qwen3-30B-A3B \
KT_WEIGHT_PATH=/mnt/data/wbw/models/Qwen3-30B-A3B-CPU \
BENCHMARK_INPUT_LENGTHS=128,1024,4096,8192 \
BENCHMARK_OUTPUT_LENGTHS=128,512,1024 \
monitor/Qwen3-30B-A3B/run_inference_experiment.sh \
  --experiment-dir /mnt/data/wbw/ktransformers/monitor/Qwen3-30B-A3B/qwen_infer_full
```

脚本内部使用的 kt-kernel/SGLang 启动命令等价于：

```bash
python monitor/Qwen3-30B-A3B/launch_with_monitor.py \
  --benchmark \
  --benchmark-input-lengths 128,512,2048,4096 \
  --benchmark-output-lengths 128,512,1024 \
  --exit-after-benchmark \
  --host 0.0.0.0 \
  --port 30001 \
  --model /mnt/data/models/Qwen3-30B-A3B \
  --trust-remote-code \
  --mem-fraction-static 0.92 \
  --chunked-prefill-size 4096 \
  --max-running-requests 1 \
  --max-total-tokens 8192 \
  --served-model-name Qwen3-30B-A3B \
  --enable-mixed-chunk \
  --kt-method AMXINT8 \
  --kt-weight-path /mnt/data/wbw/models/Qwen3-30B-A3B-CPU \
  --kt-cpuinfer 64 \
  --kt-threadpool-count 2 \
  --kt-num-gpu-experts 32 \
  --kt-max-deferred-experts-per-token 2
```

将原本直接调用 `python -m sglang.launch_server` 的命令替换为调用 `launch_with_monitor.py`，参数完全相同：

```bash
python /path/to/ktransformers/monitor/Qwen3-30B-A3B/launch_with_monitor.py \
    --model /mnt/data/models/Qwen3-30B-A3B \
    --trust-remote-code \
    --mem-fraction-static 0.92 \
    --chunked-prefill-size 4096 \
    --served-model-name Qwen3-30B-A3B \
    --enable-mixed-chunk \
    --kt-method AMXINT8 \
    --kt-weight-path /mnt/data/wbw/models/Qwen3-30B-A3B-CPU \
    --kt-cpuinfer 64 \
    --kt-threadpool-count 2 \
    --kt-num-gpu-experts 32 \
    --kt-max-deferred-experts-per-token 2
```

### 监控专属参数（可选）

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--monitor-interval SECONDS` | 内存采样间隔（秒） | `5` |
| `--experiment-dir PATH` | 覆盖默认实验目录路径 | 按时间自动命名 |
| `--benchmark` | 服务就绪后自动运行输入/输出长度矩阵 benchmark | 关闭 |
| `--benchmark-input-lengths` | 输入 token 长度列表，如 `128,512,2048` | `128,512,2048` |
| `--benchmark-output-lengths` | 输出 token 长度列表，如 `128,512` | `128,512` |
| `--benchmark-repetitions` | 每组长度重复次数 | `1` |
| `--exit-after-benchmark` | benchmark 完成后自动停止服务 | 关闭 |

示例：

```bash
python /path/to/ktransformers/monitor/Qwen3-30B-A3B/launch_with_monitor.py \
    --benchmark \
    --benchmark-input-lengths 128,512,2048,4096 \
    --benchmark-output-lengths 128,512,1024 \
    --exit-after-benchmark \
    --model /mnt/data/models/Qwen3-30B-A3B \
    ...  # 其余 sglang/kt 参数保持不变
```

---

## 目录结构

```
monitor/Qwen3-30B-A3B/
├── launch_with_monitor.py       # 主入口脚本
├── memory_monitor.py            # 后台内存监控模块
├── README.md                    # 本文件
│
├── 20260610_165000/             # 实验 1（YYYYMMDD_HHMMSS）
│   ├── server_args.json         # 启动参数快照
│   ├── startup_memory.json      # 启动时 GPU + CPU 内存状态
│   ├── loaded_memory.json       # 模型加载完成后的 GPU + CPU 内存状态（启用 --benchmark 时）
│   ├── memory_timeline.jsonl    # 推理过程中的周期性内存采样
│   ├── inference_benchmark.jsonl # TTFT / Prefill / Decode / 吞吐 benchmark（启用 --benchmark 时）
│   ├── inference_metrics_summary.json # 汇总后的可读指标
│   ├── sglang-request-metrics-20260610_16.log  # SGLang 请求指标（按小时分文件）
│   └── experiment_summary.json  # 实验摘要（运行时长、总请求数）
│
└── 20260611_093000/             # 实验 2
    └── ...
```

`plots/` 中会生成：

- `memory_timeline.png`: 显存和主机内存时间线
- `request_metrics.png`: 请求级 token、端到端时延和输出吞吐
- `inference_benchmark_metrics.png`: TTFT、Prefill 输入吞吐、Decode TPOT/ITL、输出吞吐（启用 `--benchmark` 时）

---

## 数据格式说明

### `server_args.json`

记录本次实验的所有启动参数：

```json
{
  "start_time": "2026-06-10T16:50:00",
  "sglang_argv": ["--model", "/mnt/data/models/Qwen3-30B-A3B", "--kt-method", "AMXINT8", "..."],
  "monitor_interval_secs": 5.0,
  "python_executable": "/usr/bin/python3",
  "working_directory": "/mnt/data/wbw/ktransformers"
}
```

### `startup_memory.json`

服务启动前的内存基准快照：

```json
{
  "label": "startup",
  "timestamp": "2026-06-10T16:50:01.234",
  "gpu": [
    {
      "index": 0,
      "name": "NVIDIA A100-SXM4-80GB",
      "used_mb": 512,
      "total_mb": 81920,
      "percent": 0.6
    }
  ],
  "cpu": {
    "used_gb": 32.1,
    "total_gb": 256.0,
    "available_gb": 223.9,
    "percent": 12.5
  }
}
```

### `memory_timeline.jsonl`

每隔 `--monitor-interval` 秒追加一行，记录实验过程中的内存变化：

```jsonl
{"label": "periodic", "timestamp": "2026-06-10T16:50:06.345", "elapsed_seconds": 5.1, "gpu": [{"index": 0, "name": "NVIDIA A100-SXM4-80GB", "used_mb": 65432, "total_mb": 81920, "percent": 79.9}], "cpu": {"used_gb": 180.2, "total_gb": 256.0, "available_gb": 75.8, "percent": 70.4}}
{"label": "periodic", "timestamp": "2026-06-10T16:50:11.456", "elapsed_seconds": 10.2, "gpu": [...], "cpu": {...}}
{"label": "final",    "timestamp": "2026-06-10T17:30:00.000", "elapsed_seconds": 2399.0, "gpu": [...], "cpu": {...}}
```

字段说明：

| 字段 | 说明 |
|------|------|
| `label` | `periodic`（周期采样）或 `final`（结束时采样） |
| `timestamp` | ISO 8601 时间戳（毫秒精度） |
| `elapsed_seconds` | 距实验开始的秒数 |
| `gpu[].used_mb` | GPU 显存使用量（MB） |
| `gpu[].total_mb` | GPU 显存总量（MB） |
| `gpu[].percent` | GPU 显存使用率（%） |
| `cpu.used_gb` | 系统内存使用量（GB） |
| `cpu.total_gb` | 系统内存总量（GB） |
| `cpu.percent` | 系统内存使用率（%） |

### `sglang-request-metrics-YYYYMMDD_HH.log`

由 SGLang 原生文件导出器写入，每个完成的请求一行 JSON：

```jsonl
{"request_parameters": "{\"text\": \"...\", \"sampling_params\": {...}}", "prompt_tokens": 128, "completion_tokens": 512, "ttft_s": 0.35, "finish_reason": {"type": "stop"}}
```

关键字段：

| 字段 | 说明 |
|------|------|
| `prompt_tokens` | 输入 token 数 |
| `completion_tokens` | 输出 token 数 |
| `ttft_s` | 首 token 延迟（秒） |
| `finish_reason` | 结束原因（`stop` / `length` 等） |
| `request_parameters` | 完整请求参数（JSON 字符串） |

### `experiment_summary.json`

实验结束后写入的摘要：

```json
{
  "end_time": "2026-06-10T17:30:00",
  "elapsed_seconds": 2400.0,
  "sglang_exit_code": 0,
  "total_logged_requests": 1024
}
```

---

## 依赖

| 包 | 用途 | 安装 |
|-----|------|------|
| `psutil` | CPU 内存采样 | `pip install psutil`（通常已安装） |
| `nvidia-ml-py` | GPU 显存采样（优先） | `pip install nvidia-ml-py` |

若 `nvidia-ml-py` 未安装，自动 fallback 到调用 `nvidia-smi` subprocess。  
若 `psutil` 未安装，自动 fallback 到读取 `/proc/meminfo`。

---

## 实现说明

- `launch_with_monitor.py`：主控脚本，所有 sglang/kt 参数完整透传，通过 `--export-metrics-to-file --export-metrics-to-file-dir` 将 SGLang 原生指标写入实验目录
- `memory_monitor.py`：后台守护线程，生命周期与 sglang 子进程绑定；sglang 退出时自动记录 `final` 快照并停止
- 信号处理：`SIGTERM` / `SIGINT` 自动转发给 sglang 子进程，确保优雅退出
