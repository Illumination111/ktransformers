# DeepSeek-V3.2 推理实验监控

每次使用 sglang 启动 DeepSeek-V3.2 MoE 异构推理时，自动创建以 **日期时间命名** 的实验子目录，记录：

- LLM 请求的 IO 参数（input/output token 长度、延迟、吞吐等）
- 服务器 GPU 显存和 CPU 内存的初始状态及推理过程中的变化

---

## 前置条件

**1. CPU 权重量化**（首次运行前执行一次）

将 FP8 GPU 权重转换为 AMXINT4 CPU 权重：

```bash
cd /mnt/data/wbw/ktransformers/kt-kernel

python scripts/convert_cpu_weights.py \
    --input-path /mnt/data/models/DeepSeek-V3.2 \
    --input-type fp8 \
    --output /mnt/data/models/DeepSeek-V3.2-INT4 \
    --quant-method int4 \
    --cpuinfer-threads 64 \
    --threadpool-count 2 \
    --no-merge-safetensor
```

**2. 依赖安装**

```bash
pip install psutil nvidia-ml-py matplotlib numpy
```

---

## 快速启动

```bash
cd /mnt/data/wbw/ktransformers

python monitor/DeepSeek-V3.2/launch_with_monitor.py \
    --host 0.0.0.0 \
    --port 30000 \
    --model /mnt/data/models/DeepSeek-V3.2 \
    --trust-remote-code \
    --mem-fraction-static 0.98 \
    --chunked-prefill-size 4096 \
    --max-running-requests 32 \
    --max-total-tokens 40000 \
    --served-model-name DeepSeek-V3.2 \
    --enable-mixed-chunk \
    --attention-backend triton \
    --tensor-parallel-size 1 \
    --enable-p2p-check \
    --disable-shared-experts-fusion \
    --skip-server-warmup \
    --kt-method AMXINT4 \
    --kt-weight-path /mnt/data/models/DeepSeek-V3.2-INT4 \
    --kt-cpuinfer 64 \
    --kt-threadpool-count 2 \
    --kt-num-gpu-experts 1 \
    --kt-max-deferred-experts-per-token 2
```

### 监控专属参数（可选）

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--monitor-interval SECONDS` | 内存采样间隔（秒） | `5` |
| `--experiment-dir PATH` | 覆盖默认实验目录路径 | 按时间自动命名 |

---

## 交互式对话客户端

服务启动后，在另一个终端运行：

```bash
cd /mnt/data/wbw/ktransformers

# 默认连接 localhost:30000，最大输出 512 tokens
python monitor/DeepSeek-V3.2/chat_client.py

# 设置最大输出 2048 tokens
python monitor/DeepSeek-V3.2/chat_client.py --max-tokens 2048
```

会话内可用命令：

| 命令 | 说明 |
|------|------|
| `/tokens 1024` | 动态修改最大输出 token 数 |
| `/clear` | 清空对话历史 |
| `/history` | 查看对话历史摘要 |
| `/system <text>` | 修改 system prompt |
| `/quit` | 退出 |

---

## 目录结构

```
monitor/DeepSeek-V3.2/
├── launch_with_monitor.py       # 主入口脚本
├── memory_monitor.py            # 后台内存监控模块
├── chat_client.py               # 交互式对话客户端
├── plot_experiment.py           # 实验数据可视化脚本
├── README.md                    # 本文件
│
└── 20260611_220000/             # 实验目录（YYYYMMDD_HHMMSS）
    ├── server_args.json         # 启动参数快照
    ├── startup_memory.json      # 启动时 GPU + CPU 内存状态
    ├── memory_timeline.jsonl    # 推理过程中的周期性内存采样
    ├── sglang-request-metrics-*.log  # SGLang 请求指标
    ├── experiment_summary.json  # 实验摘要
    └── plots/
        ├── memory_timeline.png  # GPU/CPU 内存时间线图
        └── request_metrics.png  # Token 统计与延迟图
```

---

## 硬件要求

| 资源 | 最低配置 | 测试配置 |
|------|---------|---------|
| GPU | NVIDIA L20 48GB（或 ≥27 GB 可用显存） | NVIDIA L20 48GB × N |
| CPU | Intel Xeon AMX（Sapphire Rapids+） | Xeon Platinum 8488C |
| 系统内存 | 350 GB（INT4 量化） | 2 TB DDR5 |
| 存储 | ~1 TB（FP8 + INT4 权重） | — |

> 本机配置（8 × RTX 4090 / 128 线程 / 2 NUMA 节点）显存不足（单卡 24 GB < 27 GB 最低要求），需要权重转换完成后验证实际占用。

---

## 数据格式说明

### `memory_timeline.jsonl`（每 5 秒一条）

```jsonl
{"label": "periodic", "timestamp": "2026-06-11T22:00:05.000", "elapsed_seconds": 5.0,
 "gpu": [{"index": 0, "name": "NVIDIA L20", "used_mb": 27000, "total_mb": 49152, "percent": 54.9}],
 "cpu": {"used_gb": 380.0, "total_gb": 2048.0, "available_gb": 1668.0, "percent": 18.6}}
```

### `sglang-request-metrics-*.log`（每请求一条）

关键字段：`prompt_tokens`、`completion_tokens`、`e2e_latency`、`cached_tokens`、`finish_reason`

---

## 手动绘图

```bash
# 对指定实验目录绘图
python monitor/DeepSeek-V3.2/plot_experiment.py monitor/DeepSeek-V3.2/20260611_220000/

# 自动选择最新实验目录
python monitor/DeepSeek-V3.2/plot_experiment.py
```
