# Qwen3-30B-A3B SFT 训练监控

这个目录用于监控 LLaMA-Factory + KTransformers 的 Qwen3-30B-A3B LoRA SFT 过程。每次启动都会创建一个时间戳实验目录，记录训练命令、环境、日志、CPU 内存、GPU 显存、GPU 利用率和 accelerate 多进程树资源占用。

## 快速启动

Qwen3-30B-A3B 的本地模型目录约 `57G`，AMXINT8 CPU 权重目录约 `31G`，比 DeepSeek-V3.2 更适合在这台服务器上先验证 SFT 流程。默认使用 4GPU INT8 KT 配置：

推荐直接运行本目录的一键 SFT 实验脚本，它会完成 LoRA SFT 启动、资源监控、训练日志采集、训练产物解析和可视化：

```bash
cd /mnt/data/wbw/ktransformers

monitor/Qwen3-30B-A3B-SFT/run_sft_experiment.sh
```

默认 SFT 命令：

```bash
/mnt/data/wbw/miniconda3/envs/Kllama/bin/accelerate launch \
  --config_file examples/ktransformers/accelerate/fsdp2_kt_int8.yaml \
  -m llamafactory.cli train \
  examples/ktransformers/train_lora/qwen3_30b_a3b_lora_sft_kt.yaml
```

常用覆盖方式：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
ACCELERATE_CONFIG=examples/ktransformers/accelerate/fsdp2_kt_int8.yaml \
TRAIN_YAML=examples/ktransformers/train_lora/qwen3_30b_a3b_lora_sft_kt.yaml \
PRETRAIN_METRICS_JSON=/path/to/qwen_before.json \
FINETUNED_METRICS_JSON=/path/to/qwen_after.json \
monitor/Qwen3-30B-A3B-SFT/run_sft_experiment.sh \
  --experiment-dir /mnt/data/wbw/ktransformers/monitor/Qwen3-30B-A3B-SFT/qwen_sft_full
```

```bash
cd /mnt/data/wbw/LLaMA-Factory

unset ALL_PROXY all_proxy

CUDA_VISIBLE_DEVICES=0,1,2,3 \
USE_KT=1 \
ACCELERATE_USE_KT=true \
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
/mnt/data/wbw/miniconda3/envs/Kllama/bin/python \
  /mnt/data/wbw/ktransformers/monitor/Qwen3-30B-A3B-SFT/launch_sft_with_monitor.py \
  --offline \
  --monitor-interval 5
```

默认监控命令等价于：

```bash
/mnt/data/wbw/miniconda3/envs/Kllama/bin/accelerate launch \
  --config_file examples/ktransformers/accelerate/fsdp2_kt_int8.yaml \
  -m llamafactory.cli train \
  examples/ktransformers/train_lora/qwen3_30b_a3b_lora_sft_kt.yaml
```

也可以监控自定义命令：

```bash
/mnt/data/wbw/miniconda3/envs/Kllama/bin/python \
  /mnt/data/wbw/ktransformers/monitor/Qwen3-30B-A3B-SFT/launch_sft_with_monitor.py \
  --workdir /mnt/data/wbw/LLaMA-Factory \
  -- \
  /mnt/data/wbw/miniconda3/envs/Kllama/bin/accelerate launch \
    --config_file examples/ktransformers/accelerate/fsdp2_kt_int8.yaml \
    -m llamafactory.cli train \
    examples/ktransformers/train_lora/qwen3_30b_a3b_lora_sft_kt.yaml
```

## 输出文件

```text
monitor/Qwen3-30B-A3B-SFT/YYYYMMDD_HHMMSS/
├── sft_args.json
├── startup_memory.json
├── process.json
├── train.log
├── resource_timeline.jsonl
├── trainer_state_tail.json
├── sft_metrics_summary.json
├── experiment_summary.json
└── plots/
    ├── sft_resource_timeline.png
    ├── sft_summary_metrics.png
    ├── sft_training_loss.png
    └── sft_task_performance_delta.png  # 若提供微调前后任务指标
```

## 采样内容

`resource_timeline.jsonl` 每条记录包含：

- `cpu`: 系统 CPU 内存 used/total/available/percent
- `gpu`: 每张 GPU 的显存、GPU 利用率、显存控制器利用率、GPU 进程
- `process_tree`: accelerate 根进程及所有子进程的 RSS、VMS、CPU、线程数、命令行

## 手动绘图

```bash
python /mnt/data/wbw/ktransformers/monitor/Qwen3-30B-A3B-SFT/plot_sft_experiment.py

python /mnt/data/wbw/ktransformers/monitor/Qwen3-30B-A3B-SFT/plot_sft_experiment.py \
  /mnt/data/wbw/ktransformers/monitor/Qwen3-30B-A3B-SFT/YYYYMMDD_HHMMSS
```

## 常用参数

| 参数 | 说明 | 默认值 |
| --- | --- | --- |
| `--monitor-interval` | 资源采样间隔秒数 | `5` |
| `--experiment-dir` | 指定实验输出目录 | 时间戳目录 |
| `--workdir` | 训练命令工作目录 | `/mnt/data/wbw/LLaMA-Factory` |
| `--config-file` | accelerate 配置 | `fsdp2_kt_int8.yaml` |
| `--train-yaml` | LLaMA-Factory 训练配置 | `qwen3_30b_a3b_lora_sft_kt.yaml` |
| `--offline` | 设置 HF/Transformers 离线模式 | 关闭 |
| `--dry-run` | 只生成元数据，不启动训练 | 关闭 |
| `--pretrain-metrics-json` | 微调前任务评测指标 JSON，用于计算性能差异 | 空 |
| `--finetuned-metrics-json` | 微调后任务评测指标 JSON，用于计算性能差异 | 空 |

`sft_metrics_summary.json` 会汇总：

- LoRA 峰值显存、主机内存和训练进程树 RSS
- 可训练参数量、总参数量和 LoRA 参数比例
- 平均 step 时间、samples/s、tokens/s、loss 曲线与收敛摘要
- 训练日志中可解析的 forward/backward/update 阶段耗时
- 可选的微调前后任务指标差异

## 前置依赖

建议安装：

```bash
/mnt/data/wbw/miniconda3/envs/Kllama/bin/python -m pip install psutil nvidia-ml-py matplotlib numpy
```

没有 `pynvml` 时会 fallback 到 `nvidia-smi`；没有 `psutil` 时仍会记录系统内存，但不会记录训练进程树。
