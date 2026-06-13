# DeepSeek-V3.2 SFT 训练监控

这个目录用于监控 LLaMA-Factory + KTransformers 的 DeepSeek-V3.2 LoRA SFT 过程。每次启动都会创建一个时间戳实验目录，记录训练命令、环境、日志、CPU 内存、GPU 显存、GPU 利用率和 accelerate 多进程树资源占用。

## 快速启动

DeepSeek-V3.2 的 AMXINT8 per-NUMA CPU 权重目录约 `644G`。在 4 个 accelerate rank 下，rank0 会持有完整 KT CPU MoE wrapper，其他 rank 仍会叠加各自的模型/加载内存，容易在 KT 初始化后段被系统 `SIGKILL`。建议先用 1GPU 低内存配置验证完整初始化：

```bash
cd /mnt/data/wbw/LLaMA-Factory

unset ALL_PROXY all_proxy

CUDA_VISIBLE_DEVICES=0 \
USE_KT=1 \
ACCELERATE_USE_KT=true \
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
/mnt/data/wbw/miniconda3/envs/Kllama/bin/python \
  /mnt/data/wbw/ktransformers/monitor/DeepSeek-V3.2-SFT/launch_sft_with_monitor.py \
  --offline \
  --monitor-interval 5 \
  --config-file examples/ktransformers/accelerate/fsdp2_kt_int8_1gpu_lowmem.yaml
```

如需回到 4GPU 配置，可以使用默认命令，但需要预期 CPU 内存峰值会显著上升。

```bash
cd /mnt/data/wbw/LLaMA-Factory

unset ALL_PROXY all_proxy

CUDA_VISIBLE_DEVICES=0,1,2,3 \
USE_KT=1 \
ACCELERATE_USE_KT=true \
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
/mnt/data/wbw/miniconda3/envs/Kllama/bin/python \
  /mnt/data/wbw/ktransformers/monitor/DeepSeek-V3.2-SFT/launch_sft_with_monitor.py \
  --offline \
  --monitor-interval 5
```

默认监控命令等价于：

```bash
/mnt/data/wbw/miniconda3/envs/Kllama/bin/accelerate launch \
  --config_file examples/ktransformers/accelerate/fsdp2_kt_int8.yaml \
  -m llamafactory.cli train \
  examples/ktransformers/train_lora/deepseek_v32_lora_sft_kt.yaml
```

也可以监控自定义命令：

```bash
/mnt/data/wbw/miniconda3/envs/Kllama/bin/python \
  /mnt/data/wbw/ktransformers/monitor/DeepSeek-V3.2-SFT/launch_sft_with_monitor.py \
  --workdir /mnt/data/wbw/LLaMA-Factory \
  -- \
  /mnt/data/wbw/miniconda3/envs/Kllama/bin/accelerate launch \
    --config_file examples/ktransformers/accelerate/fsdp2_kt_int8.yaml \
    -m llamafactory.cli train \
    examples/ktransformers/train_lora/deepseek_v32_lora_sft_kt.yaml
```

## 输出文件

```text
monitor/DeepSeek-V3.2-SFT/YYYYMMDD_HHMMSS/
├── sft_args.json              # 命令、环境变量、版本、nvidia-smi、git 信息
├── startup_memory.json        # 启动前资源快照
├── process.json               # accelerate 根进程 PID
├── train.log                  # 训练 stdout/stderr 完整日志
├── resource_timeline.jsonl    # 周期资源采样
├── trainer_state_tail.json    # 若 output_dir 中存在 trainer_state.json，则保存尾部日志
├── experiment_summary.json    # 退出码、耗时、output_dir、训练产物摘要
└── plots/
    └── sft_resource_timeline.png
```

## 采样内容

`resource_timeline.jsonl` 每条记录包含：

- `cpu`: 系统 CPU 内存 used/total/available/percent
- `gpu`: 每张 GPU 的显存、GPU 利用率、显存控制器利用率、GPU 进程
- `process_tree`: accelerate 根进程及所有子进程的 RSS、VMS、CPU、线程数、命令行

这比推理监控多记录了进程树，因为 SFT 会启动多个 rank，单看系统内存很难判断是哪一个 rank 或 dataloader 在增长。

## 手动绘图

```bash
python /mnt/data/wbw/ktransformers/monitor/DeepSeek-V3.2-SFT/plot_sft_experiment.py

python /mnt/data/wbw/ktransformers/monitor/DeepSeek-V3.2-SFT/plot_sft_experiment.py \
  /mnt/data/wbw/ktransformers/monitor/DeepSeek-V3.2-SFT/YYYYMMDD_HHMMSS
```

## 常用参数

| 参数 | 说明 | 默认值 |
| --- | --- | --- |
| `--monitor-interval` | 资源采样间隔秒数 | `5` |
| `--experiment-dir` | 指定实验输出目录 | 时间戳目录 |
| `--workdir` | 训练命令工作目录 | `/mnt/data/wbw/LLaMA-Factory` |
| `--config-file` | accelerate 配置 | `fsdp2_kt_int8.yaml` |
| `--train-yaml` | LLaMA-Factory 训练配置 | `deepseek_v32_lora_sft_kt.yaml` |
| `--offline` | 设置 HF/Transformers 离线模式 | 关闭 |
| `--dry-run` | 只生成元数据，不启动训练 | 关闭 |

## 前置依赖

建议安装：

```bash
/mnt/data/wbw/miniconda3/envs/Kllama/bin/python -m pip install psutil nvidia-ml-py matplotlib numpy
```

没有 `pynvml` 时会 fallback 到 `nvidia-smi`；没有 `psutil` 时仍会记录系统内存，但不会记录训练进程树。
