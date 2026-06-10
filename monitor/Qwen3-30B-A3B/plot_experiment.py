"""
实验数据可视化脚本。

读取实验目录中的以下文件并生成图表：
  - memory_timeline.jsonl  → GPU 显存 + CPU 内存随时间变化图
  - sglang-request-metrics-*.log → 每次请求的 token 统计和延迟图

用法：
    # 指定实验目录
    python plot_experiment.py /path/to/20260610_165000

    # 自动找最新实验目录（在脚本同目录下查找）
    python plot_experiment.py

输出：
    <experiment_dir>/plots/memory_timeline.png
    <experiment_dir>/plots/request_metrics.png
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # 无显示器环境
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

plt.rcParams.update(
    {
        "font.family": ["DejaVu Sans", "sans-serif"],
        "axes.grid": True,
        "grid.alpha": 0.3,
        "figure.dpi": 150,
    }
)

# ---------------------------------------------------------------------------
# 数据加载
# ---------------------------------------------------------------------------

def load_memory_timeline(exp_dir: Path):
    """加载 memory_timeline.jsonl，返回结构化数据。"""
    path = exp_dir / "memory_timeline.jsonl"
    if not path.exists():
        return None
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return records


def load_request_metrics(exp_dir: Path):
    """加载所有 sglang-request-metrics-*.log，返回请求记录列表。"""
    records = []
    for log_file in sorted(exp_dir.glob("sglang-request-metrics-*.log")):
        with open(log_file, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
    return records


def load_startup_memory(exp_dir: Path):
    """加载 startup_memory.json。"""
    path = exp_dir / "startup_memory.json"
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_server_args(exp_dir: Path):
    """加载 server_args.json 中的启动参数摘要。"""
    path = exp_dir / "server_args.json"
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# 图表 1：内存时间线
# ---------------------------------------------------------------------------

def plot_memory_timeline(records, startup, exp_dir: Path, out_dir: Path):
    if not records:
        print("[plot] memory_timeline.jsonl has no data, skipping memory plot")
        return

    elapsed = [r["elapsed_seconds"] for r in records]

    # GPU 数量（从第一条有 GPU 数据的记录中取）
    gpu_count = 0
    for r in records:
        if r.get("gpu"):
            gpu_count = len(r["gpu"])
            break

    fig, axes = plt.subplots(
        2, 1, figsize=(14, 9), sharex=True,
        gridspec_kw={"height_ratios": [2, 1]}
    )
    fig.suptitle(
        f"Memory Usage Timeline\nExperiment: {exp_dir.name}",
        fontsize=13, fontweight="bold"
    )

    # ── Sub-plot 1: GPU VRAM ──────────────────────────────────────────────
    ax_gpu = axes[0]
    ax_gpu.set_ylabel("GPU Memory Used (MB)", fontsize=11)
    ax_gpu.set_title("GPU VRAM", fontsize=11)

    colors = plt.cm.tab10(np.linspace(0, 1, max(gpu_count, 1)))

    if gpu_count > 0:
        gpu_total_mb = None
        for gi in range(gpu_count):
            used = []
            for r in records:
                gpus = r.get("gpu", [])
                val = gpus[gi]["used_mb"] if gi < len(gpus) else 0
                used.append(val)
                if gpu_total_mb is None and gi < len(gpus):
                    gpu_total_mb = gpus[gi]["total_mb"]
            lw = 2.0 if gi == 0 else 1.0
            alpha = 1.0 if gi == 0 else 0.6
            label = f"GPU {gi}" + (" (inference GPU)" if gi == 0 else "")
            ax_gpu.plot(elapsed, used, color=colors[gi], lw=lw,
                        alpha=alpha, label=label)

        if gpu_total_mb:
            ax_gpu.axhline(gpu_total_mb, color="red", lw=1, ls="--",
                           alpha=0.5, label=f"Total ({gpu_total_mb} MB)")
            ax_gpu.set_ylim(0, gpu_total_mb * 1.05)

        ax_gpu.legend(loc="upper left", fontsize=8, ncol=2)
        ax_gpu.yaxis.set_major_formatter(
            mticker.FuncFormatter(lambda x, _: f"{int(x):,}")
        )
    else:
        ax_gpu.text(0.5, 0.5, "No GPU data", transform=ax_gpu.transAxes,
                    ha="center", va="center", color="gray")

    # ── Sub-plot 2: CPU Memory ────────────────────────────────────────────
    ax_cpu = axes[1]
    ax_cpu.set_xlabel("Elapsed Time (s)", fontsize=11)
    ax_cpu.set_ylabel("CPU Memory Used (GB)", fontsize=11)
    ax_cpu.set_title("System Memory (CPU)", fontsize=11)

    cpu_used = [r["cpu"].get("used_gb", 0) for r in records]
    cpu_total = records[0]["cpu"].get("total_gb", None) if records else None

    ax_cpu.fill_between(elapsed, cpu_used, alpha=0.25, color="steelblue")
    ax_cpu.plot(elapsed, cpu_used, color="steelblue", lw=2, label="Used")

    if cpu_total:
        ax_cpu.axhline(cpu_total, color="red", lw=1, ls="--",
                       alpha=0.5, label=f"Total ({cpu_total:.0f} GB)")
        ax_cpu.set_ylim(0, cpu_total * 1.05)

    if startup and startup.get("cpu", {}).get("used_gb"):
        ax_cpu.axhline(
            startup["cpu"]["used_gb"], color="gray", lw=1, ls=":",
            alpha=0.8, label=f"Baseline ({startup['cpu']['used_gb']} GB)"
        )

    ax_cpu.legend(loc="upper left", fontsize=9)
    ax_cpu.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"{x:.1f}")
    )

    plt.tight_layout()
    out_path = out_dir / "memory_timeline.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] 内存时间线图表 → {out_path}")


# ---------------------------------------------------------------------------
# 图表 2：请求指标
# ---------------------------------------------------------------------------

def plot_request_metrics(records, exp_dir: Path, out_dir: Path):
    if not records:
        print("[plot] No request data, skipping request metrics plot")
        return

    n = len(records)
    req_ids = [f"#{i+1}" for i in range(n)]
    prompt_tokens = [r.get("prompt_tokens", 0) for r in records]
    completion_tokens = [r.get("completion_tokens", 0) for r in records]
    cached_tokens = [r.get("cached_tokens", 0) for r in records]
    e2e_latency = [r.get("e2e_latency", 0) for r in records]
    throughput = [
        c / l if l > 0 else 0
        for c, l in zip(completion_tokens, e2e_latency)
    ]

    fig, axes = plt.subplots(3, 1, figsize=(max(10, n * 1.5 + 2), 12))
    fig.suptitle(
        f"Request Metrics\nExperiment: {exp_dir.name}  ({n} requests)",
        fontsize=13, fontweight="bold"
    )

    x = np.arange(n)
    bar_w = 0.28

    # ── Sub-plot 1: Token counts ──────────────────────────────────────────
    ax1 = axes[0]
    ax1.set_title("Token Count per Request", fontsize=11)

    new_tokens = [max(p - c, 0) for p, c in zip(prompt_tokens, cached_tokens)]

    b1 = ax1.bar(x - bar_w, prompt_tokens, bar_w, label="Input tokens (total)",
                 color="#4C72B0", alpha=0.85)
    b2 = ax1.bar(x, new_tokens, bar_w, label="Input tokens (new)",
                 color="#55A868", alpha=0.85)
    b3 = ax1.bar(x + bar_w, completion_tokens, bar_w, label="Output tokens",
                 color="#C44E52", alpha=0.85)

    for bar in [b1, b2, b3]:
        for rect in bar:
            h = rect.get_height()
            if h > 0:
                ax1.text(
                    rect.get_x() + rect.get_width() / 2, h + 5,
                    f"{int(h)}", ha="center", va="bottom", fontsize=7
                )

    ax1.set_xticks(x)
    ax1.set_xticklabels(req_ids)
    ax1.set_ylabel("Token count", fontsize=10)
    ax1.legend(fontsize=9)
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{int(v):,}"))

    # ── Sub-plot 2: E2E Latency ───────────────────────────────────────────
    ax2 = axes[1]
    ax2.set_title("End-to-End Latency", fontsize=11)

    bars = ax2.bar(x, e2e_latency, color="#DD8452", alpha=0.85, width=0.5)
    for rect in bars:
        h = rect.get_height()
        if h > 0:
            ax2.text(
                rect.get_x() + rect.get_width() / 2, h + 0.3,
                f"{h:.1f}s", ha="center", va="bottom", fontsize=8
            )

    ax2.set_xticks(x)
    ax2.set_xticklabels(req_ids)
    ax2.set_ylabel("Latency (s)", fontsize=10)

    mean_lat = np.mean(e2e_latency)
    ax2.axhline(mean_lat, color="red", lw=1.5, ls="--",
                label=f"Mean {mean_lat:.1f}s")
    ax2.legend(fontsize=9)

    # ── Sub-plot 3: Output Throughput ─────────────────────────────────────
    ax3 = axes[2]
    ax3.set_title("Output Throughput (tokens/s)", fontsize=11)

    bars = ax3.bar(x, throughput, color="#8172B2", alpha=0.85, width=0.5)
    for rect in bars:
        h = rect.get_height()
        if h > 0:
            ax3.text(
                rect.get_x() + rect.get_width() / 2, h + 0.2,
                f"{h:.1f}", ha="center", va="bottom", fontsize=8
            )

    ax3.set_xticks(x)
    ax3.set_xticklabels(req_ids)
    ax3.set_xlabel("Request #", fontsize=10)
    ax3.set_ylabel("tokens/s", fontsize=10)

    mean_tps = np.mean([t for t in throughput if t > 0])
    ax3.axhline(mean_tps, color="red", lw=1.5, ls="--",
                label=f"Mean {mean_tps:.1f} tokens/s")
    ax3.legend(fontsize=9)

    plt.tight_layout()
    out_path = out_dir / "request_metrics.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] 请求指标图表 → {out_path}")


# ---------------------------------------------------------------------------
# 主函数
# ---------------------------------------------------------------------------

def find_latest_exp_dir(base_dir: Path) -> Path:
    """在 base_dir 下找最新的实验目录（格式 YYYYMMDD_HHMMSS）。"""
    candidates = sorted(
        [d for d in base_dir.iterdir() if d.is_dir() and len(d.name) == 15
         and d.name[8] == "_"],
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(f"在 {base_dir} 下找不到实验目录")
    return candidates[0]


def generate_plots(exp_dir: Path):
    """对指定实验目录生成全部图表，返回输出目录 Path。"""
    out_dir = exp_dir / "plots"
    out_dir.mkdir(exist_ok=True)

    startup = load_startup_memory(exp_dir)
    mem_records = load_memory_timeline(exp_dir)
    req_records = load_request_metrics(exp_dir)

    print(f"[plot] 实验目录: {exp_dir}")
    print(f"[plot] 内存采样点: {len(mem_records) if mem_records else 0}")
    print(f"[plot] 请求记录数: {len(req_records)}")

    plot_memory_timeline(mem_records, startup, exp_dir, out_dir)
    plot_request_metrics(req_records, exp_dir, out_dir)

    print(f"[plot] 所有图表已保存至: {out_dir}")
    return out_dir


def main():
    parser = argparse.ArgumentParser(description="实验数据可视化")
    parser.add_argument(
        "exp_dir",
        nargs="?",
        default=None,
        help="实验目录路径（不填则自动找脚本同目录下最新实验）",
    )
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent

    if args.exp_dir:
        exp_dir = Path(args.exp_dir).resolve()
    else:
        exp_dir = find_latest_exp_dir(script_dir)
        print(f"[plot] 自动选择最新实验: {exp_dir}")

    if not exp_dir.exists():
        print(f"[错误] 目录不存在: {exp_dir}")
        sys.exit(1)

    generate_plots(exp_dir)


if __name__ == "__main__":
    main()
