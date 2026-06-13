"""Generate SFT monitoring plots from resource_timeline.jsonl."""

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
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


def load_jsonl(path: Path) -> list[dict]:
    records = []
    if not path.exists():
        return records
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return records


def latest_experiment_dir(base_dir: Path) -> Path | None:
    dirs = [p for p in base_dir.iterdir() if p.is_dir() and (p / "resource_timeline.jsonl").exists()]
    if not dirs:
        return None
    return sorted(dirs)[-1]


def plot_resources(records: list[dict], exp_dir: Path, out_dir: Path) -> None:
    if not records:
        print("[plot] no resource records")
        return

    elapsed = [r.get("elapsed_seconds", 0) for r in records]
    gpu_count = 0
    for record in records:
        if record.get("gpu"):
            gpu_count = len(record["gpu"])
            break

    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
    fig.suptitle(f"SFT Resource Timeline\nExperiment: {exp_dir.name}", fontsize=13, fontweight="bold")

    colors = plt.cm.tab10(np.linspace(0, 1, max(gpu_count, 1)))

    ax = axes[0]
    ax.set_title("GPU VRAM")
    ax.set_ylabel("Used MB")
    if gpu_count:
        for idx in range(gpu_count):
            used = []
            total = None
            for record in records:
                gpus = record.get("gpu", [])
                if idx < len(gpus):
                    used.append(gpus[idx].get("used_mb", 0))
                    total = total or gpus[idx].get("total_mb")
                else:
                    used.append(0)
            ax.plot(elapsed, used, color=colors[idx], lw=1.8, label=f"GPU {idx}")
        if total:
            ax.axhline(total, color="red", lw=1, ls="--", alpha=0.5, label=f"Total {total} MB")
        ax.legend(ncol=2, fontsize=8)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    else:
        ax.text(0.5, 0.5, "No GPU data", transform=ax.transAxes, ha="center", va="center", color="gray")

    ax = axes[1]
    ax.set_title("CPU Memory")
    ax.set_ylabel("Used GB")
    cpu_used = [r.get("cpu", {}).get("used_gb", 0) for r in records]
    proc_rss = [r.get("process_tree", {}).get("total_rss_gb", 0) for r in records]
    ax.fill_between(elapsed, cpu_used, alpha=0.2, color="steelblue")
    ax.plot(elapsed, cpu_used, color="steelblue", lw=2, label="System used")
    ax.plot(elapsed, proc_rss, color="darkorange", lw=2, label="Training process tree RSS")
    ax.legend(fontsize=9)

    ax = axes[2]
    ax.set_title("CPU Utilization / Process Count")
    ax.set_xlabel("Elapsed seconds")
    ax.set_ylabel("CPU percent")
    proc_cpu = [r.get("process_tree", {}).get("total_cpu_percent", 0) for r in records]
    proc_count = [r.get("process_tree", {}).get("process_count", 0) for r in records]
    ax.plot(elapsed, proc_cpu, color="purple", lw=1.8, label="Process tree CPU %")
    ax2 = ax.twinx()
    ax2.set_ylabel("Process count")
    ax2.plot(elapsed, proc_count, color="gray", lw=1.2, ls="--", label="Process count")
    ax.legend(loc="upper left", fontsize=9)
    ax2.legend(loc="upper right", fontsize=9)

    plt.tight_layout()
    out_path = out_dir / "sft_resource_timeline.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] resource timeline -> {out_path}")


def generate_plots(exp_dir: Path | str) -> None:
    exp_dir = Path(exp_dir)
    out_dir = exp_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    records = load_jsonl(exp_dir / "resource_timeline.jsonl")
    plot_resources(records, exp_dir, out_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot SFT monitoring results")
    parser.add_argument("experiment_dir", nargs="?", default=None)
    args = parser.parse_args()

    if args.experiment_dir:
        exp_dir = Path(args.experiment_dir)
    else:
        exp_dir = latest_experiment_dir(Path(__file__).resolve().parent)
        if exp_dir is None:
            raise SystemExit("No experiment directory with resource_timeline.jsonl found")
    generate_plots(exp_dir)


if __name__ == "__main__":
    main()
