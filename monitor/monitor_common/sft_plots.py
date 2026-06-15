"""Additional plots for SFT monitor summaries."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def plot_sft_training_summary(exp_dir: Path | str, out_dir: Path | str) -> None:
    exp_dir = Path(exp_dir)
    out_dir = Path(out_dir)
    summary = _load_json(exp_dir / "sft_metrics_summary.json")
    if not summary:
        print("[plot] no sft_metrics_summary.json, skipping SFT summary plots")
        return

    training = summary.get("training", {})
    loss_curve = training.get("loss_curve", [])
    resources = summary.get("resources", {})
    lora = summary.get("lora_parameters", {})

    if loss_curve:
        fig, ax = plt.subplots(figsize=(10, 5))
        xs = [row.get("step") for row in loss_curve]
        ys = [row.get("loss") for row in loss_curve]
        ax.plot(xs, ys, marker="o", lw=2, color="#4C72B0")
        ax.set_title("Training Loss / Convergence")
        ax.set_xlabel("Step")
        ax.set_ylabel("Loss")
        for x, y in zip(xs, ys):
            if x is not None and y is not None:
                ax.text(x, y, f"{y:.3f}", fontsize=8, ha="center", va="bottom")
        out_path = out_dir / "sft_training_loss.png"
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)
        print(f"[plot] SFT training loss -> {out_path}")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("SFT Summary Metrics", fontsize=13, fontweight="bold")

    ax = axes[0]
    labels = ["CPU peak GB", "Process RSS GB"]
    values = [resources.get("peak_system_cpu_used_gb") or 0, resources.get("peak_process_tree_rss_gb") or 0]
    ax.bar(labels, values, color=["#55A868", "#DD8452"], alpha=0.85)
    ax.set_title("Host Memory")
    ax.set_ylabel("GB")
    for idx, value in enumerate(values):
        ax.text(idx, value, f"{value:.2f}", ha="center", va="bottom", fontsize=9)

    ax = axes[1]
    gpu_items = sorted((resources.get("peak_gpu_used_mb") or {}).items())
    if gpu_items:
        labels = [f"GPU {key}" for key, _ in gpu_items]
        values = [value for _, value in gpu_items]
        ax.bar(labels, values, color="#8172B2", alpha=0.85)
        ax.set_ylabel("MB")
        for idx, value in enumerate(values):
            ax.text(idx, value, f"{value:,}", ha="center", va="bottom", fontsize=8)
    else:
        ax.text(0.5, 0.5, "No GPU data", transform=ax.transAxes, ha="center", va="center", color="gray")
    ax.set_title("Peak GPU VRAM")

    caption = []
    if lora:
        caption.append(
            f"LoRA trainable: {lora.get('trainable_params'):,} / {lora.get('total_params'):,} "
            f"({lora.get('trainable_ratio_percent')}%)"
        )
    if training.get("average_step_time_seconds"):
        caption.append(f"Avg step: {training['average_step_time_seconds']:.2f}s")
    if training.get("train_tokens_per_second"):
        caption.append(f"Tokens/s: {training['train_tokens_per_second']:.2f}")
    if caption:
        fig.text(0.5, 0.01, " | ".join(caption), ha="center", fontsize=9)

    out_path = out_dir / "sft_summary_metrics.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] SFT summary metrics -> {out_path}")

    delta = summary.get("task_performance_delta", {})
    if delta.get("available") and delta.get("deltas"):
        keys = list(delta["deltas"].keys())
        before = [delta["deltas"][k]["before"] for k in keys]
        after = [delta["deltas"][k]["after"] for k in keys]
        x = range(len(keys))
        fig, ax = plt.subplots(figsize=(max(8, len(keys) * 1.2), 5))
        ax.bar([i - 0.2 for i in x], before, width=0.4, label="Before", color="#4C72B0")
        ax.bar([i + 0.2 for i in x], after, width=0.4, label="After", color="#C44E52")
        ax.set_xticks(list(x))
        ax.set_xticklabels(keys, rotation=30, ha="right")
        ax.set_title("Task Performance Before vs After SFT")
        ax.legend()
        out_path = out_dir / "sft_task_performance_delta.png"
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)
        print(f"[plot] SFT task performance delta -> {out_path}")

