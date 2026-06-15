"""Additional inference plots built from monitor_common inference summaries."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return rows


def _mean(values: list[float]) -> float | None:
    clean = [float(v) for v in values if v is not None]
    return sum(clean) / len(clean) if clean else None


def _group_by_shape(records: list[dict]) -> list[dict]:
    groups = {}
    for rec in records:
        key = (rec.get("target_input_tokens"), rec.get("target_output_tokens"))
        groups.setdefault(key, []).append(rec)
    rows = []
    for (inp, out), group in sorted(groups.items()):
        rows.append(
            {
                "input": inp,
                "output": out,
                "ttft": _mean([r.get("ttft_seconds") for r in group]),
                "prefill_tps": _mean([r.get("prefill_tokens_per_second") for r in group]),
                "tpot_ms": (_mean([r.get("decode_tpot_seconds") for r in group]) or 0) * 1000,
                "output_tps": _mean([r.get("output_tokens_per_second") for r in group]),
                "e2e": _mean([r.get("e2e_latency_seconds") for r in group]),
            }
        )
    return rows


def plot_benchmark_metrics(exp_dir: Path | str, out_dir: Path | str) -> None:
    exp_dir = Path(exp_dir)
    out_dir = Path(out_dir)
    records = _load_jsonl(exp_dir / "inference_benchmark.jsonl")
    if not records:
        print("[plot] no inference_benchmark.jsonl, skipping benchmark plot")
        return

    rows = _group_by_shape(records)
    input_lengths = sorted({r["input"] for r in rows})
    output_lengths = sorted({r["output"] for r in rows})

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Inference Benchmark Metrics\nExperiment: {exp_dir.name}", fontsize=13, fontweight="bold")

    metric_specs = [
        ("ttft", "TTFT / Prefill Latency", "seconds"),
        ("prefill_tps", "Input Token Throughput", "tokens/s"),
        ("tpot_ms", "Decode TPOT / ITL", "ms/token"),
        ("output_tps", "Output Token Throughput", "tokens/s"),
    ]
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(output_lengths), 1)))

    for ax, (metric, title, ylabel) in zip(axes.flat, metric_specs):
        ax.set_title(title)
        ax.set_xlabel("Input tokens")
        ax.set_ylabel(ylabel)
        for idx, out_len in enumerate(output_lengths):
            series = [r for r in rows if r["output"] == out_len]
            xs = [r["input"] for r in series]
            ys = [r.get(metric) for r in series]
            ax.plot(xs, ys, marker="o", lw=2, color=colors[idx], label=f"out={out_len}")
            for x, y in zip(xs, ys):
                if y is not None:
                    ax.text(x, y, f"{y:.2f}", fontsize=8, ha="center", va="bottom")
        ax.legend(fontsize=8)

    plt.tight_layout()
    out_path = out_dir / "inference_benchmark_metrics.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] inference benchmark metrics -> {out_path}")

