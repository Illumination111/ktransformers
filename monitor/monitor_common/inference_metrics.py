"""Metric aggregation for inference monitor outputs."""

from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records = []
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


def load_request_metrics(exp_dir: Path) -> list[dict[str, Any]]:
    records = []
    for log_file in sorted(exp_dir.glob("sglang-request-metrics-*.log")):
        records.extend(load_jsonl(log_file))
    return records


def _mean(values: list[float]) -> float | None:
    clean = [float(v) for v in values if v is not None]
    return round(statistics.mean(clean), 6) if clean else None


def _max(values: list[float]) -> float | None:
    clean = [float(v) for v in values if v is not None]
    return round(max(clean), 6) if clean else None


def _snapshot_memory(snapshot: dict[str, Any] | None) -> dict[str, Any]:
    if not snapshot:
        return {}
    return {
        "cpu_used_gb": snapshot.get("cpu", {}).get("used_gb"),
        "gpu_used_mb": {str(g.get("index", idx)): g.get("used_mb") for idx, g in enumerate(snapshot.get("gpu", []))},
        "gpu_total_mb": {str(g.get("index", idx)): g.get("total_mb") for idx, g in enumerate(snapshot.get("gpu", []))},
    }


def summarize_memory(records: list[dict[str, Any]], startup: dict[str, Any] | None, loaded: dict[str, Any] | None) -> dict[str, Any]:
    peak_gpu: dict[str, Any] = {}
    for record in records:
        for idx, gpu in enumerate(record.get("gpu", [])):
            key = str(gpu.get("index", idx))
            used = gpu.get("used_mb")
            if used is not None:
                peak_gpu[key] = max(peak_gpu.get(key, 0), used)

    return {
        "startup_after_wrapper": _snapshot_memory(startup),
        "model_loaded": _snapshot_memory(loaded),
        "peak_during_inference": {
            "cpu_used_gb": _max([r.get("cpu", {}).get("used_gb") for r in records]),
            "gpu_used_mb": peak_gpu,
        },
    }


def summarize_requests(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        return {"count": 0}
    output_tps = []
    for record in records:
        completion = record.get("completion_tokens") or 0
        latency = record.get("e2e_latency") or 0
        if completion and latency:
            output_tps.append(completion / latency)
    return {
        "count": len(records),
        "prompt_tokens_mean": _mean([r.get("prompt_tokens") for r in records]),
        "completion_tokens_mean": _mean([r.get("completion_tokens") for r in records]),
        "e2e_latency_seconds_mean": _mean([r.get("e2e_latency") for r in records]),
        "output_tokens_per_second_mean": _mean(output_tps),
    }


def summarize_benchmark(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        return {"count": 0}
    return {
        "count": len(records),
        "ttft_seconds_mean": _mean([r.get("ttft_seconds") for r in records]),
        "prefill_latency_seconds_mean": _mean([r.get("prefill_latency_seconds") for r in records]),
        "prefill_tokens_per_second_mean": _mean([r.get("prefill_tokens_per_second") for r in records]),
        "decode_tpot_seconds_mean": _mean([r.get("decode_tpot_seconds") for r in records]),
        "output_tokens_per_second_mean": _mean([r.get("output_tokens_per_second") for r in records]),
        "stable_generation_tokens_per_second_mean": _mean(
            [r.get("stable_generation_tokens_per_second") for r in records]
        ),
        "e2e_latency_seconds_mean": _mean([r.get("e2e_latency_seconds") for r in records]),
        "by_shape": _summarize_benchmark_shapes(records),
    }


def _summarize_benchmark_shapes(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for record in records:
        key = (int(record.get("target_input_tokens", 0)), int(record.get("target_output_tokens", 0)))
        groups.setdefault(key, []).append(record)
    rows = []
    for (input_tokens, output_tokens), group in sorted(groups.items()):
        rows.append(
            {
                "target_input_tokens": input_tokens,
                "target_output_tokens": output_tokens,
                "runs": len(group),
                "ttft_seconds_mean": _mean([r.get("ttft_seconds") for r in group]),
                "prefill_tokens_per_second_mean": _mean([r.get("prefill_tokens_per_second") for r in group]),
                "decode_tpot_seconds_mean": _mean([r.get("decode_tpot_seconds") for r in group]),
                "output_tokens_per_second_mean": _mean([r.get("output_tokens_per_second") for r in group]),
                "e2e_latency_seconds_mean": _mean([r.get("e2e_latency_seconds") for r in group]),
            }
        )
    return rows


def generate_inference_metrics_summary(exp_dir: Path | str) -> dict[str, Any]:
    exp_dir = Path(exp_dir)
    memory_records = load_jsonl(exp_dir / "memory_timeline.jsonl")
    request_records = load_request_metrics(exp_dir)
    benchmark_records = load_jsonl(exp_dir / "inference_benchmark.jsonl")
    summary = {
        "memory": summarize_memory(
            memory_records,
            load_json(exp_dir / "startup_memory.json"),
            load_json(exp_dir / "loaded_memory.json"),
        ),
        "request_metrics": summarize_requests(request_records),
        "benchmark_metrics": summarize_benchmark(benchmark_records),
        "artifacts": {
            "memory_timeline": str(exp_dir / "memory_timeline.jsonl"),
            "request_metrics_logs": [str(p) for p in sorted(exp_dir.glob("sglang-request-metrics-*.log"))],
            "inference_benchmark": str(exp_dir / "inference_benchmark.jsonl")
            if (exp_dir / "inference_benchmark.jsonl").exists()
            else None,
        },
    }
    out_path = exp_dir / "inference_metrics_summary.json"
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary

