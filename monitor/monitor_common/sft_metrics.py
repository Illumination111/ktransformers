"""Metric aggregation for LLaMA-Factory / KTransformers SFT monitor outputs."""

from __future__ import annotations

import json
import re
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
    rows = []
    if not path.exists():
        return rows
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


def parse_simple_yaml(path: Path | None) -> dict[str, Any]:
    if not path or not path.exists():
        return {}
    data: dict[str, Any] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line or ":" not in line:
            continue
        key, value = line.split(":", 1)
        value = value.strip().strip("'\"")
        if value.lower() in {"true", "false"}:
            parsed: Any = value.lower() == "true"
        else:
            try:
                parsed = int(value)
            except ValueError:
                try:
                    parsed = float(value)
                except ValueError:
                    parsed = value
        data[key.strip()] = parsed
    return data


def _mean(values: list[float]) -> float | None:
    clean = [float(v) for v in values if v is not None]
    return round(statistics.mean(clean), 6) if clean else None


def _max(values: list[float]) -> float | None:
    clean = [float(v) for v in values if v is not None]
    return round(max(clean), 6) if clean else None


def parse_lora_params(train_log: Path) -> dict[str, Any]:
    if not train_log.exists():
        return {}
    text = train_log.read_text(encoding="utf-8", errors="replace")
    match = re.search(
        r"trainable params:\s*([0-9,]+)\s*\|\|\s*all params:\s*([0-9,]+)\s*\|\|\s*trainable%:\s*([0-9.]+)",
        text,
    )
    if not match:
        return {}
    trainable = int(match.group(1).replace(",", ""))
    total = int(match.group(2).replace(",", ""))
    return {
        "trainable_params": trainable,
        "total_params": total,
        "trainable_ratio_percent": float(match.group(3)),
    }


def parse_phase_timings(train_log: Path) -> dict[str, Any]:
    if not train_log.exists():
        return {"source": "train.log not found", "records": []}
    records = []
    pattern = re.compile(
        r"(?:step[=:]\s*(?P<step>\d+).*?)?"
        r"(?:forward|fwd)[_\s-]*(?:time|latency)?[=:]\s*(?P<forward>[0-9.]+).*?"
        r"(?:backward|bwd)[_\s-]*(?:time|latency)?[=:]\s*(?P<backward>[0-9.]+).*?"
        r"(?:optimizer|optim|update)[_\s-]*(?:time|latency)?[=:]\s*(?P<update>[0-9.]+)",
        re.IGNORECASE,
    )
    for line in train_log.read_text(encoding="utf-8", errors="replace").splitlines():
        match = pattern.search(line)
        if match:
            records.append(
                {
                    "step": int(match.group("step")) if match.group("step") else None,
                    "forward_seconds": float(match.group("forward")),
                    "backward_seconds": float(match.group("backward")),
                    "update_seconds": float(match.group("update")),
                }
            )
    return {
        "source": "train.log phase timing lines" if records else "not found in train.log",
        "records": records[-200:],
        "forward_seconds_mean": _mean([r["forward_seconds"] for r in records]),
        "backward_seconds_mean": _mean([r["backward_seconds"] for r in records]),
        "update_seconds_mean": _mean([r["update_seconds"] for r in records]),
    }


def summarize_resources(records: list[dict[str, Any]], startup: dict[str, Any] | None) -> dict[str, Any]:
    peak_gpu: dict[str, int] = {}
    for rec in records:
        for idx, gpu in enumerate(rec.get("gpu", [])):
            key = str(gpu.get("index", idx))
            used = gpu.get("used_mb")
            if used is not None:
                peak_gpu[key] = max(peak_gpu.get(key, 0), int(used))
    return {
        "startup_cpu_used_gb": (startup or {}).get("cpu", {}).get("used_gb"),
        "startup_gpu_used_mb": {
            str(gpu.get("index", idx)): gpu.get("used_mb")
            for idx, gpu in enumerate((startup or {}).get("gpu", []))
        },
        "peak_system_cpu_used_gb": _max([r.get("cpu", {}).get("used_gb") for r in records]),
        "peak_process_tree_rss_gb": _max([r.get("process_tree", {}).get("total_rss_gb") for r in records]),
        "peak_gpu_used_mb": peak_gpu,
        "sample_count": len(records),
    }


def _training_logs(output_dir: str | None) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    if not output_dir:
        return [], {}, {}
    out = Path(output_dir)
    trainer_log = load_jsonl(out / "trainer_log.jsonl")
    trainer_state = load_json(out / "trainer_state.json") or {}
    all_results = load_json(out / "all_results.json") or {}
    if not trainer_log and trainer_state.get("log_history"):
        trainer_log = [row for row in trainer_state["log_history"] if isinstance(row, dict)]
    return trainer_log, trainer_state, all_results


def summarize_training(output_dir: str | None, train_config: dict[str, Any]) -> dict[str, Any]:
    logs, trainer_state, all_results = _training_logs(output_dir)
    losses = [row.get("loss") for row in logs if row.get("loss") is not None]
    runtime = all_results.get("train_runtime")
    steps = trainer_state.get("global_step") or all_results.get("global_step")
    cutoff_len = train_config.get("cutoff_len")
    samples_per_second = all_results.get("train_samples_per_second")
    tokens_per_second = None
    token_source = None
    if trainer_state.get("num_input_tokens_seen") and runtime:
        tokens_per_second = trainer_state["num_input_tokens_seen"] / runtime
        token_source = "trainer_state.num_input_tokens_seen"
    elif samples_per_second and cutoff_len:
        tokens_per_second = samples_per_second * cutoff_len
        token_source = "estimated_from_samples_per_second_x_cutoff_len"

    return {
        "global_step": steps,
        "train_runtime_seconds": runtime,
        "average_step_time_seconds": (runtime / steps) if runtime and steps else None,
        "train_samples_per_second": samples_per_second,
        "train_steps_per_second": all_results.get("train_steps_per_second"),
        "train_tokens_per_second": tokens_per_second,
        "train_tokens_per_second_source": token_source,
        "train_loss": all_results.get("train_loss"),
        "loss_curve": [
            {"step": row.get("current_steps") or row.get("step"), "epoch": row.get("epoch"), "loss": row.get("loss")}
            for row in logs
            if row.get("loss") is not None
        ],
        "convergence": {
            "first_loss": losses[0] if losses else None,
            "last_loss": losses[-1] if losses else None,
            "best_loss": min(losses) if losses else None,
            "loss_delta": (losses[-1] - losses[0]) if len(losses) >= 2 else None,
        },
    }


def load_performance_delta(before_path: str | None, after_path: str | None, fallback_after: dict[str, Any]) -> dict[str, Any]:
    before = load_json(Path(before_path)) if before_path else None
    after = load_json(Path(after_path)) if after_path else None
    if after is None:
        after = fallback_after or None
    if not before or not after:
        return {
            "available": False,
            "before_metrics_path": before_path,
            "after_metrics_path": after_path,
            "note": "Provide --pretrain-metrics-json and --finetuned-metrics-json to compute before/after task deltas.",
        }
    deltas = {}
    for key, before_val in before.items():
        after_val = after.get(key)
        if isinstance(before_val, (int, float)) and isinstance(after_val, (int, float)):
            deltas[key] = {
                "before": before_val,
                "after": after_val,
                "delta": after_val - before_val,
                "relative_delta_percent": ((after_val - before_val) / before_val * 100.0) if before_val else None,
            }
    return {
        "available": bool(deltas),
        "before_metrics_path": before_path,
        "after_metrics_path": after_path,
        "deltas": deltas,
    }


def generate_sft_metrics_summary(
    exp_dir: Path | str,
    output_dir: str | None,
    train_yaml: str | None,
    workdir: str | None,
    before_metrics_path: str | None = None,
    after_metrics_path: str | None = None,
) -> dict[str, Any]:
    exp_dir = Path(exp_dir)
    train_yaml_path = Path(train_yaml) if train_yaml else None
    if train_yaml_path and not train_yaml_path.is_absolute() and workdir:
        train_yaml_path = Path(workdir) / train_yaml_path
    train_config = parse_simple_yaml(train_yaml_path)
    _, _, all_results = _training_logs(output_dir)

    summary = {
        "resources": summarize_resources(
            load_jsonl(exp_dir / "resource_timeline.jsonl"),
            load_json(exp_dir / "startup_memory.json"),
        ),
        "lora_parameters": parse_lora_params(exp_dir / "train.log"),
        "training": summarize_training(output_dir, train_config),
        "phase_timings": parse_phase_timings(exp_dir / "train.log"),
        "task_performance_delta": load_performance_delta(before_metrics_path, after_metrics_path, all_results),
        "train_config": train_config,
    }
    out_path = exp_dir / "sft_metrics_summary.json"
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary

