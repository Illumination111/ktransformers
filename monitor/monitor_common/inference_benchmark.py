"""OpenAI-compatible streaming benchmark for local inference experiments."""

from __future__ import annotations

import argparse
import json
import statistics
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    import requests
except ImportError as exc:  # pragma: no cover - runtime dependency
    requests = None
    _REQUESTS_IMPORT_ERROR = exc
else:
    _REQUESTS_IMPORT_ERROR = None


BASE_PROMPT = (
    "You are benchmarking a local MoE model. Continue with concise, factual text. "
    "This sentence is repeated only to control prompt token length. "
)


def parse_int_list(value: str | None, default: list[int]) -> list[int]:
    if not value:
        return default
    result = []
    for item in value.split(","):
        item = item.strip()
        if item:
            result.append(int(item))
    return result or default


def wait_for_server(
    base_url: str,
    timeout_secs: float = 600.0,
    poll_secs: float = 2.0,
    is_process_alive=None,
) -> dict[str, Any]:
    if requests is None:
        raise RuntimeError(f"requests is unavailable: {_REQUESTS_IMPORT_ERROR}")

    deadline = time.monotonic() + timeout_secs
    last_error = None
    while time.monotonic() < deadline:
        if is_process_alive is not None and not is_process_alive():
            raise RuntimeError("server process exited before the OpenAI API became ready")
        try:
            resp = requests.get(f"{base_url}/v1/models", timeout=10)
            if resp.ok:
                return resp.json()
            last_error = f"HTTP {resp.status_code}: {resp.text[:200]}"
        except Exception as exc:  # noqa: BLE001
            last_error = str(exc)
        time.sleep(poll_secs)
    raise TimeoutError(f"server did not become ready within {timeout_secs}s: {last_error}")


def _load_tokenizer(tokenizer_path: str | None):
    if not tokenizer_path:
        return None
    try:
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True, local_files_only=True)
    except Exception as exc:  # noqa: BLE001
        print(f"[benchmark] tokenizer unavailable; using approximate prompt sizing: {exc}", flush=True)
        return None


def _count_tokens(text: str, tokenizer) -> int:
    if tokenizer is None:
        return max(1, len(text.split()))
    return len(tokenizer.encode(text, add_special_tokens=False))


def make_prompt(target_tokens: int, tokenizer=None) -> tuple[str, int]:
    text = BASE_PROMPT
    while _count_tokens(text, tokenizer) < target_tokens:
        text += BASE_PROMPT
    if tokenizer is not None:
        ids = tokenizer.encode(text, add_special_tokens=False)[:target_tokens]
        try:
            text = tokenizer.decode(ids, skip_special_tokens=True)
        except Exception:
            pass
    return text, _count_tokens(text, tokenizer)


def _percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, round((pct / 100.0) * (len(ordered) - 1))))
    return ordered[idx]


def stream_chat_completion(
    base_url: str,
    model: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
) -> dict[str, Any]:
    if requests is None:
        raise RuntimeError(f"requests is unavailable: {_REQUESTS_IMPORT_ERROR}")

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    url = f"{base_url}/v1/chat/completions"
    start = time.perf_counter()
    first_token_time = None
    previous_chunk_time = None
    inter_chunk_latencies = []
    chunk_count = 0
    full_text_parts: list[str] = []
    usage: dict[str, Any] = {}

    with requests.post(url, json=payload, stream=True, timeout=1800) as resp:
        resp.raise_for_status()
        for raw_line in resp.iter_lines():
            if not raw_line:
                continue
            line = raw_line.decode("utf-8", errors="replace")
            if not line.startswith("data: "):
                continue
            data_str = line[6:].strip()
            if data_str == "[DONE]":
                break
            try:
                data = json.loads(data_str)
            except json.JSONDecodeError:
                continue

            if data.get("usage"):
                usage = data["usage"]

            choice = (data.get("choices") or [{}])[0]
            delta = choice.get("delta") or {}
            content = delta.get("content") or ""
            if not content:
                continue
            now = time.perf_counter()
            if first_token_time is None:
                first_token_time = now
            if previous_chunk_time is not None:
                inter_chunk_latencies.append(now - previous_chunk_time)
            previous_chunk_time = now
            chunk_count += 1
            full_text_parts.append(content)

    end = time.perf_counter()
    ttft = (first_token_time - start) if first_token_time is not None else None
    e2e = end - start
    completion_tokens = usage.get("completion_tokens")
    prompt_tokens = usage.get("prompt_tokens")
    if completion_tokens is None:
        completion_tokens = max(chunk_count, 0)

    decode_seconds = max(e2e - ttft, 0.0) if ttft is not None else None
    decode_token_count = max(completion_tokens - 1, 1) if completion_tokens else None

    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "ttft_seconds": ttft,
        "prefill_latency_seconds": ttft,
        "e2e_latency_seconds": e2e,
        "decode_seconds": decode_seconds,
        "decode_tpot_seconds": (decode_seconds / decode_token_count) if decode_seconds is not None and decode_token_count else None,
        "output_tokens_per_second": (completion_tokens / decode_seconds) if decode_seconds and completion_tokens else None,
        "stable_generation_tokens_per_second": (
            1.0 / statistics.mean(inter_chunk_latencies) if inter_chunk_latencies else None
        ),
        "inter_chunk_latency_avg_seconds": statistics.mean(inter_chunk_latencies) if inter_chunk_latencies else None,
        "inter_chunk_latency_p50_seconds": _percentile(inter_chunk_latencies, 50),
        "inter_chunk_latency_p95_seconds": _percentile(inter_chunk_latencies, 95),
        "chunk_count": chunk_count,
        "text_chars": sum(len(part) for part in full_text_parts),
        "usage": usage,
    }


def run_benchmark_matrix(
    exp_dir: Path,
    base_url: str,
    model: str,
    input_lengths: list[int],
    output_lengths: list[int],
    repetitions: int = 1,
    warmup: int = 1,
    tokenizer_path: str | None = None,
    temperature: float = 0.0,
) -> list[dict[str, Any]]:
    tokenizer = _load_tokenizer(tokenizer_path)
    out_path = exp_dir / "inference_benchmark.jsonl"
    records: list[dict[str, Any]] = []

    if warmup > 0:
        warmup_prompt, _ = make_prompt(min(input_lengths), tokenizer)
        for idx in range(warmup):
            print(f"[benchmark] warmup {idx + 1}/{warmup}", flush=True)
            stream_chat_completion(base_url, model, warmup_prompt, min(output_lengths), temperature)

    with open(out_path, "a", encoding="utf-8") as f:
        for input_len in input_lengths:
            prompt, estimated_prompt_tokens = make_prompt(input_len, tokenizer)
            for output_len in output_lengths:
                for rep in range(repetitions):
                    bench_id = uuid.uuid4().hex
                    print(
                        f"[benchmark] input={input_len} output={output_len} rep={rep + 1}/{repetitions}",
                        flush=True,
                    )
                    result = stream_chat_completion(base_url, model, prompt, output_len, temperature)
                    prompt_tokens = result.get("prompt_tokens") or estimated_prompt_tokens
                    ttft = result.get("ttft_seconds")
                    result.update(
                        {
                            "benchmark_id": bench_id,
                            "timestamp": datetime.now().isoformat(timespec="milliseconds"),
                            "model": model,
                            "target_input_tokens": input_len,
                            "target_output_tokens": output_len,
                            "estimated_prompt_tokens": estimated_prompt_tokens,
                            "prompt_tokens": prompt_tokens,
                            "repetition": rep + 1,
                            "prefill_tokens_per_second": (prompt_tokens / ttft) if ttft else None,
                        }
                    )
                    f.write(json.dumps(result, ensure_ascii=False) + "\n")
                    f.flush()
                    records.append(result)

    print(f"[benchmark] results -> {out_path}", flush=True)
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a local OpenAI-compatible inference benchmark matrix")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--model", required=True)
    parser.add_argument("--experiment-dir", required=True)
    parser.add_argument("--input-lengths", default="128,512,2048")
    parser.add_argument("--output-lengths", default="128,512")
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--tokenizer-path", default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    args = parser.parse_args()

    exp_dir = Path(args.experiment_dir)
    exp_dir.mkdir(parents=True, exist_ok=True)
    base_url = f"http://{args.host}:{args.port}"
    wait_for_server(base_url)
    run_benchmark_matrix(
        exp_dir=exp_dir,
        base_url=base_url,
        model=args.model,
        input_lengths=parse_int_list(args.input_lengths, [128, 512, 2048]),
        output_lengths=parse_int_list(args.output_lengths, [128, 512]),
        repetitions=args.repetitions,
        warmup=args.warmup,
        tokenizer_path=args.tokenizer_path,
        temperature=args.temperature,
    )


if __name__ == "__main__":
    main()
