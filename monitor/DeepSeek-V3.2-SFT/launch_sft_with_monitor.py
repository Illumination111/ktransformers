"""
Launch LLaMA-Factory KTransformers SFT with resource monitoring.

Default command targets the local DeepSeek-V3.2 AMXINT8 SFT config created in
LLaMA-Factory. Pass a custom command after "--" to monitor any equivalent SFT
launch command.
"""

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR))

from resource_monitor import ResourceMonitor, take_snapshot  # noqa: E402


DEFAULT_WORKDIR = "/mnt/data/wbw/LLaMA-Factory"
DEFAULT_ACCELERATE = "/mnt/data/wbw/miniconda3/envs/Kllama/bin/accelerate"
DEFAULT_CONFIG = "examples/ktransformers/accelerate/fsdp2_kt_int8.yaml"
DEFAULT_TRAIN_YAML = "examples/ktransformers/train_lora/deepseek_v32_lora_sft_kt.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Monitor a DeepSeek-V3.2 KTransformers SFT run",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--monitor-interval", type=float, default=5.0, help="resource sampling interval in seconds")
    parser.add_argument("--experiment-dir", type=str, default=None, help="override experiment output directory")
    parser.add_argument("--workdir", type=str, default=DEFAULT_WORKDIR, help="working directory for the training command")
    parser.add_argument("--accelerate-bin", type=str, default=DEFAULT_ACCELERATE, help="accelerate executable")
    parser.add_argument("--config-file", type=str, default=DEFAULT_CONFIG, help="accelerate config file")
    parser.add_argument("--train-yaml", type=str, default=DEFAULT_TRAIN_YAML, help="LLaMA-Factory training YAML")
    parser.add_argument("--env", action="append", default=[], metavar="KEY=VALUE", help="extra environment variable")
    parser.add_argument("--offline", action="store_true", help="set HF_HUB_OFFLINE=1 and TRANSFORMERS_OFFLINE=1")
    parser.add_argument("--no-tee", action="store_true", help="write train.log without echoing to console")
    parser.add_argument("--dry-run", action="store_true", help="prepare files and print command without launching")
    parser.add_argument("command", nargs=argparse.REMAINDER, help="optional custom command after --")
    return parser.parse_args()


def create_experiment_dir(override_path: str | None) -> Path:
    if override_path:
        exp_dir = Path(override_path)
    else:
        exp_dir = _SCRIPT_DIR / datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir.mkdir(parents=True, exist_ok=True)
    return exp_dir


def build_command(args: argparse.Namespace) -> list[str]:
    if args.command:
        command = args.command
        if command and command[0] == "--":
            command = command[1:]
        if command:
            return command

    return [
        args.accelerate_bin,
        "launch",
        "--config_file",
        args.config_file,
        "-m",
        "llamafactory.cli",
        "train",
        args.train_yaml,
    ]


def build_env(args: argparse.Namespace, exp_dir: Path) -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault("USE_KT", "1")
    env.setdefault("ACCELERATE_USE_KT", "true")
    env.setdefault("PYTHONUNBUFFERED", "1")
    if args.offline:
        env["HF_HUB_OFFLINE"] = "1"
        env["TRANSFORMERS_OFFLINE"] = "1"
    for item in args.env:
        if "=" not in item:
            raise ValueError(f"--env expects KEY=VALUE, got: {item}")
        key, value = item.split("=", 1)
        env[key] = value
    env["SFT_MONITOR_EXPERIMENT_DIR"] = str(exp_dir)
    return env


def _run_text(cmd: list[str], cwd: str | None, env: dict[str, str], timeout: int = 30) -> str:
    try:
        out = subprocess.check_output(cmd, cwd=cwd, env=env, timeout=timeout, stderr=subprocess.STDOUT)
        return out.decode("utf-8", errors="replace").strip()
    except Exception as exc:
        return f"<unavailable: {exc}>"


def collect_metadata(cmd: list[str], args: argparse.Namespace, env: dict[str, str]) -> dict[str, Any]:
    selected_env_keys = [
        "CUDA_VISIBLE_DEVICES",
        "USE_KT",
        "ACCELERATE_USE_KT",
        "ACCELERATE_KT_WEIGHT_PATH",
        "HF_HUB_OFFLINE",
        "TRANSFORMERS_OFFLINE",
        "HF_HOME",
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "NO_PROXY",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
    ]
    metadata = {
        "start_time": datetime.now().isoformat(timespec="seconds"),
        "command": cmd,
        "workdir": args.workdir,
        "python_executable": sys.executable,
        "monitor_interval_secs": args.monitor_interval,
        "env": {key: env.get(key) for key in selected_env_keys if key in env},
        "versions": {
            "python": sys.version.replace("\n", " "),
            "torch": _run_text([sys.executable, "-c", "import torch; print(torch.__version__)"], args.workdir, env),
            "transformers": _run_text([sys.executable, "-c", "import transformers; print(transformers.__version__)"], args.workdir, env),
            "accelerate": _run_text([sys.executable, "-c", "import accelerate; print(accelerate.__version__)"], args.workdir, env),
            "kt_kernel": _run_text([sys.executable, "-c", "import kt_kernel; print(kt_kernel.__version__)"], args.workdir, env),
        },
        "nvidia_smi": _run_text(["nvidia-smi"], args.workdir, env),
        "git": {
            "llama_factory": _run_text(["git", "-C", args.workdir, "rev-parse", "--short", "HEAD"], None, env),
            "ktransformers": _run_text(["git", "-C", "/mnt/data/wbw/ktransformers", "rev-parse", "--short", "HEAD"], None, env),
        },
    }
    return metadata


def save_json(path: Path, data: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def infer_train_yaml(cmd: list[str]) -> str | None:
    for item in reversed(cmd):
        if item.endswith((".yaml", ".yml")):
            return item
    return None


def infer_output_dir(train_yaml: str | None, workdir: str) -> str | None:
    if not train_yaml:
        return None
    path = Path(train_yaml)
    if not path.is_absolute():
        path = Path(workdir) / path
    if not path.exists():
        return None
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            match = re.match(r"^\s*output_dir\s*:\s*(.+?)\s*$", line)
            if match:
                value = match.group(1).split("#", 1)[0].strip().strip("'\"")
                out = Path(value)
                return str(out if out.is_absolute() else Path(workdir) / out)
    except Exception:
        return None
    return None


def collect_training_artifacts(exp_dir: Path, output_dir: str | None) -> dict[str, Any]:
    artifacts: dict[str, Any] = {"output_dir": output_dir, "files": []}
    if not output_dir:
        return artifacts
    out = Path(output_dir)
    if not out.exists():
        return artifacts

    for name in ["trainer_state.json", "training_args.yaml", "trainer_log.jsonl", "all_results.json"]:
        path = out / name
        if path.exists():
            artifacts["files"].append(str(path))
            if name == "trainer_state.json":
                try:
                    data = json.loads(path.read_text(encoding="utf-8"))
                    artifacts["trainer_state"] = {
                        "global_step": data.get("global_step"),
                        "best_metric": data.get("best_metric"),
                        "log_history_tail": data.get("log_history", [])[-20:],
                    }
                    save_json(exp_dir / "trainer_state_tail.json", artifacts["trainer_state"])
                except Exception:
                    pass
    return artifacts


def write_summary(exp_dir: Path, start_ts: float, exit_code: int, output_dir: str | None) -> None:
    elapsed = round(time.monotonic() - start_ts, 1)
    summary = {
        "end_time": datetime.now().isoformat(timespec="seconds"),
        "elapsed_seconds": elapsed,
        "train_exit_code": exit_code,
        "output_dir": output_dir,
        "training_artifacts": collect_training_artifacts(exp_dir, output_dir),
    }
    save_json(exp_dir / "experiment_summary.json", summary)


def main() -> None:
    args = parse_args()
    exp_dir = create_experiment_dir(args.experiment_dir)
    cmd = build_command(args)
    env = build_env(args, exp_dir)
    output_dir = infer_output_dir(infer_train_yaml(cmd), args.workdir)

    print(f"[monitor] experiment dir: {exp_dir}", flush=True)
    print(f"[monitor] workdir: {args.workdir}", flush=True)
    print(f"[monitor] command: {' '.join(cmd)}", flush=True)

    save_json(exp_dir / "sft_args.json", collect_metadata(cmd, args, env))
    save_json(exp_dir / "startup_memory.json", take_snapshot(label="startup"))

    if args.dry_run:
        print("[monitor] dry-run requested; not launching training", flush=True)
        write_summary(exp_dir, time.monotonic(), 0, output_dir)
        return

    timeline_path = exp_dir / "resource_timeline.jsonl"
    monitor = ResourceMonitor(timeline_path, interval_secs=args.monitor_interval)
    monitor.start()

    train_log_path = exp_dir / "train.log"
    start_ts = time.monotonic()
    proc = subprocess.Popen(
        cmd,
        cwd=args.workdir,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    monitor.set_root_pid(proc.pid)
    save_json(exp_dir / "process.json", {"root_pid": proc.pid, "start_time": datetime.now().isoformat(timespec="seconds")})

    def _forward_signal(signum, frame):
        if proc.poll() is None:
            proc.send_signal(signum)

    signal.signal(signal.SIGTERM, _forward_signal)
    signal.signal(signal.SIGINT, _forward_signal)

    exit_code = 0
    try:
        with open(train_log_path, "w", encoding="utf-8") as log_f:
            assert proc.stdout is not None
            for line in proc.stdout:
                log_f.write(line)
                log_f.flush()
                if not args.no_tee:
                    print(line, end="", flush=True)
        exit_code = proc.wait()
    except KeyboardInterrupt:
        if proc.poll() is None:
            proc.send_signal(signal.SIGINT)
        try:
            exit_code = proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
            exit_code = proc.wait()
    finally:
        monitor.stop()
        write_summary(exp_dir, start_ts, exit_code, output_dir)

    try:
        from plot_sft_experiment import generate_plots

        generate_plots(exp_dir)
    except Exception as exc:
        print(f"[monitor] plot generation failed: {exc}", flush=True)

    print(f"[monitor] finished with exit code {exit_code}; data saved in {exp_dir}", flush=True)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
