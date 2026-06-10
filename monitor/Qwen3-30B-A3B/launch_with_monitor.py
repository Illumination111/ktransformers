"""
带实验监控的 sglang 启动包装脚本。

用法（等价于直接调用 python -m sglang.launch_server，但会额外记录实验数据）：
    python launch_with_monitor.py \\
        --model /path/to/Qwen3-30B-A3B \\
        --kt-weight-path /path/to/weights \\
        --kt-method AMXINT8 \\
        --kt-cpuinfer 64 \\
        --kt-threadpool-count 2 \\
        --kt-num-gpu-experts 32 \\
        [其他 sglang 参数...]

每次执行会在脚本所在目录下创建以 YYYYMMDD_HHMMSS 命名的实验子目录，包含：
    server_args.json          启动参数快照
    startup_memory.json       启动时 GPU 显存 + CPU 内存状态
    memory_timeline.jsonl     推理过程中的周期性内存采样（每 5 秒一条）
    sglang-request-metrics-*  SGLang 原生请求指标（含 prompt/completion tokens 等）

可选参数（不传给 sglang）：
    --monitor-interval        内存采样间隔秒数，默认 5
    --experiment-dir          覆盖默认实验目录路径（默认在脚本同目录下按时间自动命名）
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# 将脚本所在目录加入 sys.path，方便导入 memory_monitor
_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR))

from memory_monitor import MemoryMonitor, take_memory_snapshot  # noqa: E402


# ---------------------------------------------------------------------------
# 参数解析
# ---------------------------------------------------------------------------

def parse_args():
    """
    解析监控专属参数，所有未知参数均透传给 sglang.launch_server。
    """
    parser = argparse.ArgumentParser(
        description="带实验监控的 sglang MoE 异构推理启动脚本",
        add_help=True,
        # 允许未知参数（sglang 的大量参数）
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--monitor-interval",
        type=float,
        default=5.0,
        metavar="SECONDS",
        help="内存采样间隔（秒），默认 5",
    )
    parser.add_argument(
        "--experiment-dir",
        type=str,
        default=None,
        metavar="PATH",
        help="覆盖默认实验目录路径；默认在脚本同目录下以 YYYYMMDD_HHMMSS 自动命名",
    )
    # parse_known_args：未识别的参数存入 sglang_argv
    monitor_args, sglang_argv = parser.parse_known_args()
    return monitor_args, sglang_argv


# ---------------------------------------------------------------------------
# 实验目录创建
# ---------------------------------------------------------------------------

def create_experiment_dir(base_dir: Path, override_path: str = None) -> Path:
    """创建并返回实验子目录。"""
    if override_path:
        exp_dir = Path(override_path)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_dir = base_dir / timestamp
    exp_dir.mkdir(parents=True, exist_ok=True)
    return exp_dir


# ---------------------------------------------------------------------------
# sglang 命令构建
# ---------------------------------------------------------------------------

def build_sglang_cmd(sglang_argv: list, exp_dir: Path) -> list:
    """
    构造最终的 sglang 启动命令，注入 --export-metrics-to-file 相关参数。
    若调用方已提供这些参数，则不重复添加（以调用方为准）。
    """
    cmd = [sys.executable, "-m", "sglang.launch_server"] + sglang_argv

    # 仅在未手动指定时注入
    if "--export-metrics-to-file" not in sglang_argv:
        cmd += ["--export-metrics-to-file"]
    if "--export-metrics-to-file-dir" not in sglang_argv:
        cmd += ["--export-metrics-to-file-dir", str(exp_dir)]

    return cmd


# ---------------------------------------------------------------------------
# 实验元数据保存
# ---------------------------------------------------------------------------

def save_server_args(exp_dir: Path, sglang_argv: list, monitor_args):
    """将启动参数保存为 server_args.json。"""
    record = {
        "start_time": datetime.now().isoformat(timespec="seconds"),
        "sglang_argv": sglang_argv,
        "monitor_interval_secs": monitor_args.monitor_interval,
        "python_executable": sys.executable,
        "working_directory": os.getcwd(),
    }
    out_path = exp_dir / "server_args.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)
    print(f"[monitor] 启动参数已保存 → {out_path}", flush=True)


def save_startup_memory(exp_dir: Path):
    """采集并保存启动时的内存快照为 startup_memory.json。"""
    snapshot = take_memory_snapshot(label="startup")
    out_path = exp_dir / "startup_memory.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(snapshot, f, ensure_ascii=False, indent=2)
    print(f"[monitor] 启动内存快照已保存 → {out_path}", flush=True)
    return snapshot


def save_experiment_summary(exp_dir: Path, start_ts: float, exit_code: int):
    """实验结束后写入摘要文件 experiment_summary.json。"""
    elapsed = round(time.monotonic() - start_ts, 1)
    summary = {
        "end_time": datetime.now().isoformat(timespec="seconds"),
        "elapsed_seconds": elapsed,
        "sglang_exit_code": exit_code,
    }
    # 统计请求数（统计 sglang 日志文件行数）
    total_requests = 0
    for log_file in exp_dir.glob("sglang-request-metrics-*.log"):
        try:
            with open(log_file, encoding="utf-8") as f:
                total_requests += sum(1 for _ in f)
        except Exception:
            pass
    summary["total_logged_requests"] = total_requests

    out_path = exp_dir / "experiment_summary.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"[monitor] 实验摘要已保存 → {out_path}", flush=True)


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def main():
    monitor_args, sglang_argv = parse_args()

    # 1. 创建实验目录
    exp_dir = create_experiment_dir(_SCRIPT_DIR, monitor_args.experiment_dir)
    print(f"[monitor] 实验目录: {exp_dir}", flush=True)

    # 2. 保存启动参数
    save_server_args(exp_dir, sglang_argv, monitor_args)

    # 3. 采集启动时内存快照
    startup_snap = save_startup_memory(exp_dir)
    gpu_info = startup_snap.get("gpu", [])
    cpu_info = startup_snap.get("cpu", {})
    gpu_summary = [
        f"{g['name']} {g['used_mb']}/{g['total_mb']} MB ({g['percent']}%)"
        for g in gpu_info
    ]
    print(f"[monitor] 启动时显存: {gpu_summary}", flush=True)
    print(
        f"[monitor] 启动时CPU内存: {cpu_info.get('used_gb', '?')} GB / {cpu_info.get('total_gb', '?')} GB ({cpu_info.get('percent', '?')}%)",
        flush=True,
    )

    # 4. 构造 sglang 命令
    cmd = build_sglang_cmd(sglang_argv, exp_dir)
    print(f"[monitor] 启动命令: {' '.join(cmd)}", flush=True)

    # 5. 启动后台内存监控线程
    timeline_path = exp_dir / "memory_timeline.jsonl"
    monitor = MemoryMonitor(str(timeline_path), interval_secs=monitor_args.monitor_interval)
    monitor.start()
    print(
        f"[monitor] 内存监控已启动（间隔 {monitor_args.monitor_interval}s）→ {timeline_path}",
        flush=True,
    )

    # 6. 启动 sglang 子进程
    start_ts = time.monotonic()
    proc = subprocess.Popen(cmd)

    # 7. 信号转发：将 SIGTERM/SIGINT 传递给 sglang 子进程
    def _forward_signal(signum, frame):
        if proc.poll() is None:
            proc.send_signal(signum)

    signal.signal(signal.SIGTERM, _forward_signal)
    signal.signal(signal.SIGINT, _forward_signal)

    # 8. 等待 sglang 退出
    exit_code = 0
    try:
        exit_code = proc.wait()
    except KeyboardInterrupt:
        if proc.poll() is None:
            proc.send_signal(signal.SIGINT)
        try:
            exit_code = proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
            exit_code = proc.wait()

    # 9. 停止内存监控
    print("[monitor] sglang 进程已退出，停止内存监控...", flush=True)
    monitor.stop()

    # 10. 写实验摘要
    save_experiment_summary(exp_dir, start_ts, exit_code)
    print(f"[monitor] 实验结束，所有数据已保存至: {exp_dir}", flush=True)

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
