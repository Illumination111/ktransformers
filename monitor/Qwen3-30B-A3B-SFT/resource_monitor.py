"""
Resource monitor for LLaMA-Factory / KTransformers SFT runs.

Writes JSONL samples with:
  - system CPU memory
  - per-GPU memory/utilization
  - GPU compute processes
  - the launched training process tree RSS/CPU/thread counts
"""

import json
import os
import subprocess
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    import psutil

    _PSUTIL_AVAILABLE = True
except ImportError:
    _PSUTIL_AVAILABLE = False

try:
    import pynvml

    _PYNVML_AVAILABLE = True
except ImportError:
    _PYNVML_AVAILABLE = False


def _gb(value: int | float) -> float:
    return round(float(value) / (1024**3), 3)


def _mb(value: int | float) -> int:
    return int(float(value) / (1024**2))


def sample_cpu_memory() -> dict[str, Any]:
    if _PSUTIL_AVAILABLE:
        vm = psutil.virtual_memory()
        return {
            "used_gb": _gb(vm.used),
            "total_gb": _gb(vm.total),
            "available_gb": _gb(vm.available),
            "percent": vm.percent,
        }

    try:
        meminfo: dict[str, int] = {}
        with open("/proc/meminfo", encoding="utf-8") as f:
            for line in f:
                key, val = line.split(":", 1)
                meminfo[key.strip()] = int(val.strip().split()[0]) * 1024
        total = meminfo.get("MemTotal", 0)
        available = meminfo.get("MemAvailable", 0)
        used = total - available
        return {
            "used_gb": _gb(used),
            "total_gb": _gb(total),
            "available_gb": _gb(available),
            "percent": round(used / total * 100, 1) if total else 0.0,
        }
    except Exception:
        return {}


def _sample_gpu_pynvml() -> list[dict[str, Any]]:
    gpus: list[dict[str, Any]] = []
    try:
        count = pynvml.nvmlDeviceGetCount()
        for idx in range(count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(idx)
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            name = pynvml.nvmlDeviceGetName(handle)
            if isinstance(name, bytes):
                name = name.decode("utf-8")

            processes = []
            try:
                for proc in pynvml.nvmlDeviceGetComputeRunningProcesses(handle):
                    processes.append(
                        {
                            "pid": int(proc.pid),
                            "used_memory_mb": _mb(getattr(proc, "usedGpuMemory", 0) or 0),
                        }
                    )
            except Exception:
                pass

            gpus.append(
                {
                    "index": idx,
                    "name": name,
                    "used_mb": _mb(mem.used),
                    "total_mb": _mb(mem.total),
                    "percent": round(mem.used / mem.total * 100, 1) if mem.total else 0.0,
                    "gpu_util_percent": getattr(util, "gpu", None),
                    "memory_util_percent": getattr(util, "memory", None),
                    "processes": processes,
                }
            )
    except Exception:
        return []
    return gpus


def _sample_gpu_nvidia_smi() -> list[dict[str, Any]]:
    gpus: list[dict[str, Any]] = []
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.used,memory.total,utilization.gpu,utilization.memory",
                "--format=csv,noheader,nounits",
            ],
            timeout=10,
            stderr=subprocess.DEVNULL,
        ).decode("utf-8")
        for line in out.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 6:
                continue
            used_mb = int(parts[2])
            total_mb = int(parts[3])
            gpus.append(
                {
                    "index": int(parts[0]),
                    "name": parts[1],
                    "used_mb": used_mb,
                    "total_mb": total_mb,
                    "percent": round(used_mb / total_mb * 100, 1) if total_mb else 0.0,
                    "gpu_util_percent": int(parts[4]),
                    "memory_util_percent": int(parts[5]),
                    "processes": [],
                }
            )
    except Exception:
        pass

    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-compute-apps=gpu_uuid,pid,used_memory",
                "--format=csv,noheader,nounits",
            ],
            timeout=10,
            stderr=subprocess.DEVNULL,
        ).decode("utf-8")
        rows = [line.strip() for line in out.strip().splitlines() if line.strip()]
        flat_processes = []
        for row in rows:
            parts = [p.strip() for p in row.split(",")]
            if len(parts) >= 3:
                flat_processes.append({"pid": int(parts[1]), "used_memory_mb": int(parts[2])})
        if flat_processes:
            for gpu in gpus:
                gpu["processes"] = flat_processes
    except Exception:
        pass

    return gpus


def sample_gpu_memory() -> list[dict[str, Any]]:
    if _PYNVML_AVAILABLE:
        gpus = _sample_gpu_pynvml()
        if gpus:
            return gpus
    return _sample_gpu_nvidia_smi()


def sample_process_tree(root_pid: int | None) -> dict[str, Any]:
    if not root_pid or not _PSUTIL_AVAILABLE:
        return {"root_pid": root_pid, "processes": [], "total_rss_gb": 0.0, "total_cpu_percent": 0.0}

    try:
        root = psutil.Process(root_pid)
    except Exception:
        return {"root_pid": root_pid, "processes": [], "total_rss_gb": 0.0, "total_cpu_percent": 0.0}

    processes = []
    total_rss = 0
    total_cpu = 0.0
    try:
        procs = [root] + root.children(recursive=True)
    except Exception:
        procs = [root]

    for proc in procs:
        try:
            mem = proc.memory_info()
            cmdline = " ".join(proc.cmdline())
            cpu = proc.cpu_percent(interval=None)
            rec = {
                "pid": proc.pid,
                "ppid": proc.ppid(),
                "name": proc.name(),
                "status": proc.status(),
                "rss_gb": _gb(mem.rss),
                "vms_gb": _gb(mem.vms),
                "cpu_percent": cpu,
                "num_threads": proc.num_threads(),
                "cmdline": cmdline[:500],
            }
            total_rss += mem.rss
            total_cpu += cpu
            processes.append(rec)
        except Exception:
            continue

    return {
        "root_pid": root_pid,
        "process_count": len(processes),
        "total_rss_gb": _gb(total_rss),
        "total_cpu_percent": round(total_cpu, 1),
        "processes": processes,
    }


def take_snapshot(label: str = "snapshot", root_pid: int | None = None) -> dict[str, Any]:
    return {
        "label": label,
        "timestamp": datetime.now().isoformat(timespec="milliseconds"),
        "elapsed_seconds": 0.0,
        "cpu": sample_cpu_memory(),
        "gpu": sample_gpu_memory(),
        "process_tree": sample_process_tree(root_pid),
    }


class ResourceMonitor:
    def __init__(self, output_path: str | os.PathLike[str], interval_secs: float = 5.0, root_pid: int | None = None):
        self.output_path = Path(output_path)
        self.interval_secs = interval_secs
        self.root_pid = root_pid
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._start_time: float | None = None
        self._nvml_initialized = False

        if _PYNVML_AVAILABLE:
            try:
                pynvml.nvmlInit()
                self._nvml_initialized = True
            except Exception:
                self._nvml_initialized = False

    def set_root_pid(self, pid: int | None) -> None:
        self.root_pid = pid

    def start(self) -> None:
        self._start_time = time.monotonic()
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, name="SFTResourceMonitor", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 10.0) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            self._thread = None
        if _PYNVML_AVAILABLE and self._nvml_initialized:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass

    def _run(self) -> None:
        self._sample_and_write("startup")
        while not self._stop_event.wait(timeout=self.interval_secs):
            self._sample_and_write("periodic")
        self._sample_and_write("final")

    def _sample_and_write(self, label: str) -> None:
        record = take_snapshot(label=label, root_pid=self.root_pid)
        if self._start_time is not None:
            record["elapsed_seconds"] = round(time.monotonic() - self._start_time, 2)
        try:
            with open(self.output_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception as exc:
            print(f"[monitor] failed to write resource sample: {exc}", flush=True)
