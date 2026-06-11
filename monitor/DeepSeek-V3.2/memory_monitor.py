"""
后台内存监控线程，周期性采样 GPU 显存和 CPU 内存，写入 memory_timeline.jsonl。

支持两种 GPU 采样后端：
  1. pynvml（优先）：pip install nvidia-ml-py
  2. nvidia-smi subprocess（fallback）
"""

import json
import subprocess
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

try:
    import pynvml

    _PYNVML_AVAILABLE = True
except ImportError:
    _PYNVML_AVAILABLE = False

try:
    import psutil

    _PSUTIL_AVAILABLE = True
except ImportError:
    _PSUTIL_AVAILABLE = False


# ---------------------------------------------------------------------------
# GPU 采样
# ---------------------------------------------------------------------------

def _sample_gpu_pynvml() -> List[Dict]:
    """使用 pynvml 采样每块 GPU 的显存使用情况。"""
    results = []
    try:
        device_count = pynvml.nvmlDeviceGetCount()
        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            name = pynvml.nvmlDeviceGetName(handle)
            if isinstance(name, bytes):
                name = name.decode("utf-8")
            used_mb = info.used // (1024 * 1024)
            total_mb = info.total // (1024 * 1024)
            results.append(
                {
                    "index": i,
                    "name": name,
                    "used_mb": used_mb,
                    "total_mb": total_mb,
                    "percent": round(used_mb / total_mb * 100, 1) if total_mb > 0 else 0.0,
                }
            )
    except Exception:
        pass
    return results


def _sample_gpu_nvidia_smi() -> List[Dict]:
    """使用 nvidia-smi 采样每块 GPU 的显存使用情况（pynvml 不可用时的 fallback）。"""
    results = []
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            timeout=10,
            stderr=subprocess.DEVNULL,
        ).decode("utf-8")
        for line in out.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 4:
                continue
            idx = int(parts[0])
            name = parts[1]
            used_mb = int(parts[2])
            total_mb = int(parts[3])
            results.append(
                {
                    "index": idx,
                    "name": name,
                    "used_mb": used_mb,
                    "total_mb": total_mb,
                    "percent": round(used_mb / total_mb * 100, 1) if total_mb > 0 else 0.0,
                }
            )
    except Exception:
        pass
    return results


def sample_gpu_memory() -> List[Dict]:
    """采样 GPU 显存，自动选择后端。"""
    if _PYNVML_AVAILABLE:
        return _sample_gpu_pynvml()
    return _sample_gpu_nvidia_smi()


# ---------------------------------------------------------------------------
# CPU 内存采样
# ---------------------------------------------------------------------------

def sample_cpu_memory() -> Dict:
    """采样系统 CPU 内存使用情况。"""
    if _PSUTIL_AVAILABLE:
        vm = psutil.virtual_memory()
        return {
            "used_gb": round(vm.used / (1024 ** 3), 2),
            "total_gb": round(vm.total / (1024 ** 3), 2),
            "available_gb": round(vm.available / (1024 ** 3), 2),
            "percent": vm.percent,
        }
    # fallback：读取 /proc/meminfo
    try:
        meminfo = {}
        with open("/proc/meminfo") as f:
            for line in f:
                key, val = line.split(":", 1)
                meminfo[key.strip()] = int(val.strip().split()[0])
        total_kb = meminfo.get("MemTotal", 0)
        avail_kb = meminfo.get("MemAvailable", 0)
        used_kb = total_kb - avail_kb
        total_gb = round(total_kb / (1024 ** 2), 2)
        used_gb = round(used_kb / (1024 ** 2), 2)
        avail_gb = round(avail_kb / (1024 ** 2), 2)
        percent = round(used_kb / total_kb * 100, 1) if total_kb > 0 else 0.0
        return {
            "used_gb": used_gb,
            "total_gb": total_gb,
            "available_gb": avail_gb,
            "percent": percent,
        }
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# 快照：一次性采集（用于 startup_memory.json）
# ---------------------------------------------------------------------------

def take_memory_snapshot(label: str = "snapshot") -> Dict:
    """
    取一次完整内存快照，返回包含 gpu 和 cpu 字段的字典。
    用于保存 startup_memory.json。
    """
    return {
        "label": label,
        "timestamp": datetime.now().isoformat(timespec="milliseconds"),
        "gpu": sample_gpu_memory(),
        "cpu": sample_cpu_memory(),
    }


# ---------------------------------------------------------------------------
# 后台轮询线程
# ---------------------------------------------------------------------------

class MemoryMonitor:
    """
    后台守护线程，每隔 interval_secs 秒采样一次 GPU + CPU 内存，
    以 JSONL 格式追加写入 output_path。

    用法：
        monitor = MemoryMonitor(output_path, interval_secs=5)
        monitor.start()
        ...
        monitor.stop()
    """

    def __init__(self, output_path: str, interval_secs: float = 5.0):
        self.output_path = Path(output_path)
        self.interval_secs = interval_secs
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._start_time: Optional[float] = None

        # 初始化 pynvml（只做一次，避免频繁 init/shutdown）
        if _PYNVML_AVAILABLE:
            try:
                pynvml.nvmlInit()
                self._nvml_initialized = True
            except Exception:
                self._nvml_initialized = False
        else:
            self._nvml_initialized = False

    def start(self):
        """启动后台采样线程。"""
        self._start_time = time.monotonic()
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run, name="MemoryMonitor", daemon=True
        )
        self._thread.start()

    def stop(self, timeout: float = 10.0):
        """停止后台采样线程并等待其结束。"""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            self._thread = None
        if _PYNVML_AVAILABLE and self._nvml_initialized:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass

    def _run(self):
        """线程主循环：每隔 interval_secs 采样并写入文件。"""
        while not self._stop_event.wait(timeout=self.interval_secs):
            self._sample_and_write()
        # 线程退出前再采样一次，记录结束时刻
        self._sample_and_write(label="final")

    def _sample_and_write(self, label: str = "periodic"):
        elapsed = (
            round(time.monotonic() - self._start_time, 2)
            if self._start_time is not None
            else 0.0
        )
        record = {
            "label": label,
            "timestamp": datetime.now().isoformat(timespec="milliseconds"),
            "elapsed_seconds": elapsed,
            "gpu": sample_gpu_memory(),
            "cpu": sample_cpu_memory(),
        }
        try:
            with open(self.output_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception as e:
            print(f"[MemoryMonitor] 写入失败: {e}", flush=True)
