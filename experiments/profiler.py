"""Profiler for wall time, GPU peak memory, and CPU RSS delta.

    with Profiler(label="task_0") as p:
        method.fit(...)
    print(p.result.summary())
"""

import os
import time
from dataclasses import dataclass

import torch

try:
    import psutil

    _PSUTIL_AVAILABLE = True
except ImportError:
    _PSUTIL_AVAILABLE = False


@dataclass
class ProfileResult:
    time_s: float = 0.0
    gpu_peak_allocated_mb: float = 0.0
    gpu_peak_reserved_mb: float = 0.0
    cpu_start_mb: float = 0.0
    cpu_end_mb: float = 0.0
    label: str = ""

    @property
    def cpu_delta_mb(self):
        return self.cpu_end_mb - self.cpu_start_mb

    def summary(self):
        return {
            "label": self.label,
            "time_s": round(self.time_s, 4),
            "gpu_peak_allocated_mb": round(self.gpu_peak_allocated_mb, 2),
            "gpu_peak_reserved_mb": round(self.gpu_peak_reserved_mb, 2),
            "cpu_start_mb": round(self.cpu_start_mb, 2),
            "cpu_end_mb": round(self.cpu_end_mb, 2),
            "cpu_delta_mb": round(self.cpu_delta_mb, 2),
        }


def get_cpu_memory_mb():
    if not _PSUTIL_AVAILABLE:
        return 0.0
    return psutil.Process(os.getpid()).memory_info().rss / 1024**2


def reset_gpu_stats():
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def read_gpu_peak_mb():
    if not torch.cuda.is_available():
        return 0.0, 0.0
    allocated = torch.cuda.max_memory_allocated() / 1024**2
    reserved = torch.cuda.max_memory_reserved() / 1024**2
    return allocated, reserved


class Profiler:
    def __init__(self, label="", sync_cuda=True):
        self.label = label
        self.sync_cuda = sync_cuda
        self.result = None

    def __enter__(self):
        reset_gpu_stats()
        if self.sync_cuda and torch.cuda.is_available():
            torch.cuda.synchronize()
        self._cpu_start = get_cpu_memory_mb()
        self._t_start = time.perf_counter()
        return self

    def __exit__(self, *_):
        if self.sync_cuda and torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - self._t_start
        gpu_alloc, gpu_reserved = read_gpu_peak_mb()
        cpu_end = get_cpu_memory_mb()

        self.result = ProfileResult(
            time_s=elapsed,
            gpu_peak_allocated_mb=gpu_alloc,
            gpu_peak_reserved_mb=gpu_reserved,
            cpu_start_mb=self._cpu_start,
            cpu_end_mb=cpu_end,
            label=self.label,
        )
