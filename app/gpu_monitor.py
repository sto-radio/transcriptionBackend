import os
import threading

from pynvml import (
    NVMLError,
    nvmlDeviceGetHandleByIndex,
    nvmlDeviceGetMemoryInfo,
    nvmlDeviceGetUtilizationRates,
    nvmlInit,
)

GPU_INDEX = int(os.getenv("WHISPERX_GPU_INDEX", "0"))
MIN_FREE_MB = int(os.getenv("WHISPERX_MIN_FREE_GPU_MB", "2000"))
MAX_GPU_UTIL = int(os.getenv("WHISPERX_MAX_GPU_UTIL", "85"))

nvml_initialized = False
nvml_lock = threading.Lock()


def ensure_nvml_initialized():
    global nvml_initialized

    with nvml_lock:
        if nvml_initialized:
            return
        nvmlInit()
        nvml_initialized = True


def get_gpu_status(gpu_index=GPU_INDEX):
    try:
        ensure_nvml_initialized()
        handle = nvmlDeviceGetHandleByIndex(gpu_index)
        mem = nvmlDeviceGetMemoryInfo(handle)
        util = nvmlDeviceGetUtilizationRates(handle)
    except NVMLError as exc:
        raise RuntimeError(
            f"No se pudo leer el estado de la GPU {gpu_index}: {exc}"
        ) from exc

    return {
        "gpu_util_percent": util.gpu,
        "mem_free_mb": round(mem.free / 1024 / 1024),
        "mem_used_mb": round(mem.used / 1024 / 1024),
        "mem_total_mb": round(mem.total / 1024 / 1024),
    }


def gpu_has_capacity():
    if os.getenv("WHISPERX_DEVICE", "cuda").startswith("cpu"):
        return True

    status = get_gpu_status()

    return (
        status["mem_free_mb"] >= MIN_FREE_MB
        and status["gpu_util_percent"] <= MAX_GPU_UTIL
    )
