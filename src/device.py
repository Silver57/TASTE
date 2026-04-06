"""Device and dtype auto-detection."""

import gc
import random

import torch


def detect_device():
    """Return (device_str, torch_dtype) for the best available accelerator."""
    if torch.cuda.is_available():
        return "cuda", torch.bfloat16
    if torch.backends.mps.is_available():
        return "mps", torch.float16
    return "cpu", torch.float32


DEVICE, TORCH_DTYPE = detect_device()
USE_BF16 = TORCH_DTYPE == torch.bfloat16
USE_FP16 = TORCH_DTYPE == torch.float16 and DEVICE == "cuda"


def free_memory():
    gc.collect()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()


def reset_seeds(seed: int, extra: int = 0):
    s = seed + extra
    random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def gpu_mb() -> float:
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024**2
    return 0.0
