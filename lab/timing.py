import gc
import torch
from .stats import summarize_times


def is_oom(e):
    return isinstance(e, torch.cuda.OutOfMemoryError)


def time_cuda(fn, warmup=3, reps=20, slow_ms=2000.0, slow_reps=7):
    """Median/IQR of GPU time via torch.cuda.Event. If the first warmup call is slower than
    slow_ms, only slow_reps timed repetitions are used (the actual count is reported)."""
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record(); fn(); e.record(); e.synchronize()
    first = s.elapsed_time(e)
    if first > slow_ms:
        reps, warmup = min(reps, slow_reps), 0
    for _ in range(max(warmup - 1, 0)):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record(); fn(); e.record(); e.synchronize()
        ts.append(s.elapsed_time(e))
    r = summarize_times(ts)
    r.update(reps=reps, warmup=warmup)
    return r


def measure_memory(fn):
    """Peak allocated bytes during one call: (peak above pre-call allocation, absolute peak).
    Includes the output tensor; excludes inputs already allocated before the call."""
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    base = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    out = fn()
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated()
    del out
    return peak - base, peak
