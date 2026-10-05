"""Efficiency benchmark: latency (CUDA events) and peak memory for softmax / SDPA / local-window / linear.
Non-causal, B x H x N x D inputs, fp16 and fp32. No CPU path, no estimated numbers."""
import argparse, csv, os, time
import torch
from lab import attention as A
from lab.env import require_cuda, utcnow, gather, write_config
from lab.timing import time_cuda, measure_memory, is_oom

FIELDS = ["dtype", "method", "N", "B", "H", "D", "window", "status", "error", "median_ms", "q25_ms", "q75_ms",
          "min_ms", "reps", "warmup", "peak_extra_mb", "peak_total_mb", "finite_output",
          "max_abs_err_vs_fp32_sdpa"]
MB = 1024 ** 2


def methods(dtype, window):
    m = {
        "softmax_naive": lambda q, k, v: A.softmax_attention(q, k, v),
        "sdpa": lambda q, k, v: A.sdpa_attention(q, k, v),
        "local_window": lambda q, k, v: A.local_window_attention(q, k, v, window=window),
        "linear": lambda q, k, v: A.linear_attention(q, k, v),
    }
    if dtype == torch.float16:  # in fp32 this would be identical to "linear"
        m["linear_acc32"] = lambda q, k, v: A.linear_attention(q, k, v, acc_dtype=torch.float32)
    return m


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="results")
    p.add_argument("--require-gpu", default="T4")
    p.add_argument("--Ns", type=int, nargs="+", default=[512, 1024, 2048, 4096, 8192, 16384, 32768, 65536])
    p.add_argument("--B", type=int, default=1)
    p.add_argument("--H", type=int, default=8)
    p.add_argument("--D", type=int, default=64)
    p.add_argument("--window", type=int, default=256)
    p.add_argument("--reps", type=int, default=20)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--ref-max-n", type=int, default=4096, help="error vs fp32 SDPA computed only up to this N")
    args = p.parse_args()
    gpu = require_cuda(args.require_gpu)
    os.makedirs(args.out, exist_ok=True)
    started = utcnow()
    print("GPU:", gpu, "torch", torch.__version__, flush=True)
    rows = []
    for dtype_name, dtype in [("fp32", torch.float32), ("fp16", torch.float16)]:
        for N in args.Ns:
            g = torch.Generator(device="cuda").manual_seed(args.seed * 1000003 + N)
            base = [torch.randn(args.B, args.H, N, args.D, device="cuda", generator=g) for _ in range(3)]
            q, k, v = (t.to(dtype) for t in base)
            ref = None
            if N <= args.ref_max_n:
                ref = A.sdpa_attention(*base).float()
            for name, fn in methods(dtype, args.window).items():
                row = dict(dtype=dtype_name, method=name, N=N, B=args.B, H=args.H, D=args.D,
                           window=args.window if name == "local_window" else "", error="")
                call = lambda: fn(q, k, v)
                try:
                    out = call()
                    torch.cuda.synchronize()
                    row["finite_output"] = bool(torch.isfinite(out).all().item())
                    if ref is not None:
                        row["max_abs_err_vs_fp32_sdpa"] = float((out.float() - ref).abs().max().item())
                    del out
                    t = time_cuda(call, warmup=args.warmup, reps=args.reps)
                    extra, peak = measure_memory(call)
                    row.update(status="ok", peak_extra_mb=extra / MB, peak_total_mb=peak / MB, **t)
                except Exception as e:  # only OOM is an expected, reportable outcome
                    if not is_oom(e):
                        raise
                    row.update(status="oom", error="torch.cuda.OutOfMemoryError")
                torch.cuda.empty_cache()
                rows.append(row)
                print({k_: (round(x, 3) if isinstance(x, float) else x) for k_, x in row.items()
                       if k_ in ("dtype", "method", "N", "status", "median_ms", "peak_extra_mb", "finite_output")}, flush=True)
            del base, q, k, v, ref
            torch.cuda.empty_cache()
            with open(os.path.join(args.out, "efficiency.csv"), "w", newline="") as f:
                w = csv.DictWriter(f, FIELDS); w.writeheader(); w.writerows(rows)
    write_config(os.path.join(args.out, "run_config_efficiency.json"), gather("efficiency", started, utcnow(), args))


if __name__ == "__main__":
    main()
