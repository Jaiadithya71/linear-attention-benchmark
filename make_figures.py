"""Plots straight from results/*.csv. Nothing is plotted that is not in a CSV."""
import os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

R = sys.argv[1] if len(sys.argv) > 1 else "results"
F = sys.argv[2] if len(sys.argv) > 2 else "figures"
os.makedirs(F, exist_ok=True)
COL = {"softmax_naive": "tab:red", "sdpa": "tab:blue", "local_window": "tab:green", "linear": "tab:orange", "linear_acc32": "tab:purple"}

p = os.path.join(R, "efficiency.csv")
if os.path.exists(p):
    d = pd.read_csv(p)
    for metric, ylabel, fn in [("median_ms", "median latency (ms), error bars = IQR", "latency"),
                               ("peak_extra_mb", "peak extra memory (MB)", "memory")]:
        fig, axs = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
        for ax, dt in zip(axs, ["fp32", "fp16"]):
            for m, g in d[(d.dtype == dt) & (d.status == "ok")].groupby("method"):
                g = g.sort_values("N")
                if metric == "median_ms":
                    ax.errorbar(g.N, g.median_ms, yerr=[g.median_ms - g.q25_ms, g.q75_ms - g.median_ms], label=m, color=COL.get(m), marker="o", ms=3, capsize=2)
                else:
                    ax.plot(g.N, g[metric], label=m, color=COL.get(m), marker="o", ms=3)
            oom = d[(d.dtype == dt) & (d.status == "oom")]
            for m, g in oom.groupby("method"):
                ax.scatter([g.N.min()], [ax.get_ylim()[0] if ax.get_ylim()[0] > 0 else 1], marker="x", color=COL.get(m), label=f"{m}: OOM from N={g.N.min()}")
            ax.set_xscale("log", base=2); ax.set_yscale("log"); ax.set_title(f"{dt}"); ax.set_xlabel("N"); ax.grid(alpha=.3)
        axs[0].set_ylabel(ylabel); axs[1].legend(fontsize=7)
        fig.tight_layout(); fig.savefig(os.path.join(F, f"{fn}.png"), dpi=130); plt.close(fig)

p = os.path.join(R, "retrieval.csv")
if os.path.exists(p):
    d = pd.read_csv(p)
    betas = sorted(d.beta.unique())
    fig, axs = plt.subplots(len(betas), 2, figsize=(12, 4 * len(betas)), sharey=True, squeeze=False)
    for i, b in enumerate(betas):
        for j, dt in enumerate(["fp32", "fp16"]):
            ax = axs[i][j]
            for m, g in d[(d.dtype == dt) & (d.beta == b)].groupby("method"):
                g = g.sort_values("N")
                ax.errorbar(g.N, g.accuracy, yerr=[(g.accuracy - g.wilson95_lo).clip(lower=0), (g.wilson95_hi - g.accuracy).clip(lower=0)], label=m, color=COL.get(m), marker="o", ms=3, capsize=2)
            ax.set_xscale("log", base=2); ax.set_ylim(-0.02, 1.02); ax.set_title(f"{dt}, query scale beta={b:g}"); ax.set_xlabel("N"); ax.grid(alpha=.3)
        axs[i][0].set_ylabel("retrieval accuracy (95% Wilson CI)")
    axs[0][1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(F, "retrieval.png"), dpi=130); plt.close(fig)

p = os.path.join(R, "qwen_prefill.csv")
if os.path.exists(p):
    d = pd.read_csv(p)
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.5))
    for var, g in d[d.status == "ok"].groupby("variant"):
        g = g.sort_values("N")
        axs[0].errorbar(g.N, g.median_ms, yerr=[g.median_ms - g.q25_ms, g.q75_ms - g.median_ms], label=var, marker="o", ms=3, capsize=2)
        axs[1].plot(g.N, g.peak_extra_mb, label=var, marker="o", ms=3)
    for ax, t in zip(axs, ["Qwen2.5-0.5B prefill latency (ms)", "peak extra memory (MB)"]):
        ax.set_xscale("log", base=2); ax.set_yscale("log"); ax.set_title(t); ax.set_xlabel("context length"); ax.grid(alpha=.3)
    axs[0].legend()
    fig.tight_layout(); fig.savefig(os.path.join(F, "qwen_prefill.png"), dpi=130); plt.close(fig)
print("figures written to", F)

p = os.path.join(R, "adaptive_efficiency.csv")
if os.path.exists(p):
    d = pd.read_csv(p)
    d = d[(d.dtype == "fp32") & d.status.eq("ok") & d.method.isin(["sdpa", "rf64", "rf256", "rf1024", "adaptive_rank", "adaptive_rank_fb"])]
    fig, axs = plt.subplots(1, 3, figsize=(17, 4.5), sharey=True)
    for ax, rg in zip(axs, ["small", "mixed", "large"]):
        for m, g in d[d.regime == rg].groupby("method"):
            g = g.sort_values("N")
            ax.errorbar(g.N, g.median_ms, yerr=[g.median_ms - g.q25_ms, g.q75_ms - g.median_ms], label=m, marker="o", ms=3, capsize=2)
        ax.set_xscale("log", base=2); ax.set_yscale("log"); ax.set_title(f"adaptive-rank latency, regime={rg}, fp32"); ax.set_xlabel("N"); ax.grid(alpha=.3)
    axs[0].set_ylabel("median ms (IQR)"); axs[2].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(F, "adaptive_latency.png"), dpi=130); plt.close(fig)
