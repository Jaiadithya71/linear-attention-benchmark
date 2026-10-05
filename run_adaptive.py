"""Adaptive-rank linear attention vs fixed-rank random-feature attention vs exact SDPA.
Parts: (1) efficiency + error, (2) key-value retrieval (same seeded trials as run_retrieval.py), (3) real Qwen2.5-0.5B.
Same rules as the other scripts: GPU only, measured numbers only, OOM reported as 'oom'."""
import argparse, csv, math, os
import numpy as np
import torch
from lab import attention as A
from lab import adaptive as R
from lab.env import require_cuda, utcnow, gather, write_config
from lab.timing import time_cuda, measure_memory, is_oom
from lab.stats import wilson_interval, bootstrap_ci
from run_retrieval import make_trial

MB = 1024 ** 2
RANKCOLS = ["heads_rank64", "heads_rank256", "heads_rank1024", "heads_exact", "heads_tol_not_met"]
EFF = ["regime", "dtype", "method", "tol", "N", "status", "median_ms", "q25_ms", "q75_ms", "min_ms", "reps", "warmup",
       "peak_extra_mb", "peak_total_mb", "finite_output", "rel_err_mean_vs_exact", "rel_err_max_vs_exact"] + RANKCOLS
RET = ["N", "method", "dtype", "beta", "tol", "trials", "successes", "accuracy", "wilson95_lo", "wilson95_hi",
       "bootstrap95_lo", "bootstrap95_hi", "nonfinite_outputs", "seed", "per_trial_success"] + RANKCOLS
QPF = ["variant", "N", "status", "median_ms", "q25_ms", "q75_ms", "min_ms", "reps", "warmup", "peak_extra_mb",
       "peak_total_mb", "finite_output"] + RANKCOLS
QQF = ["variant", "ctx", "windows", "tokens_scored", "nll_per_token", "perplexity", "nonfinite_windows"] + RANKCOLS
# q and k are multiplied by a per-head scale. Random features approximate exp(q.k/sqrt(D)) well only when |q||k| is small;
# regimes were fixed AFTER a first T4 run showed that scale 1.0 (unit-Gaussian, |x|~8) defeats every rank (relative error > 3).
REGIMES = {"small": [0.25] * 8, "mixed": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0], "large": [1.0] * 8}


def counts(info):
    r, met = info["rank"], info["met"]
    return {"heads_rank64": int((r == 64).sum()), "heads_rank256": int((r == 256).sum()),
            "heads_rank1024": int((r == 1024).sum()), "heads_exact": int((r == 0).sum()),
            "heads_tol_not_met": int(((r != 0) & ~met).sum())}


def add(c, d):
    for k in RANKCOLS:
        c[k] = c.get(k, 0) + d[k]


def variants(tol_sweep, tol):
    v = {"sdpa": None, "rf64": 64, "rf256": 256, "rf1024": 1024}
    v["adaptive_rank"] = ("adaptive", tol, False)
    v["adaptive_rank_fb"] = ("adaptive", tol, True)
    for t in tol_sweep:
        v[f"adaptive_rank_tol{t}"] = ("adaptive", t, False)
        v[f"adaptive_rank_fb_tol{t}"] = ("adaptive", t, True)
    return v


def run_variant(spec, q, k, v, causal=False, probe_q=None):
    if spec is None:
        return A.sdpa_attention(q, k, v, causal), None
    if isinstance(spec, int):
        B, H, N, D = q.shape
        o, _ = R.rf_attention(*(x.float().reshape(B * H, 1, x.shape[-2], D) for x in (q, k, v)), R.omega(spec, D, q.device), causal)
        return o.reshape(B, H, q.shape[-2], D).to(q.dtype), None
    _, t, fb = spec
    return R.adaptive_rank_attention(q, k, v, causal=causal, tol=t, fallback=fb, probe_q=probe_q)


def part_efficiency(args, out):
    rows = []
    var = variants(args.tol_sweep, args.tol)
    for regime in REGIMES:
        for dname, dt in (("fp32", torch.float32), ("fp16", torch.float16)):
            for N in args.Ns:
                g = torch.Generator(device="cuda").manual_seed(args.seed * 1000003 + N)
                q, k, v = (torch.randn(1, 8, N, 64, device="cuda", generator=g) for _ in range(3))
                sc = torch.tensor(REGIMES[regime], device="cuda").view(1, 8, 1, 1)
                q, k = (q * sc).to(dt), (k * sc).to(dt)
                v = v.to(dt)
                ref = A.sdpa_attention(q.float(), k.float(), v.float()) if N <= args.ref_max_n else None
                for name, spec in var.items():
                    row = dict(regime=regime, dtype=dname, method=name, N=N,
                               tol=spec[1] if isinstance(spec, tuple) else "")
                    try:
                        o, info = run_variant(spec, q, k, v)
                        torch.cuda.synchronize()
                        row["finite_output"] = bool(torch.isfinite(o).all().item())
                        if ref is not None:
                            e = ((o.float() - ref).norm(dim=(-1, -2)) / ref.norm(dim=(-1, -2)))
                            row["rel_err_mean_vs_exact"], row["rel_err_max_vs_exact"] = float(e.mean()), float(e.max())
                        if info is not None:
                            row.update(counts(info))
                        del o
                        call = lambda: run_variant(spec, q, k, v)[0]
                        t = time_cuda(call, warmup=args.warmup, reps=args.reps)
                        extra, peak = measure_memory(call)
                        row.update(status="ok", peak_extra_mb=extra / MB, peak_total_mb=peak / MB, **t)
                    except Exception as e:
                        if not is_oom(e):
                            raise
                        row["status"] = "oom"
                    torch.cuda.empty_cache()
                    rows.append(row)
                    print({k_: (round(x, 3) if isinstance(x, float) else x) for k_, x in row.items() if k_ in
                           ("regime", "dtype", "method", "N", "status", "median_ms", "rel_err_mean_vs_exact") or k_ in RANKCOLS}, flush=True)
                del q, k, v, ref
                torch.cuda.empty_cache()
                with open(os.path.join(out, "adaptive_efficiency.csv"), "w", newline="") as f:
                    w = csv.DictWriter(f, EFF); w.writeheader(); w.writerows(rows)


def part_retrieval(args, out):
    rows = []
    var = {"sdpa": None, "rf64": 64, "rf256": 256, "rf1024": 1024, "adaptive_rank": ("adaptive", args.tol, False)}
    for N in args.ret_Ns:
        succ, nonfin, cnt = {}, {}, {}
        for t0 in range(0, args.trials, args.batch):
            tr = [make_trial(N, 64, args.seed, t) for t in range(t0, min(t0 + args.batch, args.trials))]
            K = torch.stack([t[0] for t in tr]).cuda()
            V = torch.stack([t[1] for t in tr]).cuda()
            pos = torch.tensor([t[2] for t in tr], device=K.device)
            Q = K[torch.arange(len(tr)), pos]
            gp = torch.Generator().manual_seed(int(np.random.SeedSequence([args.seed, N, 777, t0]).generate_state(1)[0]))
            pidx = torch.randint(0, N, (len(tr), 16), generator=gp).cuda()
            for dname, dt in (("fp32", torch.float32), ("fp16", torch.float16)):
                Kd, Vd = K.to(dt)[:, None], V.to(dt)[:, None]
                for beta in args.betas:
                    Qd = (beta * Q).to(dt)[:, None, None, :]
                    # probes: beta * random keys (same distribution as the query), never the query itself
                    Pq = (beta * torch.gather(K, 1, pidx[:, :, None].expand(-1, -1, 64))).to(dt)[:, None]
                    mb = max(1, min(len(tr), 2 ** 18 // N))
                    outs = {}
                    for name, spec in var.items():
                        parts, infos = [], []
                        for s in range(0, len(tr), mb):
                            sl = slice(s, s + mb)
                            o, info = run_variant(spec, Qd[sl], Kd[sl], Vd[sl], probe_q=Pq[sl] if isinstance(spec, tuple) else None)
                            parts.append(o); infos.append(info)
                        outs[name] = (torch.cat(parts), infos)
                    # router with exact fallback = router output where the tolerance was met, exact SDPA elsewhere
                    o_ad, infos = outs["adaptive_rank"]
                    met = torch.cat([i["met"] for i in infos]).reshape(-1)
                    outs["adaptive_rank_fb"] = (torch.where(met.view(-1, 1, 1, 1), o_ad, outs["sdpa"][0]), infos)
                    for name, (o, infos) in outs.items():
                        o = o.float().reshape(len(tr), -1)
                        fin = torch.isfinite(o).all(-1)
                        o = torch.nan_to_num(o)
                        sims = torch.einsum("bd,bnd->bn", o / o.norm(dim=-1, keepdim=True).clamp_min(1e-30), V)
                        ok = ((sims.argmax(-1) == pos) & fin).cpu().numpy().astype(int)
                        key = (name, dname, beta)
                        succ.setdefault(key, []).extend(ok.tolist())
                        nonfin[key] = nonfin.get(key, 0) + int((~fin).sum())
                        if name.startswith("adaptive"):
                            c = cnt.setdefault(key, {})
                            for i in infos:
                                d = counts(i)
                                if name == "adaptive_rank_fb":  # unmet inputs were answered by exact attention
                                    d = dict(d, heads_exact=d["heads_tol_not_met"], heads_rank1024=d["heads_rank1024"] - d["heads_tol_not_met"],
                                             heads_tol_not_met=0)
                                add(c, d)
            del K, V, Kd, Vd
        for (name, dname, beta), bits in succ.items():
            n, s = len(bits), int(sum(bits))
            wl, wh = wilson_interval(s, n)
            bl, bh = bootstrap_ci(bits, seed=args.seed)
            row = dict(N=N, method=name, dtype=dname, beta=beta, tol=args.tol if name.startswith("adaptive") else "", trials=n,
                       successes=s, accuracy=s / n, wilson95_lo=wl, wilson95_hi=wh, bootstrap95_lo=bl, bootstrap95_hi=bh,
                       nonfinite_outputs=nonfin[(name, dname, beta)], seed=args.seed, per_trial_success="".join(map(str, bits)))
            row.update(cnt.get((name, dname, beta), {}))
            rows.append(row)
            print(N, name, dname, beta, f"{s}/{n}", {k: row[k] for k in RANKCOLS if k in row}, flush=True)
        with open(os.path.join(out, "adaptive_retrieval.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, RET); w.writeheader(); w.writerows(rows)


def part_qwen(args, out):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import load_dataset
    from lab.qwen_patch import patched_attention, ROUTER_STATS
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16, attn_implementation="sdpa").cuda().eval()
    text = "\n\n".join(load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")["text"])
    ids_all = tok(text, return_tensors="pt").input_ids[0]
    need = max(max(args.qwen_Ns), args.ppl_ctx * args.ppl_windows)
    if ids_all.numel() < need:
        raise RuntimeError("corpus too short")
    names = ["sdpa", "sdpa_mha", "rf64", "rf256", "rf1024", "adaptive_rank", "adaptive_rank_fb"]
    kinds = {n: n for n in names}
    pf, qq = [], []
    with torch.no_grad():
        for N in args.qwen_Ns:
            ids = ids_all[:N][None].cuda()
            for var in names:
                row = dict(variant=var, N=N)
                ROUTER_STATS.clear()
                with patched_attention(kinds[var], tol=args.tol):
                    call = lambda: model.model(input_ids=ids, use_cache=False).last_hidden_state
                    try:
                        o = call(); torch.cuda.synchronize()
                        row["finite_output"] = bool(torch.isfinite(o).all().item()); del o
                        stats = dict(ROUTER_STATS)
                        t = time_cuda(call, warmup=args.warmup, reps=min(args.reps, 5))
                        extra, peak = measure_memory(call)
                        row.update(status="ok", peak_extra_mb=extra / MB, peak_total_mb=peak / MB, **t)
                        row.update(stats)
                    except Exception as e:
                        if not is_oom(e):
                            raise
                        row["status"] = "oom"
                torch.cuda.empty_cache()
                pf.append(row)
                print({k: (round(x, 2) if isinstance(x, float) else x) for k, x in row.items() if k in ("variant", "N", "status", "median_ms", "peak_extra_mb", "finite_output") or k in RANKCOLS}, flush=True)
            with open(os.path.join(out, "adaptive_qwen_prefill.csv"), "w", newline="") as f:
                w = csv.DictWriter(f, QPF); w.writeheader(); w.writerows(pf)
        C = args.ppl_ctx
        for var in names:
            nll, cnt, bad = 0.0, 0, 0
            ROUTER_STATS.clear()
            with patched_attention(kinds[var], tol=args.tol):
                for i in range(args.ppl_windows):
                    ids = ids_all[i * C:(i + 1) * C][None].cuda()
                    logits = model(input_ids=ids, use_cache=False).logits[0, :-1].float()
                    l = torch.nn.functional.cross_entropy(logits, ids[0, 1:], reduction="sum").item()
                    if not math.isfinite(l):
                        bad += 1; continue
                    nll += l; cnt += C - 1
            nl = nll / cnt if cnt else float("nan")
            row = dict(variant=var, ctx=C, windows=args.ppl_windows, tokens_scored=cnt, nll_per_token=nl,
                       perplexity=math.exp(nl) if cnt and nl < 700 else float("nan"), nonfinite_windows=bad)
            row.update(dict(ROUTER_STATS))
            qq.append(row); print(row, flush=True)
    with open(os.path.join(out, "adaptive_qwen_quality.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, QQF); w.writeheader(); w.writerows(qq)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="results")
    p.add_argument("--require-gpu", default="T4")
    p.add_argument("--parts", nargs="+", default=["efficiency", "retrieval", "qwen"])
    p.add_argument("--tol", type=float, default=0.1)
    p.add_argument("--tol-sweep", type=float, nargs="*", default=[0.05, 0.2, 0.4])
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--Ns", type=int, nargs="+", default=[1024, 4096, 16384, 65536])
    p.add_argument("--ref-max-n", type=int, default=16384)
    p.add_argument("--reps", type=int, default=15)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--ret-Ns", type=int, nargs="+", default=[256, 1024, 4096, 16384, 65536])
    p.add_argument("--trials", type=int, default=200)
    p.add_argument("--batch", type=int, default=20)
    p.add_argument("--betas", type=float, nargs="+", default=[16.0, 64.0, 256.0])
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    p.add_argument("--qwen-Ns", type=int, nargs="+", default=[1024, 2048, 4096, 8192])
    p.add_argument("--ppl-ctx", type=int, default=2048)
    p.add_argument("--ppl-windows", type=int, default=8)
    args = p.parse_args()
    assert args.trials >= 200
    gpu = require_cuda(args.require_gpu)
    os.makedirs(args.out, exist_ok=True)
    started = utcnow()
    print("GPU:", gpu, flush=True)
    for part in args.parts:
        {"efficiency": part_efficiency, "retrieval": part_retrieval, "qwen": part_qwen}[part](args, args.out)
    write_config(os.path.join(args.out, "run_config_adaptive.json"), gather("adaptive", started, utcnow(), args))


if __name__ == "__main__":
    main()
