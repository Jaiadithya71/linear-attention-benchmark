"""Key-value retrieval (needle among distractors). One query vector equal to the needle key; N unit-norm
random keys and values. Success = the attention output is closest (cosine) to the needle's value among all N values.
Every trial has its own seed derived from (seed, N, trial); data is generated on CPU so it is identical across methods,
dtypes and GPUs."""
import argparse, csv, os
import numpy as np
import torch
from lab import attention as A
from lab.env import require_cuda, utcnow, gather, write_config
from lab.stats import wilson_interval, bootstrap_ci

FIELDS = ["N", "method", "dtype", "beta", "trials", "successes", "accuracy", "wilson95_lo", "wilson95_hi",
          "bootstrap95_lo", "bootstrap95_hi", "nonfinite_outputs", "window", "seed", "per_trial_success"]


def make_trial(N, D, seed, trial):
    s = int(np.random.SeedSequence([seed, N, trial]).generate_state(1)[0])
    g = torch.Generator().manual_seed(s)
    K = torch.randn(N, D, generator=g); K = K / K.norm(dim=-1, keepdim=True)
    V = torch.randn(N, D, generator=g); V = V / V.norm(dim=-1, keepdim=True)
    pos = int(torch.randint(0, N, (1,), generator=g).item())
    return K, V, pos


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="results")
    p.add_argument("--require-gpu", default="T4")
    p.add_argument("--Ns", type=int, nargs="+", default=[64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536])
    p.add_argument("--trials", type=int, default=300)
    p.add_argument("--batch", type=int, default=20)
    p.add_argument("--D", type=int, default=64)
    p.add_argument("--betas", type=float, nargs="+", default=[64.0, 256.0])
    p.add_argument("--window", type=int, default=256)
    p.add_argument("--seed", type=int, default=1234)
    args = p.parse_args()
    assert args.trials >= 200
    gpu = require_cuda(args.require_gpu)
    os.makedirs(args.out, exist_ok=True)
    started = utcnow()
    print("GPU:", gpu, flush=True)
    rows = []
    for N in args.Ns:
        succ = {}
        nonfin = {}
        for t0 in range(0, args.trials, args.batch):
            tr = [make_trial(N, args.D, args.seed, t) for t in range(t0, min(t0 + args.batch, args.trials))]
            K = torch.stack([t[0] for t in tr]).cuda()
            V = torch.stack([t[1] for t in tr]).cuda()
            pos = torch.tensor([t[2] for t in tr], device=K.device)
            Q = K[torch.arange(len(tr)), pos]  # [b, D]
            for dname, dt in [("fp32", torch.float32), ("fp16", torch.float16)]:
                Kd, Vd = K.to(dt)[:, None], V.to(dt)[:, None]
                for beta in args.betas:
                    Qd = (beta * Q).to(dt)[:, None, None, :]
                    fns = {
                        "sdpa": lambda: A.sdpa_attention(Qd, Kd, Vd),
                        "local_window": lambda: A.local_window_decode(Qd, Kd, Vd, args.window),
                        "linear": lambda: A.linear_attention(Qd, Kd, Vd),
                    }
                    if dt == torch.float16:
                        fns["linear_acc32"] = lambda: A.linear_attention(Qd, Kd, Vd, acc_dtype=torch.float32)
                    for m, fn in fns.items():
                        out = fn().float().reshape(len(tr), -1)
                        fin = torch.isfinite(out).all(-1)
                        out = torch.nan_to_num(out)
                        sims = torch.einsum("bd,bnd->bn", out / out.norm(dim=-1, keepdim=True).clamp_min(1e-30), V)
                        ok = ((sims.argmax(-1) == pos) & fin).cpu().numpy().astype(int)
                        key = (m, dname, beta)
                        succ.setdefault(key, []).extend(ok.tolist())
                        nonfin[key] = nonfin.get(key, 0) + int((~fin).sum().item())
            del K, V, Kd, Vd
        for (m, dname, beta), bits in succ.items():
            n, s = len(bits), int(sum(bits))
            wl, wh = wilson_interval(s, n)
            bl, bh = bootstrap_ci(bits, seed=args.seed)
            rows.append(dict(N=N, method=m, dtype=dname, beta=beta, trials=n, successes=s, accuracy=s / n,
                             wilson95_lo=wl, wilson95_hi=wh, bootstrap95_lo=bl, bootstrap95_hi=bh,
                             nonfinite_outputs=nonfin[(m, dname, beta)],
                             window=args.window if m == "local_window" else "", seed=args.seed,
                             per_trial_success="".join(map(str, bits))))
            print(N, m, dname, beta, f"{s}/{n}", f"[{wl:.3f},{wh:.3f}]", "nonfinite", nonfin[(m, dname, beta)], flush=True)
        with open(os.path.join(args.out, "retrieval.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, FIELDS); w.writeheader(); w.writerows(rows)
    write_config(os.path.join(args.out, "run_config_retrieval.json"), gather("retrieval", started, utcnow(), args))


if __name__ == "__main__":
    main()
