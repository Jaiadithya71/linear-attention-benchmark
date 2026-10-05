"""Adaptive-rank linear attention.

"Rank" = number of positive random features m (FAVOR+ style) used to approximate the softmax kernel
exp(q.k / sqrt(D)):  phi(x) = exp(w_i . x~ - |x~|^2 / 2),  x~ = x * D^-1/4,  w_i ~ N(0, I).
Cost of linear attention grows with m, accuracy improves with m.

Router (per input = per (batch element, head) pair):
  1. take n_probe probe queries and compute their EXACT softmax attention over all keys (cheap: n_probe x N);
  2. try ranks in ascending order (default 64, 256, 1024); the first rank whose output on the probe queries has
     mean relative L2 error <= tol is accepted for that input; inputs that fail move to the next rank;
  3. inputs that fail every rank: fallback=True -> exact attention (reported as rank 0 = "exact");
     fallback=False -> the largest rank is used anyway and the input is flagged as tolerance-not-met.
Everything is computed in fp32 internally. tol, n_probe, ranks and seeds are fixed hyperparameters, not tuned on test results.
The router's probe cost and any wasted lower-rank attempts are part of its measured latency/memory.
"""
import torch

RANKS = (64, 256, 1024)
_OMEGA = {}


def omega(m, D, device, seed=0):
    key = (m, D, str(device), seed)
    if key not in _OMEGA:
        g = torch.Generator(device="cpu").manual_seed(seed * 7919 + m)
        _OMEGA[key] = torch.randn(m, D, generator=g).to(device)
    return _OMEGA[key]


def features(x, w, is_key):
    """x [G,1,N,D] fp32 -> positive random features [G,1,N,m] (constants that cancel in the ratio are dropped)."""
    xs = x * x.shape[-1] ** -0.25
    logit = xs @ w.T - (xs * xs).sum(-1, keepdim=True) / 2
    if is_key:
        logit = logit - logit.amax(dim=(-2, -1), keepdim=True)
    else:
        logit = logit - logit.amax(-1, keepdim=True)
    return torch.exp(logit)


def rf_attention(q, k, v, w, causal=False, chunk=256, extra_q=None):
    """Random-feature linear attention on [G,1,N,D] fp32 tensors. extra_q: additional (non-causal) query rows
    evaluated against the same key/value state."""
    qf, kf = features(q, w, False), features(k, w, True)
    if not causal:
        kv = kf.transpose(-1, -2) @ v
        z = kf.sum(-2).unsqueeze(-1)
        out = (qf @ kv) / (qf @ z)
        if extra_q is None:
            return out, None
        pf = features(extra_q, w, False)
        return out, (pf @ kv) / (pf @ z)
    N, m, Dv = q.shape[-2], w.shape[0], v.shape[-1]
    st = q.new_zeros(*q.shape[:-2], m, Dv)
    zs = q.new_zeros(*q.shape[:-2], m, 1)
    outs = []
    for s in range(0, N, chunk):
        qc, kc, vc = qf[..., s:s + chunk, :], kf[..., s:s + chunk, :], v[..., s:s + chunk, :]
        A = (qc @ kc.transpose(-1, -2)).tril()
        outs.append((qc @ st + A @ vc) / (qc @ zs + A.sum(-1, keepdim=True)))
        st = st + kc.transpose(-1, -2) @ vc
        zs = zs + kc.sum(-2).unsqueeze(-1)
    return torch.cat(outs, -2), None


def exact_rows(qp, k, v, row_idx=None):
    """Exact softmax attention for probe rows qp [G,1,P,D]. row_idx [P] = their sequence positions (causal mask)."""
    s = (qp @ k.transpose(-1, -2)) * k.shape[-1] ** -0.5
    if row_idx is not None:
        j = torch.arange(k.shape[-2], device=k.device)
        s = s.masked_fill(j[None, :] > row_idx[:, None], float("-inf"))
    return torch.softmax(s, -1) @ v


def adaptive_rank_attention(q, k, v, causal=False, ranks=RANKS, tol=0.1, n_probe=16, fallback=False,
                            probe_q=None, seed=0, max_elems=2 ** 28, chunk=256):
    """q,k,v [B,H,N,D]. probe_q: optional [B,H,P,D] probe queries (non-causal only) drawn by the caller from the
    expected query distribution; default: n_probe rows of q itself (seeded positions).
    Returns (out in q.dtype, info) with info['rank'] [B,H] int (0 = exact) and info['met'] [B,H] bool."""
    B, H, Nq, D = q.shape
    N = k.shape[-2]
    G = B * H
    dt = q.dtype
    q32, k32, v32 = (x.float().reshape(G, 1, x.shape[-2], D) for x in (q, k, v))
    if probe_q is None:
        g = torch.Generator(device="cpu").manual_seed(seed)
        idx = torch.randperm(Nq, generator=g)[:min(n_probe, Nq)].sort().values.to(q.device)
        qp = q32[:, :, idx]
        ridx = idx if causal else None
        extra = None
    else:
        assert not causal
        idx, ridx = None, None
        qp = probe_q.float().reshape(G, 1, probe_q.shape[-2], D)
        extra = qp
    exact = exact_rows(qp, k32, v32, ridx)
    out = torch.empty_like(q32)
    rank = torch.zeros(G, dtype=torch.long, device=q.device)
    met = torch.zeros(G, dtype=torch.bool, device=q.device)
    pending = torch.arange(G, device=q.device)
    for ri, m in enumerate(ranks):
        if pending.numel() == 0:
            break
        w = omega(m, D, q.device, seed)
        group = max(1, max_elems // (N * m))
        keep = []
        for s in range(0, pending.numel(), group):
            sel = pending[s:s + group]
            o, pa = rf_attention(q32[sel], k32[sel], v32[sel], w, causal, chunk, extra[sel] if extra is not None else None)
            if pa is None:
                pa = o[:, :, idx]
            rel = ((pa - exact[sel]).norm(dim=-1) / exact[sel].norm(dim=-1).clamp_min(1e-30)).mean(dim=(-1, -2))
            ok = rel <= tol  # NaN -> False
            last = ri == len(ranks) - 1
            use = ok if (fallback or not last) else torch.ones_like(ok)
            out[sel[use]] = o[use]
            rank[sel[use]] = m
            met[sel[ok]] = True
            keep.append(sel[~use])
        pending = torch.cat(keep) if keep else pending[:0]
    if pending.numel():  # only reachable with fallback=True: exact attention for the unresolved inputs
        qs, ks, vs = q32[pending], k32[pending], v32[pending]
        s = (qs @ ks.transpose(-1, -2)) * D ** -0.5
        if causal:
            s = s.masked_fill(torch.ones(N, N, dtype=torch.bool, device=q.device).triu(1), float("-inf"))
        out[pending] = torch.softmax(s, -1) @ vs
        rank[pending] = 0
    return out.reshape(B, H, Nq, D).to(dt), dict(rank=rank.reshape(B, H), met=met.reshape(B, H))
