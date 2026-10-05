"""Attention implementations. All take q,k,v of shape [B, H, N, D] and return [B, H, Nq, D].

softmax_attention   naive: materialises the full N x N score matrix
sdpa_attention      torch.nn.functional.scaled_dot_product_attention (PyTorch picks the kernel)
local_window_attention  blocked local attention: a query in block b sees the keys of blocks b-1, b
                    (and b+1 if not causal). Causal also masks keys after the query.
local_window_decode single (or few) trailing queries against a blocked window; identical key set
                    to local_window_attention for the last block (non-causal).
linear_attention    kernelised linear attention, feature map phi(x) = relu(x) + 1.
                    acc_dtype=None computes in the input dtype (so fp16 can overflow, and that is
                    reported, not hidden). acc_dtype=torch.float32 accumulates sums in fp32.
"""
import math
import torch
import torch.nn.functional as F


def softmax_attention(q, k, v, causal=False):
    scale = q.shape[-1] ** -0.5
    s = (q * scale) @ k.transpose(-1, -2)
    if causal:
        n, m = s.shape[-2], s.shape[-1]
        mask = torch.ones(n, m, dtype=torch.bool, device=s.device).triu(m - n + 1)
        s = s.masked_fill(mask, float("-inf"))
    return torch.softmax(s, dim=-1) @ v


def sdpa_attention(q, k, v, causal=False):
    return F.scaled_dot_product_attention(q, k, v, is_causal=causal)


def local_window_attention(q, k, v, window=256, causal=False):
    B, H, N, D = q.shape
    w = window
    nb = -(-N // w)
    pad = nb * w - N
    if pad:
        q, k, v = (F.pad(x, (0, 0, 0, pad)) for x in (q, k, v))
    qb = q.reshape(B, H, nb, w, D)
    kb = k.reshape(B, H, nb, w, D)
    vb = v.reshape(B, H, nb, w, D)
    zk = torch.zeros_like(kb[:, :, :1])
    zv = torch.zeros_like(vb[:, :, :1])
    prev_k = torch.cat([zk, kb[:, :, :-1]], 2)
    prev_v = torch.cat([zv, vb[:, :, :-1]], 2)
    if causal:
        keys = torch.cat([prev_k, kb], 3)
        vals = torch.cat([prev_v, vb], 3)
        nblocks_k = 2
    else:
        next_k = torch.cat([kb[:, :, 1:], zk], 2)
        next_v = torch.cat([vb[:, :, 1:], zv], 2)
        keys = torch.cat([prev_k, kb, next_k], 3)
        vals = torch.cat([prev_v, vb, next_v], 3)
        nblocks_k = 3
    K = nblocks_k * w
    dev = q.device
    blk = torch.arange(nb, device=dev).view(nb, 1, 1)
    qpos = blk * w + torch.arange(w, device=dev).view(1, w, 1)
    kpos = blk * w + (torch.arange(K, device=dev).view(1, 1, K) - w)
    invalid = (kpos < 0) | (kpos >= N)
    if causal:
        invalid = invalid | (kpos > qpos)
    scores = (qb * (D ** -0.5)) @ keys.transpose(-1, -2)
    invalid = invalid.expand(nb, w, K).reshape(1, 1, nb, w, K)
    scores = scores.masked_fill(invalid, float("-inf"))
    out = torch.softmax(scores, dim=-1) @ vals
    return out.reshape(B, H, nb * w, D)[:, :, :N]


def local_window_decode(q, k, v, window=256):
    """Queries sit at the end of the sequence. They see the last two key blocks
    (the final block and the one before it), matching local_window_attention
    (non-causal) for a query in the final block."""
    N = k.shape[-2]
    nb = -(-N // window)
    start = max(nb - 2, 0) * window
    return softmax_attention(q, k[..., start:, :], v[..., start:, :])


def phi(x):
    return F.relu(x) + 1.0


def linear_attention(q, k, v, causal=False, acc_dtype=None, chunk=256, eps=1e-6):
    out_dtype = q.dtype
    qf, kf, vv = phi(q), phi(k), v
    if acc_dtype is not None:
        qf, kf, vv = qf.to(acc_dtype), kf.to(acc_dtype), vv.to(acc_dtype)
    if not causal:
        kv = kf.transpose(-1, -2) @ vv                  # [B,H,D,Dv]
        z = kf.sum(-2).unsqueeze(-1)                    # [B,H,D,1]
        out = (qf @ kv) / ((qf @ z) + eps)
        return out.to(out_dtype)
    B, H, N, D = qf.shape
    Dv = vv.shape[-1]
    C = chunk
    nc = -(-N // C)
    pad = nc * C - N
    if pad:
        qf, kf, vv = (F.pad(x, (0, 0, 0, pad)) for x in (qf, kf, vv))
    qc = qf.reshape(B, H, nc, C, D)
    kc = kf.reshape(B, H, nc, C, D)
    vc = vv.reshape(B, H, nc, C, Dv)
    kv_c = kc.transpose(-1, -2) @ vc                    # [B,H,nc,D,Dv]
    z_c = kc.sum(-2).unsqueeze(-1)                      # [B,H,nc,D,1]
    kv_pre = torch.cat([torch.zeros_like(kv_c[:, :, :1]), kv_c.cumsum(2)[:, :, :-1]], 2)
    z_pre = torch.cat([torch.zeros_like(z_c[:, :, :1]), z_c.cumsum(2)[:, :, :-1]], 2)
    num = qc @ kv_pre
    den = qc @ z_pre
    A = qc @ kc.transpose(-1, -2)
    tri = torch.ones(C, C, dtype=torch.bool, device=A.device).tril()
    A = A * tri
    num = num + A @ vc
    den = den + A.sum(-1, keepdim=True)
    out = (num / (den + eps)).reshape(B, H, nc * C, Dv)[:, :, :N]
    return out.to(out_dtype)
