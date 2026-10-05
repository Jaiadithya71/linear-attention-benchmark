"""Swap the attention function used by the real Qwen2.5 model (transformers AttentionInterface).
Patched functions assume causal, unpadded, single-sequence prefill and raise if a mask is passed."""
import contextlib
import torch
from . import attention as A
from . import adaptive as R

ROUTER_STATS = {}  # cumulative per-head rank choices made by the router during the current patched run


def _expand_kv(query, key, value):
    r = query.shape[1] // key.shape[1]
    if r > 1:
        key, value = key.repeat_interleave(r, dim=1), value.repeat_interleave(r, dim=1)
    return key, value


def make_fn(kind, window=256, tol=0.1):
    def fn(module, query, key, value, attention_mask, scaling=None, dropout=0.0, **kwargs):
        if attention_mask is not None:
            raise RuntimeError("patched attention expects attention_mask=None (causal, unpadded prefill)")
        key, value = _expand_kv(query, key, value)
        if kind == "sdpa_mha":
            out = A.sdpa_attention(query, key, value, causal=True)
        elif kind == "local_window":
            out = A.local_window_attention(query, key, value, window=window, causal=True)
        elif kind == "linear":
            out = A.linear_attention(query, key, value, causal=True, acc_dtype=torch.float32)
        elif kind in ("rf64", "rf256", "rf1024"):
            B, H, N, D = query.shape
            o, _ = R.rf_attention(*(x.float().reshape(B * H, 1, N, D) for x in (query, key, value)),
                                  R.omega(int(kind[2:]), D, query.device), causal=True)
            out = o.reshape(B, H, N, D).to(query.dtype)
        elif kind in ("adaptive_rank", "adaptive_rank_fb"):
            out, info = R.adaptive_rank_attention(query, key, value, causal=True, tol=tol, fallback=(kind == "adaptive_rank_fb"))
            r, met = info["rank"], info["met"]
            for name, val in (("heads_rank64", (r == 64).sum()), ("heads_rank256", (r == 256).sum()),
                              ("heads_rank1024", (r == 1024).sum()), ("heads_exact", (r == 0).sum()),
                              ("heads_tol_not_met", ((r != 0) & ~met).sum())):
                ROUTER_STATS[name] = ROUTER_STATS.get(name, 0) + int(val)
        else:
            raise ValueError(kind)
        return out.transpose(1, 2).contiguous(), None
    return fn


@contextlib.contextmanager
def patched_attention(kind, window=256, tol=0.1):
    """Temporarily replace the 'sdpa' attention function. kind in {'sdpa' (no-op), 'local_window', 'linear'}."""
    from transformers import AttentionInterface
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    original = ALL_ATTENTION_FUNCTIONS["sdpa"]
    if kind != "sdpa":  # "sdpa" = transformers' own default path, untouched
        AttentionInterface.register("sdpa", make_fn(kind, window, tol))
    try:
        yield
    finally:
        AttentionInterface.register("sdpa", original)
