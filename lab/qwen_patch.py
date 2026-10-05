"""Swap the attention function used by the real Qwen2.5 model (transformers AttentionInterface).
Patched functions assume causal, unpadded, single-sequence prefill and raise if a mask is passed."""
import contextlib
import torch
from . import attention as A


def _expand_kv(query, key, value):
    r = query.shape[1] // key.shape[1]
    if r > 1:
        key, value = key.repeat_interleave(r, dim=1), value.repeat_interleave(r, dim=1)
    return key, value


def make_fn(kind, window=256):
    def fn(module, query, key, value, attention_mask, scaling=None, dropout=0.0, **kwargs):
        if attention_mask is not None:
            raise RuntimeError("patched attention expects attention_mask=None (causal, unpadded prefill)")
        key, value = _expand_kv(query, key, value)
        if kind == "local_window":
            out = A.local_window_attention(query, key, value, window=window, causal=True)
        elif kind == "linear":
            out = A.linear_attention(query, key, value, causal=True, acc_dtype=torch.float32)
        else:
            raise ValueError(kind)
        return out.transpose(1, 2).contiguous(), None
    return fn


@contextlib.contextmanager
def patched_attention(kind, window=256):
    """Temporarily replace the 'sdpa' attention function. kind in {'sdpa' (no-op), 'local_window', 'linear'}."""
    from transformers import AttentionInterface
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    original = ALL_ATTENTION_FUNCTIONS["sdpa"]
    if kind != "sdpa":
        AttentionInterface.register("sdpa", make_fn(kind, window))
    try:
        yield
    finally:
        AttentionInterface.register("sdpa", original)
