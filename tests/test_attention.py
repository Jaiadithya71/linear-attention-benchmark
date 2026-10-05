"""CPU correctness tests of the math (not benchmarks). Run: pytest -q tests"""
import math
import pytest
import torch
from lab import attention as A
from lab.stats import wilson_interval

torch.manual_seed(0)


def rand(n=300, h=2, d=16, b=2):
    return [torch.randn(b, h, n, d, dtype=torch.float64) for _ in range(3)]


def test_softmax_matches_sdpa():
    q, k, v = rand()
    for c in (False, True):
        assert torch.allclose(A.softmax_attention(q, k, v, c), A.sdpa_attention(q, k, v, c), atol=1e-9)


def ref_local(q, k, v, w, causal):
    N = q.shape[2]
    i = torch.arange(N)
    bi = (i // w)[:, None]; bj = (i // w)[None]
    vis = (bj == bi) | (bj == bi - 1) | ((bj == bi + 1) if not causal else torch.zeros_like(bj, dtype=torch.bool))
    if causal:
        vis = vis & (i[None] <= i[:, None])
    s = (q @ k.transpose(-1, -2)) * q.shape[-1] ** -0.5
    return torch.softmax(s.masked_fill(~vis, float("-inf")), -1) @ v


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("n", [300, 256, 100])
def test_local_window(causal, n):
    q, k, v = rand(n)
    assert torch.allclose(A.local_window_attention(q, k, v, 64, causal), ref_local(q, k, v, 64, causal), atol=1e-9)


def test_local_decode_matches_last_row():
    q, k, v = rand(300)
    full = A.local_window_attention(q, k, v, 64, False)[:, :, -1:]
    dec = A.local_window_decode(q[:, :, -1:], k, v, 64)
    assert torch.allclose(full, dec, atol=1e-9)


def ref_linear(q, k, v, causal):
    qf, kf = A.phi(q), A.phi(k)
    s = qf @ kf.transpose(-1, -2)
    if causal:
        s = s * torch.ones(s.shape[-2:], dtype=s.dtype).tril()
    return (s @ v) / s.sum(-1, keepdim=True)


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("n", [300, 256])
def test_linear(causal, n):
    q, k, v = rand(n)
    out = A.linear_attention(q, k, v, causal=causal, chunk=64, eps=0.0)
    assert torch.allclose(out, ref_linear(q, k, v, causal), atol=1e-9)


def test_wilson_known_values():
    lo, hi = wilson_interval(8, 10)
    assert abs(lo - 0.4902) < 1e-3 and abs(hi - 0.9433) < 1e-3
    from scipy.stats import binomtest
    ci = binomtest(37, 120).proportion_ci(method="wilson")
    lo, hi = wilson_interval(37, 120)
    assert abs(lo - ci.low) < 1e-9 and abs(hi - ci.high) < 1e-9


def test_qwen_patch_plumbing_tiny_random_model():
    """Checks that patched functions are actually invoked and restored (random tiny model, plumbing only)."""
    pytest.importorskip("transformers")
    from transformers import Qwen2Config, Qwen2ForCausalLM
    from lab.qwen_patch import patched_attention, make_fn
    import lab.qwen_patch as qp
    cfg = Qwen2Config(vocab_size=100, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=512)
    m = Qwen2ForCausalLM._from_config(cfg, attn_implementation="sdpa").eval()
    ids = torch.randint(0, 100, (1, 300))
    calls = []
    orig = qp.make_fn
    def counting(kind, window=256, tol=0.1):
        f = orig(kind, window, tol)
        def g(*a, **k):
            calls.append(kind); return f(*a, **k)
        return g
    qp.make_fn = counting
    with torch.no_grad():
        base = m.model(input_ids=ids).last_hidden_state
        for kind in ("local_window", "linear"):
            with patched_attention(kind, 64):
                out = m.model(input_ids=ids).last_hidden_state
            assert out.shape == base.shape and torch.isfinite(out).all()
        # window >= N makes local attention equal full causal attention
        with patched_attention("local_window", 512):
            out = m.model(input_ids=ids).last_hidden_state
        again = m.model(input_ids=ids).last_hidden_state
    qp.make_fn = orig
    assert calls.count("local_window") == 4 and calls.count("linear") == 2
    assert torch.allclose(out, base, atol=1e-4)
    assert torch.allclose(again, base, atol=1e-6)  # restored
