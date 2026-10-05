import torch
from lab import adaptive as R, attention as A

torch.manual_seed(0)


def test_causal_rf_identity():
    q, k, v = [torch.randn(3, 1, 300, 16) for _ in range(3)]
    w = R.omega(64, 16, "cpu")
    o, _ = R.rf_attention(q, k, v, w, True, chunk=64)
    qf, kf = R.features(q, w, False), R.features(k, w, True)
    s = (qf @ kf.transpose(-1, -2)).tril()
    assert torch.allclose(o, s @ v / s.sum(-1, keepdim=True), atol=1e-4)


def test_rf_error_decreases_with_rank():
    q, k, v = [torch.randn(1, 2, 200, 16) * s for s in (0.5, 0.5, 1)]
    ex = A.sdpa_attention(q, k, v)
    errs = []
    for m in (64, 256, 1024, 4096):
        o, _ = R.rf_attention(*(x.reshape(2, 1, 200, 16) for x in (q, k, v)), R.omega(m, 16, "cpu"))
        errs.append(((o.reshape(ex.shape) - ex).norm() / ex.norm()).item())
    assert errs[0] > errs[1] > errs[2] > errs[3]


def test_router_adapts_and_falls_back():
    sc = torch.tensor([0.3, 4.0]).view(1, 2, 1, 1)
    q, k, v = [torch.randn(1, 2, 256, 64) for _ in range(3)]
    q, k = q * sc, k * sc
    o, info = R.adaptive_rank_attention(q, k, v, tol=0.2, fallback=True)
    assert info["rank"][0, 0] in (64, 256, 1024) and info["met"][0, 0]   # smooth head: low rank accepted
    assert info["rank"][0, 1] == 0 and not info["met"][0, 1]            # sharp head: no rank works -> exact
    ex = A.sdpa_attention(q, k, v)
    assert torch.allclose(o[:, 1], ex[:, 1], atol=1e-5)
    o2, info2 = R.adaptive_rank_attention(q, k, v, tol=0.2, fallback=False)
    assert info2["rank"][0, 1] == 1024 and not info2["met"][0, 1]


def test_router_causal_fallback_matches_exact():
    q, k, v = [torch.randn(1, 2, 128, 32) * 4 for _ in range(3)]
    o, info = R.adaptive_rank_attention(q, k, v, causal=True, tol=0.05, fallback=True)
    assert (info["rank"] == 0).all()
    assert torch.allclose(o, A.sdpa_attention(q, k, v, True), atol=1e-4)
