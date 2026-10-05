import torch
from lab.adaptive import rf_attention,omega
from lab.router_detailed import sampled_rf,execute,choose

def test_sampled_matches_full():
    for causal in (False,True):
        torch.manual_seed(4)
        q,k,v=(torch.randn(2,1,43,8)*.2 for _ in range(3))
        ix=torch.tensor([0,4,15,16,17,42]);w=omega(64,8,'cpu')
        full,_=rf_attention(q,k,v,w,causal=causal,chunk=16)
        part=sampled_rf(q,k,v,w,ix,causal=causal,chunk=16)
        torch.testing.assert_close(part,full[...,ix,:],rtol=1e-4,atol=1e-6)

def test_exact_fallback():
    q,k,v=(torch.randn(1,2,19,8) for _ in range(3))
    for c in (False,True):
        actual=execute(q,k,v,torch.zeros(1,2,dtype=torch.long),c)
        expected=torch.nn.functional.scaled_dot_product_attention(q,k,v,is_causal=c)
        torch.testing.assert_close(actual,expected)

def test_zero_tolerance_selects_exact():
    q,k,v=(torch.randn(1,2,19,8) for _ in range(3))
    a,errors,_=choose(q,k,v,tol=0)
    assert (a==0).all() and errors.shape==(1,2,3)

def test_qwen_kv_expansion_contract():
    from lab.qwen_patch import _expand_kv
    q=torch.randn(1,4,11,8);k=torch.randn(1,2,11,8);v=torch.randn(1,2,11,8)
    k,v=_expand_kv(q,k,v)
    assert k.shape==v.shape==q.shape
