"""Probe-first rank routing, SDPA exact fallback, and explicit cached assignments.
The old router is retained unchanged for comparisons. Cached routing never means
accuracy is guaranteed on changed inputs. Selection and execution are separable.
"""
import torch
from .adaptive import omega, features, exact_rows, rf_attention


def sampled_rf(q, k, v, w, idx, causal=False, chunk=256):
    """Evaluate only requested rows; causal uses prefix sufficient statistics."""
    kf = features(k, w, True)
    qf = features(q[..., idx, :], w, False)
    if not causal:
        kv = kf.transpose(-1, -2) @ v
        z = kf.sum(-2).unsqueeze(-1)
        return (qf @ kv) / (qf @ z)
    st = q.new_zeros(*q.shape[:-2], w.shape[0], v.shape[-1])
    zs = q.new_zeros(*q.shape[:-2], w.shape[0], 1)
    out = torch.empty_like(q[..., idx, :])
    for s in range(0, k.shape[-2], chunk):
        stop = min(s+chunk, k.shape[-2])
        kc, vc = kf[..., s:stop, :], v[..., s:stop, :]
        mask = (idx >= s) & (idx < stop)
        if mask.any():
            a = qf[..., mask, :] @ kc.transpose(-1,-2)
            a = a.masked_fill(torch.arange(s,stop,device=q.device)[None,:] > idx[mask,None], 0)
            out[...,mask,:] = (qf[...,mask,:] @ st + a @ vc) / (qf[...,mask,:] @ zs + a.sum(-1,keepdim=True))
        st += kc.transpose(-1,-2) @ vc
        zs += kc.sum(-2).unsqueeze(-1)
    return out


def choose(q,k,v,tol=.1,n_probe=16,causal=False,seed=0,ranks=(64,256,1024)):
    B,H,N,D=q.shape
    g=torch.Generator().manual_seed(seed)
    idx=torch.randperm(N,generator=g)[:min(n_probe,N)].sort().values.to(q.device)
    q,k,v=(x.float().reshape(B*H,1,x.shape[-2],D) for x in (q,k,v))
    exact=exact_rows(q[...,idx,:],k,v,idx if causal else None)
    assignment=torch.zeros(B*H,dtype=torch.long,device=q.device)
    errors=torch.full((B*H,len(ranks)),float('nan'),device=q.device)
    for j,m in enumerate(ranks):
        pending=torch.where(assignment==0)[0]
        if not len(pending):break
        group=max(1,2**26//(k.shape[-2]*m))
        for st in range(0,len(pending),group):
            sel=pending[st:st+group]
            candidate=sampled_rf(q[sel],k[sel],v[sel],omega(m,D,q.device,seed),idx,causal)
            err=((candidate-exact[sel]).norm(dim=-1)/exact[sel].norm(dim=-1).clamp_min(1e-30)).mean((-1,-2))
            errors[sel,j]=err
            assignment[sel[err<=tol]]=m
    return assignment.reshape(B,H),errors.reshape(B,H,len(ranks)),idx


def execute(q,k,v,assignment,causal=False,seed=0,chunk=256):
    B,H,N,D=q.shape
    qr,kr,vr=(x.reshape(B*H,1,x.shape[-2],D) for x in (q,k,v))
    out=torch.empty_like(qr)
    a=assignment.reshape(-1)
    for m in (0,64,256,1024):
        sel=torch.where(a==m)[0]
        if not len(sel): continue
        if m==0:
            out[sel]=torch.nn.functional.scaled_dot_product_attention(qr[sel],kr[sel],vr[sel],is_causal=causal)
        else:
            group=max(1,2**26//(N*m))
            for s in range(0,len(sel),group):
                ix=sel[s:s+group]
                z,_=rf_attention(qr[ix].float(),kr[ix].float(),vr[ix].float(),omega(m,D,q.device,seed),causal,chunk)
                out[ix]=z.to(q.dtype)
    return out.reshape(B,H,N,D)


def route(q,k,v,tol=.1,n_probe=16,causal=False,seed=0):
    a,_,_=choose(q,k,v,tol,n_probe,causal,seed)
    return execute(q,k,v,a,causal,seed),a
