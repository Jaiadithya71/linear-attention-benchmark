"""Prespecified follow-up. GPU only. Every result is produced from actual tensors.
Run --parts sensitivity scaling qwen. Partial CSVs survive failure; manifest marks completion.
"""
import argparse,csv,json,math,os,time,contextlib,hashlib,subprocess
from pathlib import Path
import torch
from lab import adaptive as R
from lab import router_detailed as U
from lab.env import require_cuda,gather,utcnow,write_config
from lab.timing import time_cuda,measure_memory,is_oom

SCALES=(.1,.2,.25,.3,.4,.5,.75,1.)
REGIMES=tuple('scale_'+str(s) for s in SCALES)+('mix_easy','mix_broad','lowrank_easy','lowrank_unit','cluster_easy','cluster_unit','local_global_easy','local_global_unit','rare_sharp','rare_sharp_shift')
TOLS=(.01,.025,.05,.075,.1,.15,.2,.3,.4)
PROBES=(4,8,16,32,64,128)
SEEDS=(1234,2345,3456)
SCALE_REGIMES=('scale_0.25','scale_0.4','mix_easy','lowrank_easy','local_global_easy','rare_sharp')
LAYERS=(0,6,12,18,23)

class Writer:
    def __init__(self,path): self.path=path; self.rows=[]
    def add(self,**row): self.rows.append(row)
    def flush(self):
        if not self.rows:return
        keys=list(dict.fromkeys(k for r in self.rows for k in r))
        with open(self.path,'w',newline='') as f:
            w=csv.DictWriter(f,keys);w.writeheader();w.writerows(self.rows)
    def close(self): self.flush()


def inputs(regime,N,seed,dt,H=8,D=64):
    g=torch.Generator(device='cuda').manual_seed(seed)
    rand=lambda *shape:torch.randn(*shape,generator=g,device='cuda')
    q,k,v=[rand(1,H,N,D) for _ in range(3)]
    if regime.startswith('scale_'): q*=float(regime[6:]);k*=float(regime[6:])
    elif regime.startswith('mix_'):
        ss=[.1,.15,.2,.25,.3,.35,.4,.5] if regime=='mix_easy' else [.1,.2,.3,.4,.5,.6,.8,1.]
        s=torch.tensor(ss,device='cuda')[None,:,None,None];q*=s;k*=s
    elif regime.startswith('lowrank_'):
        basis=rand(H,4,D)/2
        q=rand(1,H,N,4)@basis;k=rand(1,H,N,4)@basis
        if regime.endswith('easy'):q*=.25;k*=.25
    elif regime.startswith('cluster_'):
        centers=rand(H,8,D)
        ids=torch.arange(N,device='cuda')%8
        q=centers[:,ids][None]+.1*q;k=centers[:,ids][None]+.1*k
        if regime.endswith('easy'):q*=.25;k*=.25
    elif regime.startswith('local_global_'):
        # Smooth positional Q/K plus a shared global component; dense softmax mask unchanged.
        pos=torch.arange(N,device='cuda').float()[:,None]
        freqs=torch.arange(1,D//2+1,device='cuda').float()[None,:]
        z=torch.cat([torch.sin(pos*freqs*2*math.pi/N),torch.cos(pos*freqs*2*math.pi/N)],-1)
        q=.1*q+z[None,None]+.2*rand(1,H,1,D)
        k=.1*k+z[None,None]+.2*rand(1,H,1,D)
        if regime.endswith('easy'):q*=.25;k*=.25
    elif regime.startswith('rare_sharp'):
        q*=.15;k*=.15
        # 1% sharp query rows, deterministic locations. Shift variant changes their positions.
        ix=torch.arange(0,N,100,device='cuda')
        if regime.endswith('shift'):ix=(ix+50)%N
        q[:,:,ix]=k[:,:,ix]*1200
        # Keep existing V distribution; sharp rows have distinct output difficulty.
    else:raise ValueError(regime)
    return q.to(dt),k.to(dt),v.to(dt)


def audit(q,k,v,causal,seed,sample=256):
    B,H,N,D=q.shape
    # Audit and routing probes use disjoint rows, fixed by separate seed.
    g=torch.Generator().manual_seed(seed)
    perm=torch.randperm(N,generator=g).to(q.device)
    p=perm[:min(128,N//2)].sort().values
    a=perm[min(128,N//2):min(128,N//2)+min(sample,N//2)].sort().values
    allidx=torch.unique(torch.cat([p,a]),sorted=True)
    qr,kr,vr=(x.float().reshape(B*H,1,N,D) for x in (q,k,v))
    exact=R.exact_rows(qr[...,allidx,:],kr,vr,allidx if causal else None)
    errors=[];curves=[]
    for m in R.RANKS:
        rr=[]
        group=max(1,2**26//(N*m))
        for st in range(0,B*H,group):
            zz=U.sampled_rf(qr[st:st+group],kr[st:st+group],vr[st:st+group],R.omega(m,D,q.device,seed),allidx,causal)
            rel=(zz-exact[st:st+group]).norm(dim=-1)/exact[st:st+group].norm(dim=-1).clamp_min(1e-30)
            rr.append(rel[:,0])
        errors.append(torch.cat(rr,0))
    return torch.stack(errors,-1),allidx,p,a


def sensitivity(args,out):
    w=Writer(out/'sensitivity_heads.csv');c=Writer(out/'rank_error_curves.csv')
    N=4096
    for regime in REGIMES:
      for causal in (False,True):
       for dt in (torch.float16,torch.float32):
        for seed in SEEDS:
            q,k,v=inputs(regime,N,seed,dt)
            er,ix,p,a=audit(q,k,v,causal,seed)
            ai=torch.searchsorted(ix,a)
            ae=er[:,ai]
            for h in range(8):
                for j,m in enumerate(R.RANKS):
                    vals=ae[h,:,j]
                    c.add(source='synthetic',regime=regime,N=N,causal=causal,dtype=str(dt),seed=seed,head=h,rank=m,audit_rows=len(a),mean_relative_l2=vals.mean().item(),p95_relative_l2=vals.quantile(.95).item(),max_relative_l2=vals.max().item(),nonfinite_rows=(~torch.isfinite(vals)).sum().item())
            for np in PROBES:
                # Reproducible nested probes from the same seeded permutation, not sorted-prefix bias.
                g=torch.Generator().manual_seed(seed)
                pp=torch.randperm(N,generator=g)[:np].to(q.device)
                pe=er[:,torch.searchsorted(ix,pp)].mean(1)
                for tol in TOLS:
                    rank=torch.zeros(8,dtype=torch.long,device=q.device)
                    for j,m in enumerate(R.RANKS):rank[(rank==0)&(pe[:,j]<=tol)]=m
                    for h in range(8):
                        m=rank[h].item();j=R.RANKS.index(m) if m else None
                        mean=ae[h,:,j].mean().item() if m else 0.
                        p95=ae[h,:,j].quantile(.95).item() if m else 0.
                        w.add(regime=regime,N=N,causal=causal,dtype=str(dt),seed=seed,head=h,tol=tol,n_probe=np,rank=m,probe_mean_relative_l2=pe[h,j].item() if m else '',audit_mean_relative_l2=mean,audit_p95_relative_l2=p95,false_accept_mean=int(bool(m and (not math.isfinite(mean) or mean>tol))),false_accept_p95=int(bool(m and (not math.isfinite(p95) or p95>tol))))
            del q,k,v,er
        w.flush();c.flush()
        print('SENSITIVITY',regime,causal,str(dt),flush=True)


def benchmark(w,base,name,fn,args,quality=None):
    row=dict(base,method=name)
    try:
        z=fn();zz=z[0] if isinstance(z,tuple) else z
        row['finite_output']=bool(torch.isfinite(zz).all().item())
        del z,zz
        # CUDA events include Python dispatch gaps between GPU launches; wall clock also recorded.
        row.update(time_cuda(fn,warmup=1,reps=args.reps,slow_ms=1000,slow_reps=3))
        t=time.perf_counter();fn();torch.cuda.synchronize();row['wall_call_ms']=(time.perf_counter()-t)*1000
        extra,peak=measure_memory(fn);row['peak_extra_mb']=extra/2**20;row['peak_total_mb']=peak/2**20
        row['status']='ok'
        if quality:row.update(quality())
    except torch.cuda.OutOfMemoryError:
        row['status']='oom';torch.cuda.empty_cache()
    w.add(**row);w.flush();print('TIMING',row,flush=True)


def scaling(args,out):
    w=Writer(out/'scaling.csv');a=Writer(out/'scaling_heads.csv')
    for regime in SCALE_REGIMES:
      for causal in (False,True):
       for dt in (torch.float16,torch.float32):
        for N in args.Ns:
            seed=1234;q,k,v=inputs(regime,N,seed,dt)
            base=dict(regime=regime,N=N,causal=causal,dtype=str(dt),seed=seed,tol=.1,n_probe=16)
            assignment,pe,_=U.choose(q,k,v,.1,16,causal,seed)
            er,ix,pp,aa=audit(q,k,v,causal,seed)
            for h,m in enumerate(assignment.flatten().tolist()):
                vals=er[h,torch.searchsorted(ix,aa),R.RANKS.index(m)] if m else torch.zeros(1,device=q.device)
                a.add(**base,head=h,rank=m,audit_mean_relative_l2=vals.mean().item(),audit_p95_relative_l2=vals.quantile(.95).item())
            a.flush()
            if N<=4096:
                exact=torch.nn.functional.scaled_dot_product_attention(q,k,v,is_causal=causal).float()
                candidate=U.execute(q,k,v,assignment,causal,seed).float()
                rel=(candidate-exact).norm(dim=-1)/exact.norm(dim=-1).clamp_min(1e-30)
                for h,m in enumerate(assignment.flatten().tolist()):
                    vals=rel[0,h]
                    a.add(**base,head=h,rank=m,validation_scope='all_query_rows',full_mean_relative_l2=vals.mean().item(),full_p95_relative_l2=vals.quantile(.95).item(),full_max_relative_l2=vals.max().item(),full_rows_above_tol=(vals>.1).sum().item(),full_nonfinite_rows=(~torch.isfinite(vals)).sum().item())
                a.flush();del exact,candidate,rel
            calls={
                'sdpa':lambda:torch.nn.functional.scaled_dot_product_attention(q,k,v,is_causal=causal),
                'probe_first':lambda:U.route(q,k,v,.1,16,causal,seed),
                'cached_same':lambda:U.execute(q,k,v,assignment,causal,seed),
                'selection_only':lambda:U.choose(q,k,v,.1,16,causal,seed),
                'original_router':lambda:R.adaptive_rank_attention(q,k,v,causal=causal,tol=.1,n_probe=16,fallback=True,seed=seed)
            }
            # Actual repeated calls, selection paid once, no arithmetic latency estimate.
            for repeats in (2,4,8):
                def amortized(repeats=repeats):
                    aa,_,_=U.choose(q,k,v,.1,16,causal,seed)
                    for _ in range(repeats):zz=U.execute(q,k,v,aa,causal,seed)
                    return zz
                calls['choose_once_'+str(repeats)+'_calls']=amortized
            # selection_only returns ranks/error/indices, not an attention tensor.
            for name,fn in calls.items():benchmark(w,base,name,fn,args)
            for m in R.RANKS:
                fixed=torch.full_like(assignment,m)
                benchmark(w,base,'rf'+str(m),lambda f=fixed:U.execute(q,k,v,f,causal,seed),args)
            # Rank cache reuse is tested on a different random draw and a forced regime shift.
            for target in (regime,'scale_1.0','rare_sharp_shift'):
                q2,k2,v2=inputs(target,N,2345,dt)
                e2,i2,_,a2=audit(q2,k2,v2,causal,seed)
                for h,m in enumerate(assignment.flatten().tolist()):
                    vals=e2[h,torch.searchsorted(i2,a2),R.RANKS.index(m)] if m else torch.zeros(1,device=q.device)
                    a.add(**base,head=h,rank=m,cache_target=target,audit_mean_relative_l2=vals.mean().item(),audit_p95_relative_l2=vals.quantile(.95).item())
                del q2,k2,v2,e2
            a.flush();del q,k,v,er;torch.cuda.empty_cache()


@contextlib.contextmanager
def model_patch(mode,tol=.1,cache=None,stats=None,capture=None):
    from transformers import AttentionInterface
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    from lab.qwen_patch import _expand_kv
    old=ALL_ATTENTION_FUNCTIONS['sdpa']
    def fn(module,q,k,v,attention_mask,scaling=None,dropout=0.,**kwargs):
        if attention_mask is not None:raise RuntimeError('Expected unpadded causal prefill')
        q,k,v=_expand_kv(q,k,v)
        li=module.layer_idx
        if capture is not None and li in LAYERS:capture[li]=(q.detach().clone(),k.detach().clone(),v.detach().clone())
        if mode in ('baseline','capture'):
            z=torch.nn.functional.scaled_dot_product_attention(q,k,v,is_causal=True)
        elif mode=='original':
            z,info=R.adaptive_rank_attention(q,k,v,causal=True,tol=tol,fallback=True,seed=1234)
            chosen=info['rank']
        else:
            if mode=='cached':chosen=cache[li].to(q.device)
            else:
                chosen,_,_=U.choose(q,k,v,tol,16,True,1234)
                if mode=='calibrate':
                    if li not in cache:cache[li]=chosen.detach().clone()
                    else:
                        # Require the SAME rank to pass on both calibration windows.
                        cache[li][cache[li]!=chosen]=0
            z=U.execute(q,k,v,chosen,True,1234)
        if stats is not None and mode not in ('baseline','capture'):
            for h,m in enumerate(chosen.flatten().tolist()):stats.append(dict(layer=li,head=h,rank=m))
        return z.transpose(1,2).contiguous(),None
    AttentionInterface.register('sdpa',fn)
    try:yield
    finally:AttentionInterface.register('sdpa',old)


def qwen(args,out):
    from transformers import AutoModelForCausalLM,AutoTokenizer
    from datasets import load_dataset
    model_id='Qwen/Qwen2.5-0.5B'
    tok=AutoTokenizer.from_pretrained(model_id)
    model=AutoModelForCausalLM.from_pretrained(model_id,torch_dtype=torch.float16,attn_implementation='sdpa').cuda().eval()
    text='\n\n'.join(load_dataset('Salesforce/wikitext','wikitext-2-raw-v1',split='test')['text'])
    ids=tok(text,return_tensors='pt').input_ids[0]
    ctx=2048;windows=16;offset=2*ctx
    assert len(ids)>offset+windows*ctx
    metadata=dict(model=model_id,model_revision=getattr(model.config,'_commit_hash',None),corpus='Salesforce/wikitext:wikitext-2-raw-v1:test',corpus_token_sha256=hashlib.sha256(ids.numpy().tobytes()).hexdigest(),calibration_windows=2,evaluation_windows=windows,ctx=ctx,eval_start_token=offset)
    (out/'qwen_data_manifest.json').write_text(json.dumps(metadata,indent=2))
    heads=Writer(out/'qwen_activation_heads.csv');curves=Writer(out/'qwen_activation_curves.csv');counts=Writer(out/'qwen_choices.csv');quality=Writer(out/'qwen_quality.csv');lat=Writer(out/'qwen_prefill.csv')
    with torch.no_grad():
      # Activations captured from original SDPA, after RoPE and KV expansion, not random tensors.
      for win in range(4):
        capture={}
        with model_patch('capture',capture=capture):model.model(input_ids=ids[win*ctx:(win+1)*ctx][None].cuda(),use_cache=False)
        for layer,(q,k,v) in capture.items():
            er,ix,pp,aa=audit(q,k,v,True,1234)
            ae=er[:,torch.searchsorted(ix,aa)]
            for h in range(q.shape[1]):
                for j,m in enumerate(R.RANKS):
                    vals=ae[h,:,j]
                    curves.add(window=win,layer=layer,head=h,rank=m,audit_rows=len(aa),mean_relative_l2=vals.mean().item(),p95_relative_l2=vals.quantile(.95).item(),max_relative_l2=vals.max().item())
            for np in PROBES:
                pp=torch.randperm(ctx,generator=torch.Generator().manual_seed(1234))[:np].to(q.device)
                pe=er[:,torch.searchsorted(ix,pp)].mean(1)
                for tol in TOLS:
                    a=torch.zeros(q.shape[1],dtype=torch.long,device=q.device)
                    for j,m in enumerate(R.RANKS):a[(a==0)&(pe[:,j]<=tol)]=m
                    for h,m in enumerate(a.tolist()):
                        vals=ae[h,:,R.RANKS.index(m)] if m else torch.zeros(1,device=q.device)
                        mean=vals.mean().item()
                        heads.add(window=win,layer=layer,head=h,n_probe=np,tol=tol,rank=m,audit_mean_relative_l2=mean,audit_p95_relative_l2=vals.quantile(.95).item(),false_accept_mean=int(bool(m and (not math.isfinite(mean) or mean>tol))))
        heads.flush();curves.flush();del capture;torch.cuda.empty_cache();print('QWEN CAPTURE',win,flush=True)
      variants=[('baseline',.1),('original',.1)]+[(m,t) for m in ('oneshot','cached') for t in (.1,.2,.4)]
      for mode,tol in variants:
        cache={}
        if mode=='cached':
            with model_patch('calibrate',tol,cache):
                for win in range(2):model.model(input_ids=ids[win*ctx:(win+1)*ctx][None].cuda(),use_cache=False)
            (out/f'cache_tol_{tol}.json').write_text(json.dumps({str(k):v.tolist() for k,v in cache.items()},indent=2))
        total=0.;n=0;bad=0
        with model_patch(mode,tol,cache):
            for win in range(windows):
                chunk=ids[offset+win*ctx:offset+(win+1)*ctx][None].cuda()
                # Measure actual forward NLL; record per-window losses as well as aggregate.
                logits=model(input_ids=chunk,use_cache=False).logits[0,:-1].float()
                loss=torch.nn.functional.cross_entropy(logits,chunk[0,1:],reduction='sum').item()
                ok=math.isfinite(loss)
                quality.add(mode=mode,tol=tol,window=win,ctx=ctx,tokens_scored=ctx-1,nll_sum=loss,status='ok' if ok else 'nonfinite')
                if ok:total+=loss;n+=ctx-1
                else:bad+=1
        nl=total/n if n else float('nan')
        quality.add(mode=mode,tol=tol,window='aggregate',ctx=ctx,windows=windows,tokens_scored=n,nonfinite_windows=bad,nll_sum=total,nll_per_token=nl,perplexity=math.exp(nl) if n and nl<700 else float('nan'),status='ok' if bad==0 else 'nonfinite')
        quality.flush();print('QWEN PPL',mode,tol,nl,math.exp(nl) if n and nl<700 else None,flush=True)
        for N in (1024,2048,4096,8192):
            chunk=ids[offset:offset+N][None].cuda();stats=[]
            with model_patch(mode,tol,cache,stats=stats):model.model(input_ids=chunk,use_cache=False)
            for row in stats:counts.add(mode=mode,tol=tol,N=N,**row)
            counts.flush()
            with model_patch(mode,tol,cache):
                benchmark(lat,dict(mode=mode,tol=tol,N=N,scope='transformer_hidden_state_prefill'),mode,lambda:model.model(input_ids=chunk,use_cache=False).last_hidden_state,args)
    del model;torch.cuda.empty_cache()


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',default='results/router_detailed');p.add_argument('--parts',nargs='+',default=['sensitivity','scaling','qwen']);p.add_argument('--reps',type=int,default=5);p.add_argument('--Ns',type=int,nargs='+',default=[512,1024,2048,4096,8192,16384,32768,65536]);args=p.parse_args()
    require_cuda('T4');out=Path(args.out);out.mkdir(parents=True,exist_ok=True);started=utcnow()
    # Record manifest BEFORE measurements; source hashes make dirty-tree runs reproducible.
    cfg=gather('router_detailed',started,None,args)
    cfg.update(regimes=REGIMES,tols=TOLS,probes=PROBES,seeds=SEEDS,scaling_regimes=SCALE_REGIMES,qwen_layers=LAYERS,post_result_changes=[],source_sha256={str(x):hashlib.sha256(x.read_bytes()).hexdigest() for x in [Path(__file__),Path('lab/router_detailed.py'),Path('lab/adaptive.py')]})
    write_config(out/'manifest.json',cfg)
    with torch.no_grad():
      for part in args.parts:
        globals()[part](args,out)
        cfg.setdefault('completed_parts',[]).append(part);write_config(out/'manifest.json',cfg)
    cfg['finished_utc']=utcnow();write_config(out/'manifest.json',cfg)

if __name__=='__main__':main()
