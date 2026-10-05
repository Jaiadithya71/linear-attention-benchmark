"""Recompute analysis and figures from measured router CSVs. No fabricated data."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path('results/router_detailed')
FIG=Path('figures/router_detailed');FIG.mkdir(parents=True,exist_ok=True)
s=pd.read_csv(ROOT/'sensitivity_heads.csv')
t=pd.read_csv(ROOT/'scaling.csv')
h=pd.read_csv(ROOT/'scaling_heads.csv')
plt.rcParams.update({'font.size':10,'axes.titlesize':12,'figure.facecolor':'white','axes.spines.top':False,'axes.spines.right':False})
colors={'probe_first':'#116466','cached_same':'#d88922','original_router':'#617394','choose_once_8_calls':'#aa4263'}
methods={'probe_first':'Probe-first, one-shot','cached_same':'Cached, same input','original_router':'Original router','choose_once_8_calls':'Choose once, 8 calls (per call)'}
# Per-regime crossover curves, never treat failing runs as wins.
b=t.pivot(index=['regime','N','causal','dtype'],columns='method',values='median_ms')
fig,axs=plt.subplots(2,3,figsize=(15,8),sharex=True,sharey=True)
for ax,regime in zip(axs.flat,t.regime.unique()):
 z=b.loc[regime]
 for m,label in methods.items():
  ratio=z.sdpa/z[m]*(8 if m=='choose_once_8_calls' else 1)
  agg=ratio.groupby('N').median();ax.plot(agg.index,agg.values,'o-',label=label,color=colors[m],ms=4)
 ax.axhline(1,color='#777',ls=':',lw=1);ax.set(xscale='log',yscale='log',title=regime.replace('_',' '));ax.grid(alpha=.15)
 ax.set_xticks([512,4096,32768,65536],labels=['512','4k','32k','65k'])
for ax in axs[:,0]:ax.set_ylabel('SDPA time / method time (higher is faster)')
for ax in axs[-1]:ax.set_xlabel('Sequence length')
handles,labels=axs[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=4,bbox_to_anchor=(.5,.012),frameon=False)
fig.suptitle('Synthetic crossover: measured T4 latency, tolerance 0.1, 16 probes',fontsize=17)
fig.text(.5,.92,'Each point is the median of causal/noncausal x fp16/fp32. Cached input is idealized; original OOMs excluded.',ha='center',fontsize=10)
fig.tight_layout(rect=[0,.06,1,.88]);fig.savefig(FIG/'synthetic_crossover.png',dpi=150);plt.close(fig)
# Probe acceptance versus held-out errors at the prespecified default tolerance.
s['linear']=s['rank']>0
x=s[s.tol==.1];g=x.groupby('n_probe').agg(linear_fraction=('linear','mean'))
l=x[x.linear].groupby('n_probe')[['false_accept_mean','false_accept_p95']].mean()
fig,axs=plt.subplots(1,2,figsize=(12,4.8))
axs[0].plot(g.index,g.linear_fraction*100,'o-',color='#116466');axs[0].set(title='Linear acceptance rate',ylabel='% of all head cases',ylim=(0,100))
for col,label,color in [('false_accept_mean','Mean > tolerance','#aa4263'),('false_accept_p95','p95 > tolerance','#d88922')]:
 axs[1].plot(l.index,l[col]*100,'o-',label=label,color=color)
axs[1].set(title='Audit failure among accepted linear heads',ylabel='% of accepted linear cases',ylim=(0,100));axs[1].legend(frameon=False)
for ax in axs:ax.set(xscale='log',xlabel='Probe query count');ax.set_xticks([4,8,16,32,64,128],labels=['4','8','16','32','64','128']);ax.grid(alpha=.15)
fig.suptitle('Probe sensitivity: 18 regimes, 3 seeds, 2 dtypes, causal/noncausal',fontsize=15)
fig.text(.5,.905,'N=4096, tolerance=0.1. Audit queries are disjoint; p95 is not the routing criterion.',ha='center',fontsize=10)
fig.tight_layout(rect=[0,0,1,.875]);fig.savefig(FIG/'probe_audit_failures.png',dpi=150);plt.close(fig)
# Cached ranks on changed inputs.
c=h[h.cache_target.notna()&(h['rank']>0)].copy();c['group']=np.where(c.cache_target=='scale_1.0','Unit-scale drift',np.where(c.cache_target=='rare_sharp_shift','Shifted rare sharp','Same distribution, new draw'))
a=c.groupby('group').agg(n=('rank','size'),mean_fail=('audit_mean_relative_l2',lambda x:(x>.1).mean()),p95_fail=('audit_p95_relative_l2',lambda x:(x>.1).mean()))
fig,ax=plt.subplots(figsize=(10,4.7));pos=np.arange(len(a));ax.bar(pos-.18,a.mean_fail*100,.36,label='Mean > 0.1',color='#aa4263');ax.bar(pos+.18,a.p95_fail*100,.36,label='p95 > 0.1',color='#d88922')
ax.set_xticks(pos,labels=[f'{i}\nn={int(a.loc[i,"n"])}' for i in a.index]);ax.set(ylabel='% of reused linear heads',ylim=(0,105),title='Rank reuse on changed inputs: accuracy, not latency');ax.legend(frameon=False);ax.grid(axis='y',alpha=.15)
fig.tight_layout();fig.savefig(FIG/'cache_drift.png',dpi=150);plt.close(fig)
# All-query worst case, especially rare sharp rows hidden by averages.
f=h[h.validation_scope=='all_query_rows'].copy()
a=f.groupby('regime')[['full_mean_relative_l2','full_p95_relative_l2','full_max_relative_l2']].max()
fig,ax=plt.subplots(figsize=(11,5.2));pos=np.arange(len(a))
for i,(col,label,color) in enumerate([('full_mean_relative_l2','Worst head mean','#116466'),('full_p95_relative_l2','Worst head p95','#d88922'),('full_max_relative_l2','Worst query','#aa4263')]):ax.bar(pos+(i-1)*.24,a[col],.24,label=label,color=color)
ax.axhline(.1,ls=':',color='#777',label='Tolerance 0.1');ax.set_xticks(pos,labels=[r.replace('_','\n',1) for r in a.index]);ax.set(ylabel='Relative L2 error',title='All-query audits reveal errors that mean probes can miss');ax.legend(frameon=False,ncol=2);ax.grid(axis='y',alpha=.15)
fig.text(.5,.025,'Maximum over measured N=512..4096, causal/noncausal and fp16/fp32. Exact-fallback heads have zero error.',ha='center',fontsize=9)
fig.tight_layout(rect=[0,.05,1,1]);fig.savefig(FIG/'all_query_errors.png',dpi=150);plt.close(fig)
cache_stats=c.groupby('group').agg(n=('rank','size'),mean_fail=('audit_mean_relative_l2',lambda x:(x>.1).mean()),p95_fail=('audit_p95_relative_l2',lambda x:(x>.1).mean()))
summary={'measurement_rows':{'sensitivity':len(s),'timings':len(t),'scaling_heads':len(h)},'timing_status':t.status.value_counts().to_dict(),'tolerance_0_1_probe16':{},'cache_groups':cache_stats.to_dict()}
x=s[(s.tol==.1)&(s.n_probe==16)];q=x[x.linear];summary['tolerance_0_1_probe16']={'total_heads':len(x),'linear_heads':len(q),'audit_mean_failures':int(q.false_accept_mean.sum()),'audit_p95_failures':int(q.false_accept_p95.sum())}
summary['speedups']={}
for m in ['probe_first','cached_same','original_router','choose_once_2_calls','choose_once_4_calls','choose_once_8_calls']:
 repeats=int(m.split('_')[-2]) if m.startswith('choose_once') else 1
 z=repeats*b.sdpa/b[m];summary['speedups'][m]={'wins':int((z>1).sum()),'tested':int(z.notna().sum()),'median_by_N':{str(k):float(v) for k,v in z.groupby('N').median().items()}}
(ROOT/'analysis_synthetic.json').write_text(json.dumps(summary,indent=2))
print('Generated',','.join(str(p) for p in FIG.glob('*.png')))
QROOT=Path('results/router_detailed_qwen')
if (QROOT/'qwen_quality.csv').exists():
 q=pd.read_csv(QROOT/'qwen_quality.csv');q=q[q.perplexity.notna()].copy()
 lat=pd.read_csv(QROOT/'qwen_prefill.csv')
 fig,axs=plt.subplots(1,2,figsize=(13,5.5))
 names=q['mode']+' '+q['tol'].astype(str);base=q[q['mode']=='baseline'].perplexity.iloc[0]
 axs[0].barh(names,(q.perplexity/base-1)*100,color=['#116466']+['#aa4263']*7);axs[0].invert_yaxis();axs[0].set(xlabel='Perplexity change vs SDPA (%)',title='Matched 16-window quality')
 for mode,tol,label,color in [('baseline',.1,'SDPA','#116466'),('original',.1,'Original .1','#617394'),('oneshot',.1,'Probe-first .1','#aa4263'),('cached',.1,'Cached .1 (all exact)','#d88922')]:
  z=lat[(lat['mode']==mode)&(lat.tol==tol)];axs[1].plot(z.N,z.median_ms,'o-',label=label,color=color)
 axs[1].set(xscale='log',yscale='log',xlabel='Sequence length',ylabel='Median prefill latency (ms)',title='No measured model speedup');axs[1].set_xticks([1024,2048,4096,8192],labels=['1k','2k','4k','8k']);axs[1].xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter());axs[1].legend(frameon=False)
 for ax in axs:ax.grid(alpha=.15)
 fig.suptitle('Qwen2.5-0.5B on T4: tolerance .1 mostly falls back, speed not improved',fontsize=15)
 fig.text(.5,.04,'32,752 scored tokens, context 2048. Latency: transformer hidden-state prefill, not token generation.',ha='center',fontsize=10)
 fig.tight_layout(rect=[0,.09,1,.93]);fig.savefig(FIG/'qwen_quality_latency.png',dpi=150);plt.close(fig)
