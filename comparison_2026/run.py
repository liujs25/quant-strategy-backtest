"""Retrospective common-risk comparison; no tuning and no overwrite."""
from pathlib import Path
import sys,json,hashlib,itertools
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import lightgbm as lgb
import joblib
from incremental.core import data,backtest,neutralize,FACTORS
from src.utils import predictive_metrics,portfolio_metrics
R=Path(__file__).resolve().parent;ROOT=R.parent
if (R/'results.json').exists():raise SystemExit('Completed comparison immutable; use separate replication directory')
panel,cols,returns=data(ROOT,final=True)
cutoff='2024-10-01';train=panel[(panel.label_end_20<cutoff)&panel.y_20.notna()].sort_values(['date','symbol']).reset_index(drop=True)
ev=panel[panel.date.between('2026-01-01','2026-11-20')&(panel.label_end_20<='2026-11-20')].sort_values(['date','symbol']).reset_index(drop=True)
original=pd.read_pickle(ROOT/'incremental/final_predictions.pkl');assert list(zip(ev.date,ev.symbol))==list(zip(original.date,original.symbol))
assert ev.groupby('date').symbol.nunique().eq(87).all()
params=dict(n_estimators=120,max_depth=3,learning_rate=.03,n_jobs=1,random_state=42,num_leaves=8,min_child_samples=200,reg_lambda=10,verbosity=-1)
rank=lgb.LGBMRanker(**params)
y=np.minimum(4,(train.groupby('date').y_20.rank(pct=True).to_numpy()*5).astype(int));rank.fit(train[cols].to_numpy(),y,group=train.groupby('date').size().to_numpy())
joblib.dump({'model':rank,'features':cols,'training_cutoff':cutoff,'params':params},R/'ranking_model.joblib',compress=3)
ck=joblib.load(ROOT/'incremental/final_models/residual_no_volume.joblib');res=ck['model'].predict(ev[ck['features']].to_numpy())/100
for _,ix in ev.groupby('date').indices.items():res[ix]=neutralize(res[ix],ev.iloc[ix][FACTORS])
np.testing.assert_allclose(res,original.score_residual_no_volume_tilt,atol=1e-10)
signals={'LightGBM Ranking':rank.predict(ev[cols].to_numpy()),'Residual ML':res,'Momentum':ev.rank_mom_12_1.to_numpy(),'Equal Weight':np.zeros(len(ev))}
risk={'name':'common_diversified','max_weight':.03,'turnover_penalty':1.0};metrics=[];daily={};slug={n:n.lower().replace(' ','_') for n in signals}
for name,score in signals.items():
 # Common allocator: no strategy-specific neutralization or rank-buffer constraints.
 d,t,w=backtest(ev,score,returns,risk,neutral=False,cost=.0015)
 k=slug[name];d.to_csv(R/f'{k}_daily.csv',index=False);t.to_csv(R/f'{k}_trades.csv.gz',index=False);w.to_csv(R/f'{k}_weights.csv.gz',index=False);daily[name]=d
 target=ev.copy();target['y']=ev.y_20;met,ic=predictive_metrics(target,score);ic.to_csv(R/f'{k}_ic.csv',index=False)
 metrics.append({'strategy':name,**portfolio_metrics(d),'rank_ic':met['rank_ic'],'rank_ic_valid_days':int(ic.rank_ic.notna().sum())});ev['score_'+k]=score
 # Independent accounting identities and constraints.
 np.testing.assert_allclose(100000*np.cumprod(1+d.net_return),d.equity,rtol=1e-10)
 np.testing.assert_allclose(d.cash+d.position_value,d.equity,atol=1e-7)
 np.testing.assert_allclose(t.cost.sum(),d.cost.sum(),atol=1e-7)
 cash=100000+np.sum(np.where(t.side=='BUY',-1,1)*t.shares*t.price-t.cost)+d.dividends.sum()
 np.testing.assert_allclose(cash,d.equity.iloc[-1],atol=1e-6)
 assert w.weight.max()<=.03+1e-8;assert w.groupby('date').weight.sum().max()<=.95+1e-8;assert d.cash.min()>=0
 assert (d.exante_before_scaling*d.vol_scale).dropna().max()<=.10000001
 if name=='Equal Weight':assert w.groupby('date').weight.std().max()<1e-9
 print(name,metrics[-1],flush=True)
ev.to_pickle(R/'predictions.pkl');pd.DataFrame(metrics).to_csv(R/'metrics.csv',index=False)
# Same bootstrap date indices for all pairs; intervals are pointwise, not multiplicity adjusted.
rng=np.random.default_rng(42);n=len(ev.date.unique());cis=[]
for block in [20,60]:
 ix=np.concatenate([(rng.integers(0,n,size=(5000,(n+block-1)//block))[:,:,None]+np.arange(block))%n],axis=0).reshape(5000,-1)[:,:n]
 for a,b in itertools.combinations(signals,2):
  delta=daily[a].net_return.to_numpy()-daily[b].net_return.to_numpy();draws=delta[ix].mean(axis=1)*252
  cis.append({'strategy':a,'comparator':b,'block_sessions':block,'annualized_mean_excess_return':float(delta.mean()*252),'ci_low':float(np.quantile(draws,.025)),'ci_high':float(np.quantile(draws,.975))})
pd.DataFrame(cis).to_csv(R/'excess_return_intervals.csv',index=False)
# Saved checkpoint replay, same sample boundaries and data provenance.
reload=joblib.load(R/'ranking_model.joblib');np.testing.assert_allclose(reload['model'].predict(ev[cols].to_numpy()),signals['LightGBM Ranking'],atol=1e-12)
manifest={'label':'RETROSPECTIVE COMPARISON, NOT A FRESH HOLDOUT','model_parameters_frozen':True,'training_label_end_exclusive':cutoff,'training_observations':len(train),'evaluation_observations':len(ev),'sessions':n,'execution_period':[str(ev.date.min()),str(ev.date.max())],'terminal_valuation_date':str(ev.label_end_1.max()),'target_end_max':str(ev.label_end_20.max()),'tradable_stocks':87,'rank_reference_stocks':88,'risk':risk,'cost_one_way':.0015,'device':'Apple M5 CPU; no MPS benefit for tree fitting and constrained optimization','ranker_refit':'Original frozen architecture/objective refit on the same permissible training sample as Residual ML; no new parameter selection. Original2024-25 weights not reused, avoiding unequal training cutoffs.','residual_checkpoint_sha256':hashlib.sha256((ROOT/'incremental/final_models/residual_no_volume.joblib').read_bytes()).hexdigest(),'allocator':'Common incremental.core.backtest with neutral=False for all; equal-weight score zero; no strategy-specific buffer or exposure constraints','prior_artifacts':'unchanged','checks':['All predictions finite on common keys','Residual checkpoint exactly reproduces prior frozen scores','Ranker checkpoint replay','Cash/holdings/dividend/cost/terminal-value reconciliation','Rebalance caps, leverage and ex ante volatility constraints','Equal Weight equal weights at rebalance'],'files_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [ROOT/'incremental/core.py',ROOT/'research/engine.py',ROOT/'research/selection_lock.json',ROOT/'incremental/selection_lock.json',R/'run.py']}}
assert all(np.isfinite(v).all() for v in signals.values())
(R/'results.json').write_text(json.dumps(manifest,indent=2));print('VERIFIED COMPLETE')
