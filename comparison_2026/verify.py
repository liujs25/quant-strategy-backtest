"""Recalculate the published experiment without raw data or feature caches."""
from pathlib import Path
import argparse
import sys
import json
import hashlib
import itertools
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import joblib
from src.utils import portfolio_metrics,predictive_metrics
from incremental.core import FACTORS,neutralize
R=Path(__file__).resolve().parent
parser=argparse.ArgumentParser();parser.add_argument('--history',action='store_true',help='Also verify the complete original local workspace, including excluded caches')
args=parser.parse_args()
meta=json.loads((R/'results.json').read_text());metrics=pd.read_csv(R/'metrics.csv');f=pd.read_pickle(R/'predictions.pkl');daily={};keys=None
assert not f.duplicated(['date','symbol']).any()
assert f.groupby('date').symbol.nunique().eq(87).all()
assert len(f)==meta['evaluation_observations']
assert f.signal_date.lt(f.date).all()
assert str(f.date.min())==meta['execution_period'][0]
assert str(f.date.max())==meta['execution_period'][1]
assert str(f.label_end_20.max())==meta['target_end_max']
for row in metrics.itertuples():
    name=row.strategy.lower().replace(' ','_')
    d=pd.read_csv(R/f'{name}_daily.csv');t=pd.read_csv(R/f'{name}_trades.csv.gz');w=pd.read_csv(R/f'{name}_weights.csv.gz')
    if keys is None:keys=d.date.tolist()
    assert keys==d.date.tolist();daily[row.strategy]=d
    for k,v in portfolio_metrics(d).items():np.testing.assert_allclose(v,getattr(row,k),atol=1e-8)
    target=f.copy();target['y']=f.y_20
    pred,ic=predictive_metrics(target,f['score_'+name])
    np.testing.assert_allclose(pred['rank_ic'],row.rank_ic,equal_nan=True,atol=1e-10)
    saved_ic=pd.read_csv(R/f'{name}_ic.csv')
    np.testing.assert_allclose(ic[['ic','rank_ic']],saved_ic[['ic','rank_ic']],equal_nan=True,atol=1e-10)
    np.testing.assert_allclose(100000*np.cumprod(1+d.net_return),d.equity,rtol=1e-10)
    np.testing.assert_allclose(d.cash+d.position_value,d.equity,atol=1e-7)
    np.testing.assert_allclose(t.cost.sum(),d.cost.sum(),atol=1e-7)
    np.testing.assert_allclose(100000+np.sum(np.where(t.side=='BUY',-1,1)*t.shares*t.price-t.cost)+d.dividends.sum(),d.equity.iloc[-1],atol=1e-6)
    assert len(d)==204 and d.position_value.iloc[-1]==0 and d.cash.min()>=0
    assert w.weight.max()<=.03000001 and w.groupby('date').weight.sum().max()<=.95000001
    assert (d.exante_before_scaling*d.vol_scale).dropna().max()<=.10000001
    if row.strategy=='Equal Weight':assert w.groupby('date').weight.std().max()<1e-9
for path,digest in meta['files_sha256'].items():assert hashlib.sha256((R.parent/path).read_bytes()).hexdigest()==digest,path
rank=joblib.load(R/'ranking_model.joblib')
np.testing.assert_allclose(rank['model'].predict(f[rank['features']].to_numpy()),f.score_lightgbm_ranking,atol=1e-12)
p=R.parent/'incremental/final_models/residual_no_volume.joblib'
assert hashlib.sha256(p.read_bytes()).hexdigest()==meta['residual_checkpoint_sha256']
ck=joblib.load(p);res=ck['model'].predict(f[ck['features']].to_numpy())/100
for _,ix in f.groupby('date').indices.items():res[ix]=neutralize(res[ix],f.iloc[ix][FACTORS])
np.testing.assert_allclose(res,f.score_residual_ml,atol=1e-10)
np.testing.assert_array_equal(f.score_momentum,f.rank_mom_12_1)
assert (f.score_equal_weight==0).all()
ci=pd.read_csv(R/'excess_return_intervals.csv');rng=np.random.default_rng(42);n=204;count=0
for block in [20,60]:
    starts=rng.integers(0,n,size=(5000,(n+block-1)//block))
    ix=((starts[:,:,None]+np.arange(block))%n).reshape(5000,-1)[:,:n]
    for a,b in itertools.combinations(metrics.strategy,2):
        delta=daily[a].net_return.to_numpy()-daily[b].net_return.to_numpy()
        draws=delta[ix].mean(axis=1)*252
        row=ci[(ci.strategy==a)&(ci.comparator==b)&(ci.block_sessions==block)].iloc[0]
        np.testing.assert_allclose([delta.mean()*252,*np.quantile(draws,[.025,.975])],[row.annualized_mean_excess_return,row.ci_low,row.ci_high],atol=1e-11)
        count+=1
checks=['Common 87-stock, 204-session sample and temporal label boundaries','All portfolio metrics and daily IC independently recomputed','All 12 pairwise block-bootstrap intervals recomputed','Both frozen model checkpoints replayed on every evaluation observation','Cash/holdings/dividend/cost reconciliation and terminal liquidation','Rebalance weight/exposure caps, ex ante volatility ceiling and equal-weight equality','Frozen training source and specification hashes']
out={'passed':True,'checks':checks,'observations':len(f),'sessions':n,'intervals_checked':count}
if args.history:
    old=json.loads((R.parent/'artifact_manifest.json').read_text());checked=0
    for path,digest in old['files'].items():
        if path.startswith(('results/','research/','incremental/')):
            assert hashlib.sha256((R.parent/path).read_bytes()).hexdigest()==digest,path;checked+=1
    out['prior_local_files_verified_unchanged']=checked
(R/'verification.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
