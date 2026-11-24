"""Independent verification of research ledgers, metrics, causal boundaries and final replay."""
from pathlib import Path
import json,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import joblib
from src.utils import portfolio_metrics
R=Path(__file__).resolve().parent;checks=[]
for p in list(R.glob('trial_*_daily.csv'))+list(R.glob('final_*_daily.csv')):
    d=pd.read_csv(p);assert (d.cash>=-1e-6).all(),p
    np.testing.assert_allclose(d.cash+d.position_value,d.equity,atol=1e-7)
    np.testing.assert_allclose(100000*np.cumprod(1+d.net_return),d.equity,rtol=1e-10)
    tpath=p.with_name(p.name.replace('_daily.csv','_trades.csv.gz'))
    if tpath.exists():
        t=pd.read_csv(tpath);np.testing.assert_allclose(t.cost.sum(),d.cost.sum(),atol=1e-7)
        sign=np.where(t.side=='BUY',-1,1);cash=100000+(sign*t.shares*t.price-t.cost).sum()+d.dividends.sum()
        np.testing.assert_allclose(cash,d.cash.iloc[-1],atol=1e-6)
        assert d.position_value.iloc[-1]==0
checks.append('All saved daily NAVs, cash, marked positions, costs and compounded returns reconciled; available trade ledgers reconcile to final cash including dividends.')
lock=json.loads((R/'selection_lock.json').read_text());final=json.loads((R/'final_results.json').read_text());f=pd.read_csv(R/'final_predictions.csv.gz',parse_dates=['date','signal_date'])
assert (f.signal_date<f.date).all();assert f.date.min()>=pd.Timestamp('2024-01-01');assert f.date.max()<=pd.Timestamp('2025-12-31')
assert set(f.symbol)==set(lock['universe']);assert not f.duplicated(['date','symbol']).any()
if (R/'models/final_selected.joblib').exists():
    m=joblib.load(R/'models/final_selected.joblib');cols=json.loads((R/'feature_names.json').read_text());ix=np.linspace(0,len(f)-1,256,dtype=int)
    np.testing.assert_allclose(m.predict(f.iloc[ix][cols].to_numpy()),f.score.to_numpy()[ix],rtol=1e-7,atol=1e-8)
checks.append('Fixed final universe and prediction dates verified; tree checkpoint replay on 256 evenly spaced observations passed when applicable.')
for name,met in final['portfolio'].items():
    d=pd.read_csv(R/f'final_{name}_daily.csv');new=portfolio_metrics(d)
    for k in new:np.testing.assert_allclose(new[k],met[k],rtol=1e-9,atol=1e-8)
checks.append('Final portfolio summary recomputed independently from daily returns.')
(R/'verification.json').write_text(json.dumps({'passed':True,'checks':checks},indent=2));print('\n'.join(checks))
from research.engine import load_panel,make_features
old=pd.read_pickle(R/'development_panel.pkl');dev=load_panel(R/'development_raw');future=load_panel(R/'final_raw');reference=set(dev.symbol)
combined=pd.concat([dev,future[(future.date>dev.date.max())&future.symbol.isin(reference)]])
new,features=make_features(combined);new=new[new.symbol.isin(lock['universe'])].set_index(['date','symbol'])
sample=old.iloc[np.linspace(0,len(old)-1,512,dtype=int)].set_index(['date','symbol'])
np.testing.assert_allclose(new.loc[sample.index,features],sample[features],equal_nan=True,rtol=1e-10,atol=1e-12)
checks.append('Appending final data leaves 512 development feature vectors unchanged; rank reference universe is consistent.')
(R/'verification.json').write_text(json.dumps({'passed':True,'checks':checks},indent=2))
