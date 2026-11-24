"""Replay saved checkpoints and independently reconcile all result artifacts."""
from pathlib import Path
import sys,json,hashlib
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import torch
import joblib
from src.data_preprocessing import build_panel,split_indices
from src.models import LSTM,Transformer,predict
from src.utils import predictive_metrics,portfolio_metrics
root=Path(__file__).resolve().parents[1];r=root/'results'
manifest=json.loads((r/'run_manifest.json').read_text());assert manifest['completed']
assert hashlib.sha256((root/'data/raw/all_stocks_5yr.csv').read_bytes()).hexdigest()==manifest['data_sha256']
for path,digest in manifest['source_sha256'].items():assert hashlib.sha256((root/path).read_bytes()).hexdigest()==digest,path
raw=pd.read_csv(root/'data/raw/all_stocks_5yr.csv');x,meta,_=build_panel(raw)
metrics=pd.read_csv(r/'model_comparison.csv');port=pd.read_csv(r/'portfolio_comparison.csv');checks=[]
for label in ['validation','test_1','test_2','test_3']:
    f=pd.read_csv(r/f'predictions_{label}.csv.gz',parse_dates=['date','signal_date','feature_start'])
    assert not f.duplicated(['date','symbol']).any();assert (f.signal_date<f.date).all()
    ti,ei,_=split_indices(meta,f.date.min(),f.date.max());target=meta.iloc[ei].reset_index(drop=True)
    assert list(zip(target.date,target.symbol))==list(zip(f.date,f.symbol))
    np.testing.assert_allclose(f.y,target.y,rtol=1e-10,atol=1e-14)
    sample=np.linspace(0,len(ei)-1,128,dtype=int);scaler=np.load(r/'models'/f'{label}_scaler.npz')
    xx=(x[ei[sample]]-scaler['mean'])/scaler['std']
    for name,cls in [('LSTM',LSTM),('Transformer',Transformer)]:
        checkpoint=torch.load(r/'models'/f'{label}_{name}.pth',map_location='cpu',weights_only=True)
        model=cls();model.load_state_dict(checkpoint['state_dict']);pred=predict(model,xx,'cpu')
        np.testing.assert_allclose(pred,f[name].to_numpy()[sample],atol=2e-6,rtol=1e-4)
    rf=joblib.load(r/'models'/f'{label}_RandomForest.joblib');np.testing.assert_allclose(rf.predict(xx.reshape(len(xx),-1)),f.RandomForest.to_numpy()[sample],atol=1e-12)
    ar=json.loads((r/'models'/f'{label}_ARIMA.json').read_text());ar_pred=[]
    for row in f.iloc[sample].itertuples():
        p=ar['params'].get(row.symbol)
        ar_pred.append((p['mean_percent']+p['ar1']*(row.last_intraday*100-p['mean_percent']))/100 if p else ar['fallback_mean'])
    np.testing.assert_allclose(ar_pred,f.ARIMA.to_numpy()[sample],atol=1e-12)
    np.testing.assert_allclose(f.historical_mean,meta.iloc[ti].y.mean(),atol=1e-12)
    checks.append(label+': 128 checkpoint replay samples per model passed')
for row in metrics.itertuples():
    f=pd.read_csv(r/f'predictions_{row.split}.csv.gz');m,_=predictive_metrics(f,f[row.model])
    for key in ['rmse','mae','r2','directional_accuracy','ic','rank_ic','icir']:
        np.testing.assert_allclose(m[key],getattr(row,key),atol=1e-10,equal_nan=True)
    if row.split=='test':
        assert m['n']==len(f)
        np.testing.assert_allclose(m['rmse']**2/(1-m['r2']),np.var(f.y),rtol=1e-10)
for row in port.itertuples():
    d=pd.read_csv(r/f'portfolio_{row.split}_{row.model}.csv');stats=portfolio_metrics(d)
    for key,value in stats.items():np.testing.assert_allclose(value,getattr(row,key),atol=1e-7,equal_nan=True)
    np.testing.assert_allclose(100000*np.cumprod(1+d.net_return),d.equity,rtol=1e-11)
    if row.model!='buy_hold':
        t=pd.read_csv(r/f'trades_{row.split}_{row.model}.csv.gz')
        assert d.cash.min()>0;assert (d.end_position_value==0).all()
        np.testing.assert_allclose(100000+t.net_pnl.sum(),d.equity.iloc[-1],atol=1e-7)
        np.testing.assert_allclose((t.entry_cost+t.exit_cost).sum(),d.cost.sum(),atol=1e-7)
        if len(t):
            pnl=t.groupby('date').net_pnl.sum().reindex(d.date,fill_value=0)
            np.testing.assert_allclose(100000+pnl.cumsum(),d.equity,atol=1e-7)
checks+=['All predictive metrics recomputed','All daily NAV, cash, compounded returns, transaction costs and trade P&L reconciled','RMSE/R² common-target variance identities verified']
(r/'verification.json').write_text(json.dumps({'passed':True,'checks':checks},indent=2));print('\n'.join(checks))
