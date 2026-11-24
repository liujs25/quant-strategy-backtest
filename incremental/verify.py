from pathlib import Path
import sys,json
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import joblib
from incremental.core import data,FACTORS,neutralize
from incremental.study import scores
from src.utils import portfolio_metrics
R=Path(__file__).resolve().parent;checks=[]
for p in R.glob('*_daily.csv'):
 d=pd.read_csv(p);t=pd.read_csv(p.with_name(p.name.replace('_daily.csv','_trades.csv.gz')))
 np.testing.assert_allclose(d.cash+d.position_value,d.equity,atol=1e-7);np.testing.assert_allclose(100000*np.cumprod(1+d.net_return),d.equity,rtol=1e-10)
 np.testing.assert_allclose(d.cost.sum(),t.cost.sum(),atol=1e-7);sign=np.where(t.side=='BUY',-1,1)
 cash=100000+np.sum(sign*t.shares*t.price-t.cost)+d.dividends.sum();np.testing.assert_allclose(cash,d.cash.iloc[-1],atol=1e-6)
 assert (d.cash>=-1e-6).all();assert d.position_value.iloc[-1]==0
 if d.neutrality_error.notna().any():assert d.neutrality_error.max()<1e-6
 w=pd.read_csv(p.with_name(p.name.replace('_daily.csv','_weights.csv.gz')));cap=.03 if 'diversified' in p.name else .05
 if p.name.startswith('final'):
  lock=json.loads((R/'selection_lock.json').read_text());cap=.05 if p.name=='final_champion_daily.csv' else (.03 if lock['risk']=='diversified' else .05)
 assert w.weight.max()<=cap+1e-7
checks.append('Every development/final ledger reconciles cash, positions, dividends, costs, returns and final liquidation; position caps and pre-scale factor neutrality passed.')
# Replay all development predictions, including factor projection, on full cross sections.
for fold in [0,1]:
 f=pd.read_pickle(R/f'predictions_fold{fold}.pkl');dates=f.date.unique()[::max(1,f.date.nunique()//8)];g=f[f.date.isin(dates)].reset_index(drop=True)
 models={};features={}
 for p in (R/f'models_fold{fold}').glob('*.joblib'):
  ck=joblib.load(p);models[p.stem]=ck['model'];features[p.stem]=ck['features']
 replay=scores(g,models,features)
 for col in replay:np.testing.assert_allclose(replay[col],g['score_'+col],atol=1e-10)
checks.append('All eight development model checkpoints and projected/composite scores replayed on eight full cross sections per fold.')
if (R/'final_results.json').exists():
 f=pd.read_pickle(R/'final_predictions.pkl');models={};features={}
 for p in (R/'final_models').glob('*.joblib'):
  ck=joblib.load(p);models[p.stem]=ck['model'];features[p.stem]=ck['features']
 replay=scores(f,models,features)
 for col in replay:np.testing.assert_allclose(replay[col],f['score_'+col],atol=1e-10)
 result=json.loads((R/'final_results.json').read_text())
 for name,met in result['portfolio'].items():
  d=pd.read_csv(R/f'final_{name}_daily.csv');actual=portfolio_metrics(d)
  for k in actual:np.testing.assert_allclose(actual[k],met[k],atol=1e-8)
 assert f.signal_date.lt(f.date).all();assert f.date.min()>=pd.Timestamp('2026-01-01');assert f.label_end_20.max()<=pd.Timestamp('2026-11-20')
 old=pd.read_pickle(R/'development_panel.pkl');combined,cols,_=data(R.parent,final=True);new=combined.set_index(['date','symbol']);sample=old.iloc[np.linspace(0,len(old)-1,256,dtype=int)].set_index(['date','symbol']);np.testing.assert_allclose(new.loc[sample.index,cols],sample[cols],atol=1e-12,equal_nan=True)
 checks.append('All final forecasts replayed; metrics recomputed; date/label boundaries passed; appending holdout preserves 256 past feature vectors.')
(R/'verification.json').write_text(json.dumps({'passed':True,'checks':checks},indent=2));print('\n'.join(checks))
