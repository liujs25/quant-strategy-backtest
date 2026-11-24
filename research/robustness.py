"""Validation-only ablation and seed diagnostics of frozen candidate; no reselection."""
from pathlib import Path
import sys,json
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from src.utils import predictive_metrics,portfolio_metrics
from research.engine import portfolio
R=Path(__file__).resolve().parent;lock=json.loads((R/'selection_lock.json').read_text());spec=lock['spec'];h=spec['horizon'];p=pd.read_pickle(R/'development_panel.pkl');allcols=json.loads((R/'feature_names.json').read_text());out=[]
if spec['family'] not in ['LightGBM','XGBoost']:
 (R/'robustness_note.txt').write_text('Selected neural candidate: all 20/60-window and return/rank ablations already retained in the complete validation search. No post-selection refit grid added.');raise SystemExit()
for fold,(start,end,cut) in enumerate([('2021-01-01','2021-12-31','2019-10-01'),('2022-01-01','2023-12-31','2020-10-01')]):
 tr=p[(p[f'label_end_{h}']<cut)&p[f'y_{h}'].notna()].sort_values(['date','symbol']);dates=pd.read_csv(R/f"{lock['trial']}_fold{fold}_{lock['portfolio']}_daily.csv").date
 ev=p[p.date.isin(pd.to_datetime(dates))].sort_values(['date','symbol']).copy();ev['y']=ev[f'y_{h}']
 for name,cols,seed in [('full_seed7',allcols,7),('full_seed99',allcols,99),('short_only',[c for c in allcols if c in ['mom_1','mom_5','mom_20','intraday','overnight','vol_20','volume_ratio_20','range_20']],42),('no_cross_section',[c for c in allcols if not c.startswith('rank_') and c!='relative_mom20'],42)]:
  rank=spec['objective']=='ranking';y=tr[f'y_{h}'].to_numpy()
  if rank:y=np.minimum(4,(tr.groupby('date')[f'y_{h}'].rank(pct=True).to_numpy()*5).astype(int))
  kw=dict(n_estimators=120,max_depth=spec['depth'],learning_rate=.03,n_jobs=1,random_state=seed)
  if spec['family']=='LightGBM':
   cls=lgb.LGBMRanker if rank else lgb.LGBMRegressor;m=cls(**kw,num_leaves=2**spec['depth'],min_child_samples=200,reg_lambda=10,verbosity=-1)
  else:
   cls=xgb.XGBRanker if rank else xgb.XGBRegressor;m=cls(**kw,objective='rank:pairwise' if rank else 'reg:squarederror',min_child_weight=100,reg_lambda=10,subsample=.8,colsample_bytree=.8,tree_method='hist')
  m.fit(tr[cols].to_numpy(),y,**({'group':tr.groupby('date').size().to_numpy()} if rank else {}));score=m.predict(ev[cols].to_numpy());met,_=predictive_metrics(ev.reset_index(drop=True),score)
  d,_=portfolio(ev,score,rebalance=h,buffer=lock['portfolio']=='buffer',inverse_vol=lock['portfolio']=='inverse_vol');pm=portfolio_metrics(d)
  out.append({'fold':fold,'ablation':name,'rank_ic':met['rank_ic'],**pm});print(fold,name,met['rank_ic'],flush=True)
pd.DataFrame(out).to_csv(R/'validation_robustness.csv',index=False)
