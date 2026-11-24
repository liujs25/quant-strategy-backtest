from pathlib import Path
import sys,json
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from incremental.core import data,backtest
from incremental.study import fit_models,scores,factor_returns,hac
from src.utils import portfolio_metrics,predictive_metrics
R=Path(__file__).resolve().parent
if (R/'final_results.json').exists():raise SystemExit('Already evaluated; no reselection')
lock=json.loads((R/'selection_lock.json').read_text());protocol=json.loads((R/'protocol.json').read_text());panel,cols,returns=data(R.parent,final=True)
train=panel[(panel.label_end_20<lock['final_fit_label_end_exclusive'])&panel.y_20.notna()].sort_values(['date','symbol']).reset_index(drop=True)
ev=panel[panel.date.between(*lock['final_period'])&(panel.label_end_20<=lock['final_period'][1])].sort_values(['date','symbol']).reset_index(drop=True)
assert ev.groupby('date').symbol.nunique().eq(87).all();assert len(ev)>0
models_dir=R/'final_models';models_dir.mkdir(exist_ok=True);models,features=fit_models(train,cols,models_dir,protocol['model_params']);ss=scores(ev,models,features)
# Fitting all fixed ablations does not evaluate/select them: only locked candidate and fixed traditional baselines traded.
for c in ss:ev['score_'+c]=ss[c].to_numpy()
ev.to_pickle(R/'final_predictions.pkl');factors=factor_returns(ev);risk=next(r for r in protocol['risk_variants'] if r['name']==lock['risk']);results={};reg=[]
strategies=list(dict.fromkeys([lock['strategy'],'equal_weight','momentum','multi_momentum','low_vol','traditional_combo']))
for name in strategies:
 d,t,w=backtest(ev,ss[name],returns,risk,neutral=name.startswith('residual_'));d.to_csv(R/f'final_{name}_daily.csv',index=False);t.to_csv(R/f'final_{name}_trades.csv.gz',index=False);w.to_csv(R/f'final_{name}_weights.csv.gz',index=False);results[name]=portfolio_metrics(d);reg.append({'strategy':name,**hac(d,factors)})
champ=lock['traditional_champion'];crisk=next(r for r in protocol['risk_variants'] if r['name']==champ['risk']);d,t,w=backtest(ev,ss[champ['strategy']],returns,crisk);d.to_csv(R/'final_champion_daily.csv',index=False);t.to_csv(R/'final_champion_trades.csv.gz',index=False);w.to_csv(R/'final_champion_weights.csv.gz',index=False);results['champion']=portfolio_metrics(d)
a=pd.read_csv(R/f"final_{lock['strategy']}_daily.csv");inference=[];rng=np.random.default_rng(42)
for comparator in ['momentum','champion']:
 b=pd.read_csv(R/f'final_{comparator}_daily.csv');delta=(a.net_return-b.net_return).to_numpy();n=len(delta)
 for block in [20,60]:
  draws=[]
  for _ in range(5000):
   starts=rng.integers(0,n,size=(n+block-1)//block);ix=np.concatenate([(s+np.arange(block))%n for s in starts])[:n];draws.append(delta[ix].mean()*252)
  inference.append({'comparator':comparator,'block':block,'mean_active_annualized':float(delta.mean()*252),'ci95':np.quantile(draws,[.025,.975]).tolist()})
target=ev.copy();target['y']=ev.y_20;met,ic=predictive_metrics(target,ss[lock['strategy']]);ic.to_csv(R/'final_ic.csv',index=False)
for k in ['rmse','mae','r2','directional_accuracy']:met[k]=None
out={'lock':lock,'portfolio':results,'inference':inference,'predictive_score_metrics':met,'factor_regressions':reg,'observations':len(ev),'days':ev.date.nunique(),'period':[str(ev.date.min()),str(ev.date.max())],'note':'Date tails shortened so20-session labels finish by cutoff. No final selection.'}
(R/'final_results.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
