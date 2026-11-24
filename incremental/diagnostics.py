from pathlib import Path
import sys,json
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from incremental.core import data,backtest,FACTORS
from incremental.study import factor_returns,hac
from src.utils import portfolio_metrics
R=Path(__file__).resolve().parent;lock=json.loads((R/'selection_lock.json').read_text());protocol=json.loads((R/'protocol.json').read_text());f=pd.read_pickle(R/'final_predictions.pkl');_,_,returns=data(R.parent,final=True);risk=next(x for x in protocol['risk_variants'] if x['name']==lock['risk']);rows=[]
for cost in [0.,.0005,.0015,.003]:
 for name in [lock['strategy'],'momentum']:
  d,_,_=backtest(f,f['score_'+name],returns,risk,neutral=name.startswith('residual_'),cost=cost);rows.append({'strategy':name,'cost_one_way':cost,**portfolio_metrics(d)})
pd.DataFrame(rows).to_csv(R/'final_cost_stress.csv',index=False)
# Factor exposure reports from actual rebalance weights; neutrality holds before scaling, not after differing risk scales.
exposures=[]
for p in R.glob('*_weights.csv.gz'):
 w=pd.read_csv(p)
 for date,g in w.groupby('date'):
  exposures.append({'portfolio':p.name.replace('_weights.csv.gz',''),'date':date,'invested':g.weight.sum(),**{c:float(g.weight@g[c]) for c in FACTORS}})
pd.DataFrame(exposures).to_csv(R/'portfolio_factor_exposures.csv',index=False)
print('Diagnostics complete')
