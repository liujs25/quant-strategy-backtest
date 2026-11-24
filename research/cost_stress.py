"""Fixed final cost stress; descriptive only, never choose a cost after the reveal."""
from pathlib import Path
import sys,json
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import pandas as pd
import numpy as np
from research.engine import portfolio
from src.utils import portfolio_metrics
R=Path(__file__).resolve().parent;lock=json.loads((R/'selection_lock.json').read_text());f=pd.read_csv(R/'final_predictions.csv.gz',parse_dates=['date','label_end_1']);h=lock['spec']['horizon'];rows=[]
for cost in [0.,.0005,.0015,.003]:
 for name,score,mode,top in [('selected',f.score,lock['portfolio'],20),('equal_weight',np.zeros(len(f)),'equal',f.symbol.nunique()),('reversal',f.reversal_5,'equal',20)]:
  d,_=portfolio(f,score,rebalance=h,buffer=mode=='buffer',inverse_vol=mode=='inverse_vol',top=top,cost=cost)
  rows.append({'strategy':name,'cost_one_way':cost,**portfolio_metrics(d)})
pd.DataFrame(rows).to_csv(R/'final_cost_stress.csv',index=False)
