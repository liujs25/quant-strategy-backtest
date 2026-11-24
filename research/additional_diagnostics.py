"""Post-reveal descriptive diagnostics only. No candidate changes or refits."""
from pathlib import Path
import sys,json
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import pandas as pd
import numpy as np
from research.engine import portfolio
from src.utils import portfolio_metrics
R=Path(__file__).resolve().parent;f=pd.read_csv(R/'final_predictions.csv.gz',parse_dates=['date','label_end_1']);d,t=portfolio(f,np.zeros(len(f)),rebalance=100000,top=f.symbol.nunique())
d.to_csv(R/'final_buy_hold_daily.csv',index=False);t.to_csv(R/'final_buy_hold_trades.csv.gz',index=False)
out={'purpose':'post-reveal descriptive checks; no reselection','buy_hold':portfolio_metrics(d)}
ic=pd.read_csv(R/'final_ic.csv');a=ic.rank_ic.to_numpy();rng=np.random.default_rng(42);n=len(a);draws=[]
for _ in range(5000):
 starts=rng.integers(0,n,size=(n+19)//20);ix=np.concatenate([(s+np.arange(20))%n for s in starts])[:n];draws.append(a[ix].mean())
out['rank_ic_block_95']=np.quantile(draws,[.025,.975]).tolist()
selected=pd.read_csv(R/'final_selected_daily.csv');momentum=pd.read_csv(R/'final_momentum_daily.csv');delta=selected.net_return.to_numpy()-momentum.net_return.to_numpy();draws=[]
for _ in range(5000):
 starts=rng.integers(0,n,size=(n+19)//20);ix=np.concatenate([(s+np.arange(20))%n for s in starts])[:n];draws.append(delta[ix].mean()*252)
out['selected_minus_momentum_mean_ann']=float(delta.mean()*252);out['selected_minus_momentum_block_95']=np.quantile(draws,[.025,.975]).tolist()
(R/'post_reveal_diagnostics.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
