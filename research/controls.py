from pathlib import Path
import sys,json
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import pandas as pd
import numpy as np
from research.engine import portfolio
from src.utils import predictive_metrics,portfolio_metrics
R=Path(__file__).resolve().parent;p=pd.read_pickle(R/'development_panel.pkl');rows=[]
for fold,(start,end) in enumerate([('2021-01-01','2021-12-31'),('2022-01-01','2023-12-31')]):
    for h in [1,5,20]:
        g=p[p.date.between(start,end)&(p[f'label_end_{h}']<=end)].copy();g['y']=g[f'y_{h}']
        for name,col in [('momentum','mom_12_1'),('reversal','reversal_5'),('low_vol','vol_20')]:
            score=g[col].to_numpy()*(-1 if name=='low_vol' else 1);m,_=predictive_metrics(g.reset_index(drop=True),score)
            for mode in ['equal','buffer','inverse_vol']:
                d,t=portfolio(g,score,rebalance=h,buffer=mode=='buffer',inverse_vol=mode=='inverse_vol');b,_=portfolio(g,np.zeros(len(g)),rebalance=h,top=g.symbol.nunique())
                pm=portfolio_metrics(d);bm=portfolio_metrics(b);rows.append({'trial':f'{name}_{h}','fold':fold,'family':'heuristic','objective':name,'horizon':h,'portfolio':mode,**m,**pm,'active_annualized':pm['annualized_return']-bm['annualized_return'],'benchmark_annualized':bm['annualized_return']})
                d.to_csv(R/f'{name}_{h}_fold{fold}_{mode}_daily.csv',index=False)
pd.DataFrame(rows).to_csv(R/'validation_controls.csv',index=False);print('Controls complete')
