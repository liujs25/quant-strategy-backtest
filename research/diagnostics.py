from pathlib import Path
import sys,json
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import pandas as pd
import numpy as np
from src.utils import corr
R=Path(__file__).resolve().parent
p=pd.read_pickle(R/'development_panel.pkl');cols=json.loads((R/'feature_names.json').read_text());rows=[]
for year in [2021,2022,2023]:
    g=p[p.date.dt.year==year]
    for h in [1,5,20]:
        valid=g[g[f'label_end_{h}'].dt.year==year]
        for col in cols:
            daily=[]
            for _,d in valid.groupby('date'):
                a=d[[col,f'y_{h}']].dropna();daily.append(corr(a[col].rank(),a[f'y_{h}'].rank()))
            vals=np.asarray(daily);rows.append({'year':year,'horizon':h,'feature':col,'mean_rank_ic':np.nanmean(vals) if np.isfinite(vals).any() else None,'valid_days':int(np.isfinite(vals).sum())})
pd.DataFrame(rows).to_csv(R/'validation_feature_ic.csv',index=False)
print('Feature diagnostics complete')
