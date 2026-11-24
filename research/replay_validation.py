from pathlib import Path
import sys,json
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import torch
import joblib
from research.search import RankNet
R=Path(__file__).resolve().parent;panel=pd.read_pickle(R/'development_panel.pkl');cols=json.loads((R/'feature_names.json').read_text());specs=json.loads((R/'trial_plan.json').read_text());checks=[];torch.set_num_threads(4)
lookup=panel.set_index(['date','symbol'])
for fold in [0,1]:
 for i,spec in enumerate(specs):
  ident=f'trial_{i:02}';f=pd.read_csv(R/f'{ident}_fold{fold}_predictions.csv.gz',parse_dates=['date']);f=f.iloc[np.linspace(0,len(f)-1,64,dtype=int)]
  if spec['family'] in ['LightGBM','XGBoost']:
   x=lookup.loc[list(zip(f.date,f.symbol)),cols].to_numpy();m=joblib.load(R/'models'/f'{ident}_fold{fold}.joblib');p=m.predict(x)
  else:
   ck=torch.load(R/'models'/f'{ident}_fold{fold}.pth',map_location='cpu',weights_only=True);length=spec['lookback'];seqcols=ck['columns'];xx=[]
   for row in f.itertuples():
    g=panel[(panel.symbol==row.symbol)&(panel.date<=row.date)].sort_values('date');xx.append(g[seqcols].tail(length).to_numpy(dtype='float32'))
   m=RankNet(spec['family'],len(seqcols),length).eval();m.load_state_dict(ck['state_dict'])
   with torch.no_grad():p=m(torch.from_numpy(np.stack(xx))).numpy()
   if spec['objective']=='regression':p/=100
  np.testing.assert_allclose(p,f.score,atol=3e-6,rtol=1e-3);checks.append(f'{ident} fold {fold}: 64 replay samples passed')
(R/'validation_replay.json').write_text(json.dumps({'passed':True,'checks':checks},indent=2));print('All 64 fitted checkpoints replayed successfully')
