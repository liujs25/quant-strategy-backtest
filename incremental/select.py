from pathlib import Path
import json,hashlib
import numpy as np
import pandas as pd
R=Path(__file__).resolve().parent
if (R/'selection_lock.json').exists():raise SystemExit('Locked')
d=pd.read_csv(R/'development_results.csv');assert len(d)==60
traditional=['equal_weight','momentum','multi_momentum','low_vol','traditional_combo'];rows=[]
for (s,r),g in d.groupby(['strategy','risk']):
    scores=[];positive=[];sharpe=[];dd=[]
    for fold in [0,1]:
        a=pd.read_csv(R/f'fold{fold}_{s}_{r}_daily.csv');b=pd.read_csv(R/f'fold{fold}_momentum_{r}_daily.csv');delta=a.net_return-b.net_return
        scores.append(delta.mean()/delta.std()*np.sqrt(252) if delta.std()>1e-12 else 0);positive.append(delta.mean()>0)
        sr=g[g.fold==fold].iloc[0];br=d[(d.fold==fold)&(d.strategy=='momentum')&(d.risk==r)].iloc[0];sharpe.append(sr.sharpe-br.sharpe);dd.append(sr.max_drawdown-br.max_drawdown)
    rows.append({'strategy':s,'risk':r,'mean_active_ir':float(np.mean(scores)),'positive_both':bool(all(positive)),'mean_sharpe_gain':float(np.mean(sharpe)),'worst_drawdown_difference':float(max(dd)),'eligible':bool(all(positive) and np.mean(sharpe)>0 and max(dd)<=0)})
s=pd.DataFrame(rows);s.to_csv(R/'selection_summary.csv',index=False)
c=s[~s.strategy.isin(traditional)].sort_values(['eligible','mean_active_ir','strategy'],ascending=[False,False,True]).iloc[0]
b=d[d.strategy.isin(traditional)].groupby(['strategy','risk']).sharpe.mean().sort_values(ascending=False).index[0]
lock={'strategy':c.strategy,'risk':c.risk,'gate_passed':bool(c.eligible),'selection':c.to_dict(),'traditional_champion':{'strategy':b[0],'risk':b[1]},'final_fit_label_end_exclusive':'2024-10-01','final_period':['2026-01-01','2026-11-20'],'protocol_sha256':hashlib.sha256((R/'protocol.json').read_bytes()).hexdigest(),'development_results_sha256':hashlib.sha256((R/'development_results.csv').read_bytes()).hexdigest(),'publication':'LOCAL ONLY','interpretation':'confirmatory candidate' if c.eligible else 'exploratory challenger; failed development gate'}
(R/'selection_lock.json').write_text(json.dumps(lock,indent=2));print(json.dumps(lock,indent=2))
