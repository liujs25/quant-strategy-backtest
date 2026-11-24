"""Recompute portfolio comparisons on the common validation date/ticker intersection.
No refitting or forecast alteration. Keep raw trials separately for audit.
"""
from pathlib import Path
import sys,json,shutil
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import pandas as pd
import numpy as np
from research.engine import portfolio
from src.utils import predictive_metrics,portfolio_metrics
R=Path(__file__).resolve().parent
if (R/'selection_lock.json').exists():raise SystemExit('Selection is frozen; do not realign completed results')
raw=pd.read_csv(R/'validation_trials.csv');assert len(raw)==192
shutil.copy(R/'validation_trials.csv',R/'validation_trials_original_boundaries.csv')
panel=pd.read_pickle(R/'development_panel.pkl');specs=json.loads((R/'trial_plan.json').read_text());rows=[];coverage=[]
for fold in [0,1]:
    files={f'trial_{i:02}':pd.read_csv(R/f'trial_{i:02}_fold{fold}_predictions.csv.gz',parse_dates=['date']) for i in range(32)}
    keys=None
    for f in files.values():
        k=set(zip(f.date,f.symbol));keys=k if keys is None else keys&k
    base=panel[[(d,s) in keys for d,s in zip(panel.date,panel.symbol)]].sort_values(['date','symbol']).copy()
    coverage.append({'fold':fold,'n':len(base),'days':base.date.nunique(),'start':str(base.date.min()),'end':str(base.date.max())})
    for i,spec in enumerate(specs):
        ident=f'trial_{i:02}';h=spec['horizon'];scores=base[['date','symbol']].merge(files[ident][['date','symbol','score']],on=['date','symbol'],validate='one_to_one').score.to_numpy()
        scored=base.copy();scored['y']=scored[f'y_{h}'];met,_=predictive_metrics(scored.reset_index(drop=True),scores)
        if spec['objective']!='regression':
            for key in ['rmse','mae','r2','directional_accuracy']:met[key]=np.nan
        bench,_=portfolio(scored,np.zeros(len(scored)),rebalance=h,top=scored.symbol.nunique());bm=portfolio_metrics(bench)
        bench.to_csv(R/f'benchmark_h{h}_fold{fold}_daily.csv',index=False)
        for mode in ['equal','buffer','inverse_vol']:
            d,t=portfolio(scored,scores,rebalance=h,buffer=mode=='buffer',inverse_vol=mode=='inverse_vol');pm=portfolio_metrics(d)
            rows.append({'trial':ident,'fold':fold,'portfolio':mode,**spec,**met,**pm,'benchmark_annualized':bm['annualized_return'],'active_annualized':pm['annualized_return']-bm['annualized_return']})
            d.to_csv(R/f'{ident}_fold{fold}_{mode}_daily.csv',index=False);t.to_csv(R/f'{ident}_fold{fold}_{mode}_trades.csv.gz',index=False)
    print('Aligned fold',fold,coverage[-1],flush=True)
pd.DataFrame(rows).to_csv(R/'validation_trials.csv',index=False);(R/'common_validation_coverage.json').write_text(json.dumps(coverage,indent=2))
# Heuristic controls on exactly the same dates as all model candidates.
control_rows=[]
for fold in [0,1]:
    dates=pd.read_csv(R/f'trial_00_fold{fold}_equal_daily.csv',parse_dates=['date']).date
    base=panel[panel.date.isin(dates)].sort_values(['date','symbol']).copy()
    for h in [1,5,20]:
        base['y']=base[f'y_{h}'];bench=pd.read_csv(R/f'benchmark_h{h}_fold{fold}_daily.csv');bm=portfolio_metrics(bench)
        for name,col in [('momentum','mom_12_1'),('reversal','reversal_5'),('low_vol','vol_20')]:
            score=base[col].to_numpy()*(-1 if name=='low_vol' else 1);m,_=predictive_metrics(base.reset_index(drop=True),score)
            for mode in ['equal','buffer','inverse_vol']:
                daily,trades=portfolio(base,score,rebalance=h,buffer=mode=='buffer',inverse_vol=mode=='inverse_vol');pm=portfolio_metrics(daily)
                control_rows.append({'trial':f'{name}_{h}','fold':fold,'objective':name,'horizon':h,'portfolio':mode,'rank_ic':m['rank_ic'],**pm,'active_annualized':pm['annualized_return']-bm['annualized_return']})
                daily.to_csv(R/f'{name}_{h}_fold{fold}_{mode}_daily.csv',index=False)
pd.DataFrame(control_rows).to_csv(R/'validation_controls_aligned.csv',index=False)
