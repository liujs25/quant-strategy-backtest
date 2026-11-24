"""Freeze exactly one candidate using development validation, before final retrieval."""
from pathlib import Path
import json,hashlib
import numpy as np
import pandas as pd
R=Path(__file__).resolve().parent
if (R/'selection_lock.json').exists():raise SystemExit('Selection already locked')
d=pd.read_csv(R/'validation_trials.csv');assert len(d)==32*2*3, len(d)
summaries=[]
for (trial,mode),g in d.groupby(['trial','portfolio']):
    years=[]
    for fold in [0,1]:
        daily=pd.read_csv(R/f'{trial}_fold{fold}_{mode}_daily.csv',parse_dates=['date'])
        # Calendar-year excess returns versus daily benchmark reconstructed later in final report.
        for year,z in daily.groupby(daily.date.dt.year):
            h=int(g.horizon.iloc[0]);bench=pd.read_csv(R/f'benchmark_h{h}_fold{fold}_daily.csv',parse_dates=['date']);bz=bench[bench.date.dt.year==year]
            years.append({'year':int(year),'net_return':float(np.prod(1+z.net_return)-1),'benchmark_return':float(np.prod(1+bz.net_return)-1),'active_return':float(np.prod(1+z.net_return)-np.prod(1+bz.net_return))})
    summaries.append({'trial':trial,'portfolio':mode,'mean_active_annualized':g.active_annualized.mean(),'worst_fold_active':g.active_annualized.min(),
                      'mean_rank_ic':g.rank_ic.mean(),'worst_rank_ic':g.rank_ic.min(),'mean_turnover':g.annualized_turnover.mean(),
                      'eligible_gate':bool((g.active_annualized>0).all() and (g.rank_ic>0).all()),'year_returns':years})
s=pd.DataFrame(summaries).sort_values(['eligible_gate','mean_active_annualized','trial'],ascending=[False,False,True]);s.to_json(R/'validation_selection_summary.json',orient='records',indent=2)
best=s.iloc[0];trial=best.trial;num=int(trial.split('_')[1]);spec=json.loads((R/'trial_plan.json').read_text())[num]
# Gate must be disclosed, never suppress a failed final result or manufacture a passing candidate.
lock={'trial':trial,'portfolio':best.portfolio,'spec':spec,'passed_validation_gate':bool(best.eligible_gate),
      'selection_score':float(best.mean_active_annualized),'gate':'positive active annualized return and Rank IC in both validation folds',
      'status':'confirmatory candidate' if best.eligible_gate else 'exploratory candidate only; no validation-supported strategy',
      'trial_count_models':32,'portfolio_variants':3,'folds':2,'final_start':'2024-01-01','final_end':'2025-12-31',
      'universe':json.loads((R/'eligibility.json').read_text())['eligible'],
      'protocol_sha256':hashlib.sha256((R/'protocol.json').read_bytes()).hexdigest(),
      'validation_sha256':hashlib.sha256((R/'validation_trials.csv').read_bytes()).hexdigest(),
      'final_rules':'Fit once on labels ending before 2022-10-01 (full feature/sequence embargo); frozen architecture/objective/portfolio; no final tuning. Compare equal-weight, momentum and simple reversal controls. Report all final results regardless of sign.',
      'publication':'LOCAL ONLY'}
if spec['family'] in ['LSTM','Transformer']:
 import torch
 ck=torch.load(R/'models'/f'{trial}_fold1.pth',map_location='cpu',weights_only=True);lock['epochs']=ck['epochs']
(R/'selection_lock.json').write_text(json.dumps(lock,indent=2));print(json.dumps(lock,indent=2))
