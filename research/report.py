from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parent;fig=R/'figures';fig.mkdir(exist_ok=True)
d=pd.read_csv(R/'validation_trials.csv');final=json.loads((R/'final_results.json').read_text());lock=final['selection']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
def save(name):
 for ext in ['png','pdf']:plt.savefig(fig/f'{name}.{ext}',dpi=300,bbox_inches='tight')
 plt.close()
f,ax=plt.subplots(2,1,figsize=(10,7),sharex=True,layout='constrained')
for name in final['portfolio']:
 p=pd.read_csv(R/f'final_{name}_daily.csv',parse_dates=['date']);eq=p.equity.to_numpy();dd=eq/np.maximum.accumulate(np.r_[100000,eq])[1:]-1
 ax[0].plot(p.date,eq/100000,label=name);ax[1].plot(p.date,dd*100)
ax[0].set(title='Frozen candidate: corrected 2024–2025 holdout evaluation',ylabel='Net wealth multiple');ax[0].legend();ax[1].set(ylabel='Drawdown (%)',xlabel='Execution date')
for a in ax:a.grid(alpha=.2)
save('final_equity')
f,a=plt.subplots(figsize=(10,5),layout='constrained')
for name,g in d.groupby('family'):a.scatter(g.annualized_turnover,g.active_annualized*100,label=name,alpha=.65)
a.axhline(0,c='black',lw=.8);a.set(xlabel='Annualized two-sided turnover',ylabel='Validation active CAGR (percentage points)',title='All validation models and portfolio variants');a.legend();a.grid(alpha=.2);save('validation_turnover')
f,a=plt.subplots(figsize=(10,5),layout='constrained')
s=d.groupby(['trial','portfolio']).active_annualized.mean().sort_values();a.bar(np.arange(len(s)),s*100,color=['#D55E00' if v<0 else '#0072B2' for v in s]);a.set(xlabel='All 96 configurations, ordered by validation score',ylabel='Mean validation active CAGR (pp)',title='Search breadth and negative results retained');a.grid(axis='y',alpha=.2);save('all_trials')
def table(df):
 out=['| '+' | '.join(df.columns)+' |','|'+'|'.join(['---']*len(df.columns))+'|']
 for row in df.itertuples(index=False,name=None):out.append('| '+' | '.join('undefined' if pd.isna(x) else f'{x:.5g}' if isinstance(x,(float,np.floating)) else str(x) for x in row)+' |')
 return '\n'.join(out)
port=pd.DataFrame(final['portfolio']).T.reset_index(names='strategy');best=d[(d.trial==lock['trial'])&(d.portfolio==lock['portfolio'])]
elig=json.loads((R/'eligibility.json').read_text());manifest=json.loads((R/'development_raw/download_manifest.json').read_text());unavailable=[x['symbol'] for x in manifest['files'] if x['status']=='unavailable']
ci=final['active_block_bootstrap_95'];active=final['active_mean_return_annualized'];passed=ci[0]>0
assessment=('The frozen candidate has positive benchmark-relative mean net return with a block-bootstrap interval above zero on this final sample. This is limited evidence of out-of-sample economic value, subject to universe, data and inference limitations.' if passed else 'The final evaluation does not establish reliable benchmark-relative economic value: the paired block-bootstrap interval does not lie wholly above zero. Positive absolute returns or a winning validation score do not establish alpha.')
text=f'''# Expanded local research: diagnosis, validation search and sealed evaluation

**Publication status: LOCAL ONLY. No commit or push.** The verified original redesign is preserved in `../results/` and `../audit/baseline_verified_20261124.zip`. Its inspected 2017–2018 interval is exploratory. See `baseline_diagnosis.md` for the signal, feature, fit and cost diagnosis.

## Experiment design

All work ran locally on Apple M5. PyTorch used MPS; LightGBM/XGBoost used single-thread CPU after a local OpenMP crash before the first completed trial. Native dependencies and exact package versions are recorded. Data is Yahoo Finance through yfinance, development requests ending strictly before 2024. Final data was downloaded only after `selection_lock.json` was written. It is a newly untouched **retrospective** holdout, not a prospective live experiment or unknown market history.

Universe: top 100 original Kaggle names by 2016 median dollar volume. Provider returned 88, and 87 met development-history/completeness rules. Unavailable: {', '.join(unavailable)}. Development exclusions: `{json.dumps(elig['excluded'])}`. Conditioning on current provider availability and complete 2021–2023 history adds survivorship bias. This is explicitly a survivor-conditioned study, not a historical S&P membership backtest. Final names are frozen and never substituted based on future performance. Final missing names/quotes fail loudly.

Training precedes 2021 validation by a conservative cutoff (labels end before 2019-10-01); second-fold training labels end before 2020-10-01, with validation 2022–2023. These gaps exceed the 252-session feature lookback plus 60-session sequence length. Final training labels end before 2022-10-01; evaluation is 2024–2025. No train/evaluation raw-history overlap under the recorded market calendar; assertions and a split audit document the boundary. No final hyperparameter or portfolio tuning. The strategy universe is retrospective and not made unbiased by these temporal safeguards.

55 features include 1/5/10/20/60/120/252-session momentum, 12-minus-1-month momentum, multi-scale volatility, moving-average distance, volume ratios, ranges, intraday/overnight changes, close location, illiquidity, relative momentum and per-date ranks. Features use information only through the prior close. Zero-range close location is neutral zero; the validation bug and superseded trials are retained. Neural inputs use 10 rank/market features and layer normalization; no future-fitted scaler.

Targets are dividend-inclusive open-to-open returns over 1, 5 or 20 sessions, entering after the signal close. Horizons deliberately differ from the original intraday baseline. Multi-day labels overlap **within** folds; their uncertainty cannot be treated as independent observations. Cross-sectional ranks use only the same date's known features; ranking targets use only training future returns. Ranker scores are not return estimates, so their RMSE/MAE/R²/directional accuracy are undefined, not comparable to regression metrics.

32 model specifications × 2 temporal validation folds × 3 portfolio variants = 192 validation rows, representing 96 candidate strategies. LightGBM and XGBoost each use regression and grouped ranking, depths 3/6 and horizons 1/5/20, 120 trees, learning rate .03 and regularization. LSTM/Transformer use 20/60-session windows and return/centered-rank regression, four epochs maximum, AdamW, weight decay, dropout and validation-only epoch choice. Neural rank regression learns centered percentile targets; it is distinct from the boosting pairwise/LambdaRank losses. Simple momentum/reversal/low-volatility controls are retained separately.

All portfolio candidates are re-evaluated on identical validation date/ticker intersections (`common_validation_coverage.json`), retaining original-boundary scores for audit. Predictive comparisons across **different** horizons are different tasks; portfolio performance uses common dates. Selection maximizes mean validation active CAGR, preferring candidates with positive active CAGR and Rank IC in both folds. Gate status: **{lock['passed_validation_gate']}**. The large search creates selection bias; validation significance is not claimed. If no candidate passes, the best is explicitly exploratory. The final is evaluated once and cannot be recycled into further tuning.

## Selected configuration and validation

`{json.dumps(lock['spec'])}`, portfolio `{lock['portfolio']}`, trial `{lock['trial']}`.

{table(best[['fold','rank_ic','annualized_return','benchmark_annualized','active_annualized','sharpe','annualized_turnover']])}

Every trial and negative result remains in `validation_trials.csv`; all learning curves, checkpoints, predictions and trades are local. The controls and feature IC tables are descriptive validation diagnostics, not post-hoc reasons to suppress outcomes.

## Accounting

Long-only top 20, 95% invested target, 5% cash buffer; fractional shares; sell before buy; fixed shares between rebalances. Horizons correspond to rebalance schedules. Rank buffer keeps incumbents until outside top 40; inverse-volatility is an alternative tested variant. Costs are 15 bp each way on traded notional, charged only on actual position changes, including final liquidation. Dividends accrue when carried into the next ex-date open. Daily equity equals cash plus marked shares. No borrowing, shorts, interest, tax or realistic auction-capacity model. Fixed costs are assumptions, not measured historical spreads.

Reference prices are Yahoo split-adjusted OHLC with auto-adjust disabled and explicit dividends; adjusted-close is not used. Historical provider revisions, split/dividend conventions and corporate-action completeness are not independently verified. Exact reference opening fills and fractional shares remain approximations. Special dividends, spinoffs and delistings need an institutional action ledger for stronger claims.

Final benchmark is the same fixed universe with 95% equal-weight allocation, same rebalance horizon/costs. Fixed momentum+buffer and simple 5-session reversal controls are also reported. These are survivor-basket benchmarks, not S&P total-return indices. This isolates some portfolio-selection effects while retaining market exposure; no risk-adjusted factor alpha claim is made.

## Untouched final results

{final['observations']:,} observations over {final['days']} execution dates: {final['period'][0]} through {final['period'][1]}. Labels must finish by 2025-12-31, so the last horizon's dates are deliberately excluded for all final comparisons.

{table(port[['strategy','annualized_return','total_return','sharpe','max_drawdown','annualized_turnover','transaction_costs','final_equity']])}

Predictive results: `{json.dumps(final['predictive'])}`. Zero and historical-mean comparisons: `{json.dumps(final['baseline_forecasts'])}`. IC/Rank IC are mean daily cross-sectional correlations; ICIR is unannualized mean/sample standard deviation. CAGR uses 252 sessions; Sharpe uses zero risk-free rate; drawdown includes initial capital; turnover is two-sided.

Selected-minus-equal-weight average daily net return ×252: **{active:.2%}**, paired circular 20-session block-bootstrap 95% interval **[{ci[0]:.2%}, {ci[1]:.2%}]**, 5,000 resamples. Resampling whole dates preserves cross-stock dependence; interval is conditional on this selected strategy/universe and fixed block length, not a universal alpha guarantee. The final bootstrap is not used for further selection.

{assessment}

![Final equity and drawdown](figures/final_equity.png)
![Validation turnover](figures/validation_turnover.png)
![All trials](figures/all_trials.png)

## Reproduce locally

Use the root environment, plus `requirements-research-lock.txt`; macOS needs `brew install libomp`. Development workflow:

```bash
python scripts/download_research.py
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python research/search.py
python research/controls.py
python research/diagnostics.py
python research/align_validation.py
python research/select.py
python scripts/download_research.py --final
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python research/final.py
python research/verify.py
python research/report.py
python -m pytest -q
```

Run replication in a separate local copy. Selection/final scripts refuse to overwrite completed locks/results. Keep frozen data hashes and source snapshots: current provider data may revise past prices. Do not interpret a new rerun after seeing this report as a fresh holdout. Local artifacts must not be published without explicit user approval; source-data redistribution rights also need checking before any future publication.

## Sources

[Yahoo download behavior](https://ranaroussi.github.io/yfinance/reference/yfinance.price_history.html), [LightGBM ranking parameters](https://lightgbm.readthedocs.io/en/latest/Parameters.html), [XGBoost grouped ranking](https://xgboost.readthedocs.io/en/stable/tutorials/learning_to_rank.html). The original source is [Kaggle camnugent/sandp500](https://www.kaggle.com/datasets/camnugent/sandp500).
'''
(R/'final_report.md').write_text(text);print('Research report generated')
if (R/'validation_robustness.csv').exists():
 with (R/'final_report.md').open('a') as handle:
  handle.write('\n## Validation-only ablation and seed checks\n\nThese checks do not change the frozen candidate. They test feature dependence on validation only. LightGBM uses full row/feature sampling here; seed changes can still affect histogram construction. Identical runs are not independent replication, and the small seed set is only a sensitivity check.\n\n'+table(pd.read_csv(R/'validation_robustness.csv'))+'\n')
if (R/'final_cost_stress.csv').exists():
 with (R/'final_report.md').open('a') as handle:
  handle.write('\n## Fixed cost stress (descriptive, no reselection)\n\n'+table(pd.read_csv(R/'final_cost_stress.csv')[['strategy','cost_one_way','annualized_return','sharpe','transaction_costs']])+'\n')
controls=pd.read_csv(R/'validation_controls_aligned.csv')
with (R/'final_report.md').open('a') as handle:
 handle.write('\n## Simple controls on aligned validation dates\n\n'+table(controls.groupby(['objective','horizon','portfolio'])[['active_annualized','rank_ic','annualized_turnover']].mean().reset_index())+'\n')
summary=json.loads((R/'validation_selection_summary.json').read_text());chosen=[x for x in summary if x['trial']==lock['trial'] and x['portfolio']==lock['portfolio']][0]
with (R/'final_report.md').open('a') as handle:
 handle.write('\n## Calendar-year stability of the chosen validation strategy\n\n'+table(pd.DataFrame(chosen['year_returns']))+'\n')
if (R/'post_reveal_diagnostics.json').exists():
 extra=json.loads((R/'post_reveal_diagnostics.json').read_text())
 with (R/'final_report.md').open('a') as handle:
  handle.write('\n## Post-reveal descriptive checks — no strategy change\n\nA fixed 95%-invested buy-and-hold basket was calculated after the primary reveal to distinguish scheduled equal-weight rebalancing from literal buy-and-hold. These checks did not change the candidate or its scores.\n\n'+table(pd.DataFrame([extra['buy_hold']]))+'\n\nRank IC block-bootstrap 95% interval: '+str(extra['rank_ic_block_95'])+'. Selected-minus-momentum annualized mean daily return: '+str(extra['selected_minus_momentum_mean_ann'])+', block-bootstrap interval: '+str(extra['selected_minus_momentum_block_95'])+'. These are descriptive additional comparisons, not fresh independent confirmations.\n')
archive=R/'superseded_final_rank_reference/final_results.json'
if archive.exists():
 old=json.loads(archive.read_text())
 with (R/'final_report.md').open('a') as handle:
  handle.write('\n## Transparent post-reveal implementation correction\n\nVerification found that development cross-sectional ranks used 88 downloaded reference names before filtering to 87 tradable names, whereas the first final run used only 87 rank-reference names. Final feature construction was corrected to the frozen 88-name development reference, then filtered to the same 87 trading names. Final-only newly responding ticker strings were excluded. No model specification, selection, threshold, horizon or portfolio rule changed. The first reveal is retained under `superseded_final_rank_reference/`. The corrected calculation is a mechanical repair after inspection, not a second untouched experiment; it cannot support a claim of repeated independent confirmation. A future truly independent sample is still needed.\n\n')
  handle.write(table(pd.DataFrame([{'version':'first reveal (rank-reference bug)',**old['portfolio']['selected']},{'version':'corrected consistent reference',**final['portfolio']['selected']}]))+'\n')
