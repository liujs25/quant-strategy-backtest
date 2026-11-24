from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parent;final=json.loads((R/'final_results.json').read_text());lock=final['lock'];dev=pd.read_csv(R/'development_results.csv');old=json.loads((R/'prior_strategy_factor_regression.json').read_text());fig=R/'figures';fig.mkdir(exist_ok=True)
def table(df):
 lines=['| '+' | '.join(df.columns)+' |','|'+'|'.join(['---']*len(df.columns))+'|']
 for row in df.itertuples(index=False,name=None):lines.append('| '+' | '.join('undefined' if pd.isna(v) else f'{v:.6g}' if isinstance(v,(float,np.floating)) else str(v) for v in row)+' |')
 return '\n'.join(lines)
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
def save(name):
 for ext in ['png','pdf']:plt.savefig(fig/f'{name}.{ext}',dpi=300,bbox_inches='tight')
 plt.close()
f,axes=plt.subplots(2,1,figsize=(11,7),sharex=True,layout='constrained')
for name in [lock['strategy'],'momentum','champion','equal_weight']:
 d=pd.read_csv(R/f'final_{name}_daily.csv',parse_dates=['date']);eq=d.equity.to_numpy();dd=eq/np.maximum.accumulate(np.r_[100000,eq])[1:]-1
 axes[0].plot(d.date,eq/100000,label=name);axes[1].plot(d.date,dd*100)
axes[0].set(title='Frozen incremental-alpha challenger: 2026 holdout',ylabel='Net wealth multiple');axes[0].legend(fontsize=8);axes[1].set(ylabel='Drawdown (%)',xlabel='Execution date')
for a in axes:a.grid(alpha=.2)
save('holdout_equity')
f,a=plt.subplots(figsize=(10,5),layout='constrained')
for name,g in dev.groupby('strategy'):a.scatter(g.annualized_turnover,g.sharpe,label=name,alpha=.7)
a.set(xlabel='Annualized two-sided turnover',ylabel='Net Sharpe',title='Bounded development study: all outcomes');a.legend(fontsize=7,ncol=2);a.grid(alpha=.2);save('development_tradeoffs')
prior=pd.read_csv(R/'prior_score_exposures.csv').drop(columns='date').mean();f,a=plt.subplots(figsize=(8,4),layout='constrained');a.barh(prior.index,prior.values);a.axvline(0,c='black',lw=.7);a.set(xlabel='Mean daily correlation',title='Prior strategy: factor overlap and residual Rank IC');save('prior_exposures')
port=pd.DataFrame(final['portfolio']).T.reset_index(names='strategy');selected=dev[(dev.strategy==lock['strategy'])&(dev.risk==lock['risk'])]
inference=pd.DataFrame([{**x,'ci_low':x['ci95'][0],'ci_high':x['ci95'][1]} for x in final['inference']]).drop(columns='ci95')
credible=all(x['ci95'][0]>0 for x in final['inference']);assessment='The frozen challenger shows positive incremental mean returns with intervals above zero under both specified block lengths and comparators, but the short horizon and survivor-conditioned universe still prevent a broad alpha claim.' if credible else 'Statistically credible incremental ML alpha is not established: at least one specified benchmark-relative interval includes zero or is negative. Do not promote a favorable absolute return, Sharpe or one uncertainty estimate into an alpha claim.'
text=f'''# Incremental ML alpha study — local, verified, unpublished

## Outcome

{assessment} The development gate passed: **{lock['gate_passed']}**. The selected implementation is an **exploratory challenger**, not an approved replacement strategy when the gate fails. All prior experiments are preserved; 2024–2025 is explicitly exploratory. No commit or push occurred.

## Why the previous LightGBM strategy was not convincing alpha

On the inspected 2024–2025 period, average daily score-rank correlations with long momentum, medium momentum and volatility were approximately 0.283, 0.191 and 0.533. After cross-sectionally projecting both scores and forward returns off the factor proxies, residual Rank IC was -0.0057. The descriptive daily return regression on the same-universe market and long-short momentum/volatility/reversal proxies explained {old['r2']:.1%} of variation; annualized intercept was {old['annualized_intercept']:.2%}, HAC t={old['intercept_t']:.2f}. These proxies are not official factor series; the regression is retrospective, conditional, and cannot prove causal exposure or independent alpha. Volatility exposure was particularly substantial.

## Focused design, not an unrestricted model search

Fixed LightGBM regression settings: 120 trees, depth 3, 8 leaves, learning rate .03, minimum leaf 200, L2 penalty 10, seed42, one CPU thread. Apple M5 local compute only; these small tree/convex-optimization workloads do not benefit from MPS. No additional neural architecture sweep.

Four target/feature configurations: raw 20-session return; residual return with all features; residual return excluding momentum/trend features; residual return excluding volume/liquidity features. For each training date, regress its future20-session returns on an intercept and **known-at-signal** long momentum, medium momentum, volatility and reversal ranks, using all training stocks. The residual is the training label. Using training future returns to define training labels is legitimate; no evaluation labels enter these regressions or fitted trees. Prediction scores are projected off the same contemporaneous factors without looking at future returns. Removing momentum features does not remove portfolio momentum: residual strategies intentionally tilt a common momentum reference.

Ten predefined signals include equal weight, long momentum, multi-horizon momentum, low volatility, a traditional factor combination, raw ML, three residual tilts and a 50/50 standardized residual/momentum blend. Three predefined risk settings vary quadratic trading inertia (0.1 or1.0) and maximum weight (5% or3%). Eight fitted models total across two development folds; 60 portfolio evaluations. All results and ablations are retained.

Development folds: 2022–2023 and 2024–2025, training labels ending before 2020-10-01 and2022-10-01 respectively. These gaps exceed the longest raw feature history. Both folds are now **research/selection data**, not independent evidence. The prior 2021–2023 searches also mean historical validation scores are affected by research history. A fixed 2026 holdout was declared before retrieval, downloaded only after selection lock, and evaluated without changes to selection or portfolio settings. Final training labels end before2024-10-01. The 2026 holdout is retrospective and under one year, not a prospective live trial.

Selection maximizes mean paired active-return information ratio versus same-risk momentum; a candidate is eligible only if both development folds have positive active mean return, average Sharpe improves, and neither fold worsens drawdown. No candidate passed all conditions; the best predefined challenger is retained rather than relaxing the gate. Locked choice: `{lock['strategy']}`, risk `{lock['risk']}`. Its worst development drawdown difference was {lock['selection']['worst_drawdown_difference']:.4%}. Traditional champion selected independently from development: `{lock['traditional_champion']}`.

{table(selected[['fold','sharpe','annualized_return','max_drawdown','annualized_turnover','rank_ic']])}

## Common risk, factor constraints and execution

Every strategy is long-only, unlevered, at most95% invested, rebalanced every20 sessions, with identical15-bp one-way commission/slippage. Covariance uses only the trailing60 close-return observations through the signal close, annualizes by252, and shrinks25% toward diagonal. Portfolios are scaled down toward10% ex ante annualized volatility, never up with leverage. This is a target, not a realized-volatility or drawdown guarantee. Overnight gaps and covariance estimation error remain.

The optimizer minimizes distance to a score-tilted diversified allocation plus a quadratic penalty relative to its previous unscaled target. The penalty is a **turnover surrogate**, not exact transaction-cost minimization against drifted actual shares. Actual share changes, costs and turnover are calculated in the ledger. Position caps apply at rebalance; weights can drift between rebalances. Same-risk traditional baselines face the same rules. The separately selected traditional champion may use a different member of the same prespecified risk grid; its comparison is disclosed rather than called identical.

Residual tilts preserve the momentum reference's four proxy exposures **before volatility scaling**. Since strategies may receive different volatility multipliers, actual dollar factor exposures are not perfectly identical. Realized exposure files and HAC return regressions document that distinction. There are no sector/value/profitability neutralizations because reliable point-in-time sector/fundamental data are absent. Factor neutrality is limited to these observable price-based proxies and is not full factor neutrality.

Orders use prior-close features and next-open reference prices. Trades sell before buying, obey available cash, retain shares between rebalances, accrue dividends entering the next ex-date open, and liquidate at the last observed open with costs. Zero cash yield, fractional shares, no taxes/borrow or auction impact. Source Yahoo OHLC is split-adjusted with explicit dividends; corporate actions and revisions are not independently validated. Universe remains87 tradable stocks with the same88-name rank reference, including development survivor conditioning; missing holdout security/price data must fail rather than be silently dropped.

## Fresh final evaluation

{final['observations']:,} observations, {final['days']} sessions, {final['period'][0]}–{final['period'][1]}. The requested endpoint was2026-11-20; observations whose20-session label would finish later are excluded consistently for every strategy. This shortens the traded period. No final retuning.

{table(port[['strategy','annualized_return','total_return','sharpe','max_drawdown','annualized_turnover','transaction_costs','final_equity']])}

Score IC/Rank IC/ICIR: `{json.dumps(final['predictive_score_metrics'])}`. These are allocation/ranking scores rather than calibrated20-day return forecasts; raw RMSE/MAE/R² and directional accuracy are deliberately undefined. Baseline raw-return prediction diagnostics are retained in development files and must not be interpreted as fair error comparisons with factor scores.

{table(inference)}

Intervals use paired whole-date circular block bootstrap, 5,000 resamples each, seed42, both20- and60-session blocks. The statistic is average daily net-return difference×252, not CAGR difference. Twenty-session forward-label overlap and regime dependence make naive independent-observation inference invalid. Daily portfolio returns are used for economic inference; block sensitivity is reported, not selected. Less than a year gives few independent long blocks. Selection among30 strategy/risk combinations occurred on development only, but prior research history, universe conditioning and short sample still constrain inference. No risk-free adjustment or official multifactor alpha is claimed.

![Holdout](figures/holdout_equity.png)
![Development](figures/development_tradeoffs.png)
![Prior exposures](figures/prior_exposures.png)

## Verification and reproduction

`verification.json` records checkpoint replay and reconciliation of cash, holdings, costs, dividends, returns and terminal liquidation; factor-neutrality constraints and rebalance caps are tested. `regression_tests.txt` records all regression tests. Source/data hashes and selection-lock hashes preserve provenance. Prior outputs remain in `research/` and `results/`; new active study is isolated in `incremental/`.

From repository root using the existing local environment and `requirements-research-lock.txt`:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python incremental/study.py
python incremental/select.py
python incremental/download_holdout.py
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python incremental/final.py
python incremental/verify.py
python incremental/diagnostics.py
python incremental/report.py
python -m pytest -q
```

Completed lock/final scripts refuse overwrite. Replicate in a separate local copy with original frozen input snapshots and empty new output folders; do not call a rerun after inspecting this report untouched. The final2026 period is now inspected. Any further model selection needs new independent data or prospective observations, not recycling this period until a positive result appears. Nothing is published without explicit user approval.
'''
(R/'final_report.md').write_text(text)
if (R/'final_cost_stress.csv').exists():
 with (R/'final_report.md').open('a') as f:f.write('\n## Prespecified cost stress, no reselection\n\n'+table(pd.read_csv(R/'final_cost_stress.csv')[['strategy','cost_one_way','annualized_return','sharpe','transaction_costs']])+'\n')
with (R/'final_report.md').open('a') as f:
 f.write('\n## Full development ablation table\n\n'+table(dev[['fold','strategy','risk','annualized_return','sharpe','max_drawdown','annualized_turnover']])+'\n')
 f.write('\n## Final descriptive factor-proxy regressions\n\n'+table(pd.DataFrame([{k:v for k,v in row.items() if k not in ['coefficients','hac_standard_errors']} for row in final['factor_regressions']]))+'\n\nFull coefficients are in final_results.json. Intercepts are descriptive conditional residual means, not independently established alpha.\n')
print('Report generated')
chosen=final['portfolio'][lock['strategy']];mom=final['portfolio']['momentum'];ew=final['portfolio']['equal_weight']
with (R/'final_report.md').open('a') as handle:
 handle.write(f'\n## Decision and negative finding\n\nDo not promote the residual ML challenger. It returned {chosen["annualized_return"]:.2%} annualized with Sharpe {chosen["sharpe"]:.2f} and drawdown {chosen["max_drawdown"]:.2%}, versus same-risk momentum {mom["annualized_return"]:.2%}, {mom["sharpe"]:.2f}, {mom["max_drawdown"]:.2%}. Equal weight returned {ew["annualized_return"]:.2%} with Sharpe {ew["sharpe"]:.2f}. The development-selected traditional champion remains the prespecified reference; equal weight\'s favorable holdout is descriptive, not a license to reselect a winner after inspection. Risk constraints and a more disciplined incremental-alpha framework are reproducible engineering improvements, but these observations do not establish an improved ML strategy. The residual no-volume ablation looked strongest on development and failed to add value versus same-risk momentum on fresh data. Keep that failure.\n')
