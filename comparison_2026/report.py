"""Build the publication report and README from verified experiment outputs."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
R=Path(__file__).resolve().parent
m=pd.read_csv(R/'metrics.csv');ci=pd.read_csv(R/'excess_return_intervals.csv');meta=json.loads((R/'results.json').read_text())
fig=R/'figures';fig.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'axes.titleweight':'bold'})
colors={'LightGBM Ranking':'#64819A','Residual ML':'#007F73','Momentum':'#CD8232','Equal Weight':'#343A40'}
curves={}
for name in m.strategy:
    d=pd.read_csv(R/(name.lower().replace(' ','_')+'_daily.csv'),parse_dates=['date'])
    eq=d.equity.to_numpy();dd=eq/np.maximum.accumulate(np.r_[100000,eq])[1:]-1
    dates=pd.DatetimeIndex([d.date.iloc[0]-pd.Timedelta(days=1),*d.date])
    curves[name]=(dates,np.r_[1,eq/100000],np.r_[0,dd])
def plot(ax,kind):
    for name,(dates,wealth,drawdown) in curves.items():
        ax.plot(dates,wealth if kind=='equity' else drawdown,label=name,color=colors[name],lw=2.2 if name=='Residual ML' else 1.5)
    ax.grid(alpha=.17);ax.set_axisbelow(True)
    if kind=='equity':ax.set_ylabel('Net wealth / initial capital')
    else:ax.set_ylabel('Drawdown');ax.yaxis.set_major_formatter(PercentFormatter(1))
def save(f,name):
    for ext in ['png','pdf']:
        metadata={'CreationDate':None,'ModDate':None} if ext=='pdf' else None
        f.savefig(fig/f'{name}.{ext}',dpi=300,bbox_inches='tight',metadata=metadata)
    plt.close(f)
f,ax=plt.subplots(2,1,figsize=(11,7.5),sharex=True,layout='constrained')
plot(ax[0],'equity');plot(ax[1],'drawdown');ax[0].set_title('Unified 2026 comparison | retrospective evaluation',loc='left');ax[0].legend(ncol=2,frameon=False)
ax[1].set_xlabel('Execution date · equity valued at the following open');save(f,'equity_drawdown')
for kind in ['equity','drawdown']:
    f,ax=plt.subplots(figsize=(11,4),layout='constrained');plot(ax,kind);ax.set_title('Net equity' if kind=='equity' else 'Peak-to-trough drawdown',loc='left');ax.legend(ncol=4,frameon=False,fontsize=9)
    ax.set_xlabel('2026 execution date · retrospective comparison');save(f,kind)
def display_table(frame):
    lines=['| Strategy | Net annualized return | Sharpe | Max drawdown | Rank IC | Annual turnover |','|---|---:|---:|---:|---:|---:|']
    for row in frame.itertuples():
        ic='—' if pd.isna(row.rank_ic) else f'{row.rank_ic:.4f}'
        lines.append(f'| {row.strategy} | {row.annualized_return:.2%} | {row.sharpe:.2f} | {row.max_drawdown:.2%} | {ic} | {row.annualized_turnover:.2f}× |')
    return '\n'.join(lines)
table=display_table(m)
r=m.set_index('strategy').loc['Residual ML']
readme=f'''# Quantitative Strategy Research

A reproducible study of cross-sectional equity signals, residual alpha learning and portfolio construction. The unified 2026 comparison evaluates two frozen ML specifications against Momentum and Equal Weight under identical risk, cost and execution assumptions.

**Residual ML achieved {r.annualized_return:.2%} annualized net return and a {r.sharpe:.2f} Sharpe ratio.** Equal Weight had the lowest drawdown. All pairwise excess-return confidence intervals include zero: the realized ML lead is not statistically established superiority.

## Unified 2026 results

**Retrospective comparison—not a fresh holdout.** The sample contains 87 stocks, 17,748 ticker/date observations and 204 execution sessions from January 2 to October 23, 2026. Terminal valuation is October 26; complete 20-session forecast labels extend through November 20.

{table}

Turnover is annualized, two-sided traded notional. Rank IC is the mean daily cross-sectional Spearman correlation against 20-session forward returns; it is undefined for Equal Weight's constant signal.

![Net equity curves](comparison_2026/figures/equity.png)
![Drawdown curves](comparison_2026/figures/drawdown.png)

## Methodology

- **Signals:** LightGBM LambdaRank; Residual ML trained on factor-neutralized returns with volume/liquidity features excluded; 12-minus-1-month Momentum; Equal Weight. Both ML specifications remain frozen. The ranker was refitted once to the same permissible training sample as the residual checkpoint, with labels ending before October 1, 2024.
- **Features and timing:** causal multi-horizon price, volatility, liquidity and cross-sectional features. Signals formed after close trade at the next open. Prediction labels are dividend-inclusive 20-session open-to-open returns.
- **Common portfolio:** long-only, 3% maximum stock weight at rebalance, at most 95% invested, a 10% ex ante volatility target, 20-session rebalancing and identical quadratic turnover penalty. Covariance uses 60 trailing sessions with diagonal shrinkage. Each strategy uses the same allocator; earlier model-specific buffers and exposure-matching allocations are excluded from this comparison.
- **Accounting:** $100,000 initial capital, fractional shares, explicit cash/dividends and terminal liquidation, zero cash yield, and 15 bp costs on each traded side. Rebalance limits can drift between trades; volatility targeting is not a realized-risk guarantee.

## Statistical assessment

Residual ML's annualized mean daily excess return over Momentum is **2.20%**, with a paired 20-session block-bootstrap 95% interval of **[−8.01%, 11.85%]**. Against Equal Weight, it is **2.00% [−3.08%, 8.53%]**. Sixty-session blocks lead to the same qualitative conclusion. These are mean-return differences, not CAGR differences; intervals are pointwise and unadjusted for multiple comparisons.

The universe is survivor-conditioned, corporate-action data is not independently verified, and execution uses reference opens with fixed costs. The short, already-inspected period supports comparison of these implementations, not a claim of persistent alpha. Earlier negative findings and the distinction between initially sealed evaluations and retrospective analyses are retained in the [research history](docs/RESEARCH_HISTORY.md).

## Reproduce and verify

Tested with Python 3.12 on Apple M5. The primary experiment uses CPU; earlier neural models used MPS.

```bash
# macOS: brew install libomp
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements-publication.txt
python -m pytest -q
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python comparison_2026/verify.py
python comparison_2026/report.py
python scripts/verify_publication.py
```

Verification replays checkpoints, recalculates metrics and confidence intervals, and reconciles trades, costs, dividends and equity. Published artifacts support verification without downloading raw market data. [Full reproduction instructions](docs/REPRODUCIBILITY.md) cover frozen model refits, source fingerprints and provider-data revisions.

## Research outputs

| Resource | Contents |
|---|---|
| [Research report](comparison_2026/report.md) | Methods, risk assumptions, full pairwise uncertainty and interpretation |
| [Metrics](comparison_2026/metrics.csv) · [confidence intervals](comparison_2026/excess_return_intervals.csv) | Machine-readable comparison |
| [Verification](comparison_2026/verification.json) | Metric, checkpoint and accounting checks |
| [Vector equity](comparison_2026/figures/equity.pdf) · [vector drawdown](comparison_2026/figures/drawdown.pdf) | Publication-quality figures |
| [Research history](docs/RESEARCH_HISTORY.md) | Audit, baseline, model search, incremental-alpha study and negative findings |

The primary pipeline is in `comparison_2026/`; causal features and portfolio accounting are in `research/engine.py` and `incremental/core.py`. Historical experiment outputs remain in `results/`, `research/` and `incremental/`. Raw provider downloads and rebuildable caches are excluded from version control.
'''
(R.parent/'README.md').write_text(readme)
interval_lines=['| Strategy − comparator | Block length | Annualized mean excess | 95% interval |','|---|---:|---:|---:|']
for row in ci.itertuples():interval_lines.append(f'| {row.strategy} − {row.comparator} | {row.block_sessions} | {row.annualized_mean_excess_return:.2%} | [{row.ci_low:.2%}, {row.ci_high:.2%}] |')
report=f'''# Unified 2026 strategy comparison

## Results

Residual ML achieved **{r.annualized_return:.2%} annualized net return**, **{r.sharpe:.2f} Sharpe** and **{r.max_drawdown:.2%} maximum drawdown**. It led realized return and Sharpe under the common allocator; Equal Weight had the smallest drawdown. All pairwise excess-return intervals include zero at both block lengths.

This is a **retrospective comparison, not a fresh holdout evaluation**. The 2026 observations had been inspected before this comparison. No model hyperparameters, strategy parameters or risk settings were tuned on these results.

{table}

![Equity and drawdown](figures/equity_drawdown.png)

## Sample and information boundaries

All four strategies use 87 tradable stocks and an identical 88-name feature-ranking reference set. There are {meta['evaluation_observations']:,} common ticker/date observations across {meta['sessions']} execution sessions, {meta['execution_period'][0][:10]}–{meta['execution_period'][1][:10]}. Terminal next-open liquidation is {meta['terminal_valuation_date'][:10]}. All 20-session forward labels finish by {meta['target_end_max'][:10]}, so the traded interval ends before the downloaded data endpoint.

Both ML models use the same {meta['training_observations']:,} training observations with label-end dates strictly before 2024-10-01. This cutoff also precedes the longest raw feature history used by evaluation observations. The earlier ranker checkpoint was trained on less recent information; refitting its frozen specification on the residual model's permissible sample removes that training-information mismatch. This is one fixed refit, not new model selection.

## Models and signals

**LightGBM Ranking:** the previously frozen LambdaRank specification, 120 trees, depth 3, 8 leaves, learning rate 0.03, minimum leaf size 200, L2 penalty 10 and seed 42. Labels are per-date forward-return relevance quintiles. All original financial features are retained.

**Residual ML:** the previously frozen no-volume/liquidity feature ablation. Training 20-session returns are projected off an intercept and known-at-signal long momentum, medium momentum, volatility and reversal proxies. A fixed shallow LightGBM regressor learns those residuals. Evaluation scores are projected off contemporaneous factor features without evaluation labels. The existing checkpoint is reused, and its full prediction vector reproduces the previous saved scores exactly.

**Momentum:** the existing 12-minus-1-month momentum rank. **Equal Weight:** a zero cross-sectional signal, producing equal stock weights at rebalance before common volatility scaling.

Model outputs are allocation/ranking scores, not equally calibrated return forecasts. Rank IC measures each pre-allocation signal against the same 20-session dividend-inclusive forward returns. It does not measure portfolio-weight correlation or isolate incremental factor alpha. Equal Weight's Rank IC is undefined. Overlapping 20-session labels are not independent observations.

## Identical portfolio construction

All signals enter the same optimizer, using per-date standardized scores, an equal-weight starting allocation, quadratic inertia 1.0 relative to the previous unscaled target, 3% stock caps at rebalance and at most 95% total investment. A 60-session trailing close-return covariance estimate, annualized by 252 and shrunk 25% toward its diagonal, scales exposure downward toward 10% ex ante volatility. There is no upward leverage or shorting. Rebalance every 20 sessions. Weight limits can drift between rebalances, and common constraints do not imply identical realized exposures or risk.

The trading-inertia penalty is a turnover surrogate against a previous target, not exact transaction-cost minimization against drifted current shares. Actual costs and turnover use executed share changes.

Earlier strategy-specific allocators are not part of this standardized experiment: the original ranker's top-20 buffer and the residual strategy's momentum-reference exposure-matching allocation are replaced by the common allocator. The residual model's score projection remains intact, but no portfolio receives special factor constraints. Thus “original LightGBM Ranking” means its frozen predictive specification under standardized portfolio construction, not its former full trading implementation.

This also explains why Residual ML returned 19.41% annualized in the preceding incremental study and 23.14% here. The former used a momentum-reference allocation with exposure matching. Both outputs and their assumptions remain preserved; neither is silently substituted for the other.

## Execution and metric definitions

Signals formed after close enter at the next reference open. The ledger sells before buying, constrains trades to cash, retains fractional shares between rebalances, credits dividends when carried into the next ex-date open, and liquidates all holdings at the common terminal open. Initial capital is $100,000. Both sides pay 15 bp on traded notional, including terminal sales. Cash yield and risk-free rate are zero.

Annualized net return is terminal-wealth CAGR using 252/N sessions. Sharpe is daily net mean divided by sample standard deviation times sqrt(252). Maximum drawdown includes initial capital. Turnover is two-sided reference notional divided by daily pre-trade NAV, annualized by 252. Rank IC is the mean daily cross-sectional Spearman correlation. Daily NAVs are indexed by execution date and valued at the following open, including the initial and final costs.

## Paired excess-return uncertainty

{chr(10).join(interval_lines)}

The statistic is mean paired daily net-return difference multiplied by 252, **not the difference in CAGR**. Intervals use a circular whole-date moving-block bootstrap with 5,000 replications, seed 42 and both 20- and 60-session blocks. Every pair uses the same resampled dates for a given block length. All six pairs are reported. Intervals are pointwise 95%, not multiplicity-adjusted discovery tests.

Residual ML's advantage over Momentum is 2.20% annualized mean excess return, with intervals [−8.01%, 11.85%] and [−3.72%, 9.16%]. Against Equal Weight, it is 2.00%, with intervals [−3.08%, 8.53%] and [−2.59%, 7.73%]. Neither establishes superiority. Its slightly negative Rank IC alongside a positive portfolio advantage reflects different statistics: all-stock/date ranking versus periodic constrained holdings and exposure paths. Both measurements are retained.

## Limitations and research status

The sample is short and already inspected; long bootstrap blocks contain few effectively independent observations. The fixed universe is survivor-conditioned and lacks verified historical constituent membership and delisting coverage. Source OHLC and explicit dividend data do not constitute an independently validated corporate-action ledger. Reference-open fills, fractional shares and fixed costs omit measured auction impact and capacity constraints. Sector, value and profitability exposures are not neutralized. These assumptions apply equally to all four strategies but limit economic interpretation.

The earlier 2024–2025 and 2026 evaluations were initially sealed until their respective selection locks. They became exploratory after inspection. The 2024–2025 study also records a post-reveal rank-reference repair and retains both calculations. The current comparison is explicitly retrospective; it does not create a new untouched test. See [research history](../docs/RESEARCH_HISTORY.md) for all stages, failed gates, ablations and negative findings.

## Verification and reproduction

The published verifier replays both checkpoints, recomputes all reported metrics and bootstrap intervals, reconciles cash, positions, dividends, costs and terminal wealth, and checks dates and allocation limits. `results.json` records frozen source/specification hashes and sample metadata. Per-strategy predictions, daily equity, trades, weights and IC observations are retained. PNG figures are 300 dpi; PDF versions are vector.

See [reproduction instructions](../docs/REPRODUCIBILITY.md), [verification output](verification.json) and [regression results](regression_tests.txt). The experiment ran locally on Apple M5 CPU; GPU acceleration is not beneficial for these small tree/optimizer workloads. Earlier neural studies used MPS. The completed-run guard prevents accidental overwrite.
'''
(R/'report.md').write_text(report)
print('Generated README, report and publication figures from frozen outputs')
