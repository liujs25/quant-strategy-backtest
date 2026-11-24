"""Generate tables and vector/300-dpi figures from saved outputs only."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path(__file__).resolve().parents[1];r=root/'results';fig=r/'figures';fig.mkdir(exist_ok=True)
m=pd.read_csv(r/'model_comparison.csv');p=pd.read_csv(r/'portfolio_comparison.csv');test=m[m.split=='test'];port=p[p.split=='test']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.bbox':'tight','pdf.fonttype':42})
colors=dict(zip(['zero','historical_mean','ARIMA','RandomForest','LSTM','Transformer','buy_hold'],['#555555','#999999','#009E73','#E69F00','#0072B2','#CC79A7','#000000']))
def save(name):
    plt.savefig(fig/(name+'.png'),dpi=300);plt.savefig(fig/(name+'.pdf'));plt.close()
f,ax=plt.subplots(2,1,figsize=(10,7),sharex=True,layout='constrained')
for name in port.model:
    d=pd.read_csv(r/f'portfolio_test_{name}.csv',parse_dates=['date']);eq=d.equity.to_numpy();dd=eq/np.maximum.accumulate(np.r_[100000,eq])[1:]-1
    ax[0].plot(d.date,eq/100000,label=name,color=colors[name],linewidth=1.5);ax[1].plot(d.date,dd*100,color=colors[name])
ax[0].set(ylabel='Net wealth / initial wealth',title='Untuned walk-forward test: January 2017–February 2018');ax[0].legend(ncol=3,fontsize=9);ax[1].set(ylabel='Drawdown (%)',xlabel='Trading date')
for a in ax:a.grid(alpha=.2)
save('equity_drawdown')
f,ax=plt.subplots(1,3,figsize=(12,4),layout='constrained')
for a,key,title in zip(ax,['rmse','r2','rank_ic'],['RMSE (return units)','Out-of-sample R²','Mean daily Rank IC']):
    a.bar(test.model,test[key],color=[colors[n] for n in test.model]);a.axhline(0,color='black',lw=.6);a.set_title(title);a.tick_params(axis='x',rotation=55);a.grid(axis='y',alpha=.2)
save('predictive_metrics')
h=pd.DataFrame(json.loads((r/'training_history.json').read_text()));f,ax=plt.subplots(1,2,figsize=(10,4),layout='constrained')
for a,name in zip(ax,['LSTM','Transformer']):
    s=h[(h.fold=='validation')&(h.model==name)];a.plot(s.epoch,s.train_mse,label='Training');a.plot(s.epoch,s.validation_mse,label='Validation');a.set(title=name,xlabel='Epoch',ylabel='MSE (return units)');a.legend();a.grid(alpha=.2)
save('validation_learning_curves')
c=pd.read_csv(r/'cost_sensitivity.csv');f,a=plt.subplots(figsize=(8,4),layout='constrained')
for name in test.model:
    s=c[c.model==name];a.plot(s.one_way_cost*10000,s.annualized_return*100,'o-',label=name,color=colors[name])
a.set(xlabel='One-way cost (basis points)',ylabel='Annualized net return (%)',title='Prespecified cost stress test — no strategy reselection');a.legend(ncol=3);a.grid(alpha=.2);save('cost_sensitivity')
old={'LSTM':(.0124,.0087,.248),'Transformer':(.0131,.0092,.221),'RandomForest':(.0143,.0099,.185),'ARIMA':(.0152,.0105,.152)}
comparison=[]
for row in test.itertuples():
    previous=old.get(row.model,(None,None,None));q=port[port.model==row.model].iloc[0]
    comparison.append({'model':row.model,'old_rmse_unverified':previous[0],'new_rmse':row.rmse,'old_r2_unverified':previous[2],'new_r2':row.r2,
        'old_annual_return_unverified':.182 if row.model=='LSTM' else None,'new_annual_return':q.annualized_return,
        'old_sharpe_unverified':1.4 if row.model=='LSTM' else None,'new_sharpe':q.sharpe})
pd.DataFrame(comparison).to_csv(r/'before_after.csv',index=False)
def table(df):
    columns=list(df.columns);lines=['| '+' | '.join(columns)+' |','|'+'|'.join(['---']*len(columns))+'|']
    for row in df.itertuples(index=False,name=None):
        lines.append('| '+' | '.join('undefined' if pd.isna(v) else f'{v:.6g}' if isinstance(v,(float,np.floating)) else str(v) for v in row)+' |')
    return '\n'.join(lines)
u=pd.read_csv(r/'uncertainty.csv');manifest=json.loads((r/'run_manifest.json').read_text());splits=pd.DataFrame(json.loads((r/'split_audit.json').read_text()));quality=json.loads((r/'data_quality.json').read_text());selection=json.loads((r/'selection_frozen.json').read_text())
text=f'''# Reproduction, audit and real-data experiment

## Scope and provenance

Upstream commit: `9b143e407ed878e0694564f953149d0456177433`. Original Kaggle `camnugent/sandp500`, member `all_stocks_5yr.csv`, downloaded 2026-11-24. SHA-256: `{manifest['data_sha256']}`. {quality['rows']:,} rows, {quality['symbols']} names, 2013-02-08 through 2018-02-07. Hardware backend: **{manifest['device']}**, PyTorch {manifest['torch']}, Python {manifest['python']}. Full experiment wall time: {manifest['seconds']/60:.2f} minutes. No synthetic financial observations or results are used. Tests use clearly isolated toy fixtures.

Historical checkpoints are literal placeholder text; original trades/curve occur in 2022–2023 and cannot derive from the stated dataset. See `audit/AUDIT.md` and its immutable archive for the file-level findings. Original claims are withdrawn. The new target, architectures and protocol differ, so before/after is not a controlled performance comparison.

## Frozen design and information clock

Features are available after close on day t. The model predicts `close[t+1]/open[t+1]-1`; orders are selected after close t, bought at open t+1 and sold at that day's close. Closing exits are scheduled in advance. No current-session feature, overnight return before entry, stop-loss hindsight or same-close signal fill is used.

All 505 source names are allowed, subject to 40 sessions of complete raw history. Eight features (1/5/20-day returns, intraday return, high-low range, overnight gap, 20-day volatility and volume ratio) use only past/current signal-day values. No outcome clipping or return outlier deletion. Missing OHLC rows are not imputed. The source calendar is reindexed for each name so gaps cannot create fictitious adjacent days. Invalid/absent windows: {quality['invalid_windows']:,}; initial 40 calendar sessions per name are warmup and counted separately. These exclusions include pre-IPO absence, gaps and rolling warmup after missing data, not just bad original rows. Prediction files contain every eligible evaluation observation.

Each boundary uses a strict purge: the latest training target precedes the earliest raw observation contributing to the evaluation window. Overlap **within** a chronological segment is ordinary causal history reuse; train/evaluation windows have no raw-history overlap. Whole dates remain together. The scaler fits training windows only. Overlapping training windows reweight repeated historical feature rows, an explicit weighting choice. There is no validation/test normalization fit.

{table(splits)}

Validation is 2016. It chooses only the neural epoch (1–6) and forest depth (6 or 10). All test folds use the frozen choices. Later test refits may consume earlier observed test labels after the same purge; this is prequential walk-forward training, never hyperparameter or strategy selection. Test results are descriptive, never fed back into design. The interrupted pre-test validation setup is preserved in `audit/interrupted_validation`.

Selected configurations: `{json.dumps(selection['selections'])}`.

LSTM: one 32-unit layer, linear output. Transformer: one 32-dimensional encoder, four heads, feedforward width 64, dropout 0.1, sinusoidal positions. Both use 20 days, batch 1024, Adam 0.001, clipped gradient norm 1, seed 42, targets scaled by 100 for optimization then inverted. Full past-window attention is valid because every token precedes execution. Forest: 80 trees, sqrt feature subsampling, minimum leaf 100; flattened identical input windows. Models are intentionally modest reproducible baselines, not exhaustive architecture searches. No claim of reproducing the obsolete nominal 128/256-wide architectures.

ARIMA(1,0,0) is fitted per stock on percent intraday returns with missing sessions retained. Parameters are frozen within each fold; next-day forecasts update using the latest observed intraday return. Parameter checkpoints include constant, AR coefficient, variance, convergence and warnings. A documented pooled-mean fallback applies below 100 training observations. Historical mean is the pooled training-target mean held fixed within each fold; zero predicts exactly zero. All six models have identical evaluation observations, including fallback cases.

## Predictive results

{table(test[['model','n','rmse','mae','r2','directional_accuracy','ic','rank_ic','icir','ic_valid_days']])}

RMSE/MAE use decimal returns. R² is `1-SSE/SST` on the common pooled target. Directional accuracy compares three-way signs (-,0,+), so zero's score counts only exact zero returns; it is not a random 50% directional classifier. IC is mean **daily cross-sectional** Pearson correlation; Rank IC uses average-rank Spearman correlation. ICIR is daily IC mean/sample standard deviation, unannualized. Constant cross-sectional forecasts have undefined IC/Rank IC/ICIR, represented as null/blank, not zero. Their valid-day count is zero. These quantities differ from a pooled time-series correlation.

{table(u)}

The uncertainty intervals use a paired circular moving-block bootstrap of daily average squared-error improvement over zero, block length 20, 2,000 draws, seed 42. Positive improvement favors the model. This accounts approximately for temporal dependence and cross-stock dependence by resampling whole dates. Intervals are unadjusted for multiple models and block-length uncertainty; they are not proof of tradable alpha. No test-based selection is made from them.

## Execution and portfolio results

Fixed long-only rule: rank predictions descending, ticker as deterministic tie break, buy up to 20 whose predicted return is strictly above 0.003. Each receives at most 1/20 starting NAV including entry costs; unfilled slots remain cash. Fractional shares, no leverage or shorts. Both legs pay 10 bp commission plus 5 bp slippage on reference notional (15 bp one way). Cash earns zero. Starting capital $100,000. Every day ends flat; all trades include both legs and reconcile to NAV. The costs are plausible research assumptions, not measured historical spreads or market impact.

Benchmark: buy and hold the 20 stocks with highest median 2016 dollar volume, equal initial dollars, selected before testing. Buy first test open and sell final test close with the same costs. Fixed shares drift naturally. It is an investable source-data basket under these assumptions, **not** the S&P 500 total-return index; it has overnight exposure and concentration differences from the intraday strategy. Missing intermediate marks are forward-valued, but missing boundary execution quotes fail explicitly. Basket is saved in `benchmark_basket.json`.

{table(port[['model','annualized_return','sharpe','max_drawdown','mean_daily_turnover','annualized_turnover','transaction_costs','final_equity','days']])}

Annualized return is terminal wealth CAGR using 252/N sessions; Sharpe is mean daily net return/sample standard deviation times sqrt(252), risk-free rate zero. Maximum drawdown includes initial NAV. Turnover is **two-sided** traded reference notional/starting daily NAV (a full round trip is approximately 2); annualized turnover is daily mean times 252. Transaction costs are cumulative dollars on the compounding equity path. Cash strategy Sharpe is undefined. The final 2018 fold has only a few weeks: its annualized statistics are unstable. All per-fold results are retained, as are zero-, 5-, 15- and 30-bp cost scenarios; none was selected for favorable test performance.

![Equity and drawdown](figures/equity_drawdown.png)
![Prediction results](figures/predictive_metrics.png)
![Validation curves](figures/validation_learning_curves.png)
![Cost stress](figures/cost_sensitivity.png)

## Before versus after

{table(pd.DataFrame(comparison))}

The old RMSE/R² pairs imply incompatible target variances and cannot substantiate a fair comparison. No inference speed or training duration is copied from them. Measured durations for the new fits are in `timings.json`.

## Limitations and scientific interpretation

This is a retrospective survivor-biased universe, not point-in-time S&P membership. Failed/delisted constituents and delisting returns are absent. Selecting from these 505 names already embeds hindsight that chronological ML splits cannot repair. Corporate-action adjustment conventions are not established by this OHLCV file; there is no adjusted-close/dividend series, security identifier history, split event table or ticker-change map. The buy-and-hold calculation is therefore a source-price return, not verified total shareholder return. Intraday labels reduce overnight corporate-action discontinuities but do not solve universe bias or all price-history issues.

Missing execution data cannot be imputed into genuine fills. Eligible-sample comparison excludes unavailable targets; the exclusion ledger records these, and missingness may not be random. The dataset has no auction depth, bid/ask, halts, order rejects or realized impact. Fractional shares and exact reference open/close executions with a fixed cost penalty are approximations. No sector constraints, borrow, tax, dividends or capacity claim is made. A 13-month test dominated by 2017 and one seed cannot establish regime robustness. The test is now inspected; further tuning requires a new untouched period/dataset.

Assess genuine value by improvement over zero **and** historical mean on the common observations, uncertainty and net profitability against the basket. Positive isolated IC, raw return or a favorable fold is insufficient. The numerical results above are retained regardless of sign. See the concise assessment at the end for the observed outcome.

## Reproducibility and artifacts

Run commands in the root README. Checkpoints, scalers, fitted ARIMA coefficients and frozen selections reproduce saved forecasts; `scripts/verify_artifacts.py` checks replay samples and full-ledger identities. Seed and exact package lock are recorded; GPU libraries can cause small cross-platform floating-point differences. For deterministic CPU reference reruns, use `CUDA_VISIBLE_DEVICES=''` plus a local CPU backend selection only in a new output experiment and record that change; do not claim bitwise GPU equivalence. The provided run uses the available GPU automatically.

All original files remain solely in the audit archive. Active results are generated by real fits. Model, trade, prediction, per-day IC, per-fold score and cost-stress files are retained; undefined statistics are explicitly null/blank. Do not load arbitrary untrusted joblib/pickle files.
'''
(r/'final_report.md').write_text(text)
print('Report and figures generated')
trained=test[test.model.isin(['ARIMA','RandomForest','LSTM','Transformer'])]
base_rmse=test[test.model.isin(['zero','historical_mean'])].rmse.min()
all_worse=bool((trained.rmse>base_rmse).all())
negative_net=bool((port[port.model.isin(trained.model)].annualized_return<0).all())
summary='No fitted model beats the better simple baseline in RMSE.' if all_worse else 'Some fitted models improve RMSE; inspect paired uncertainty before inferring a reliable gain.'
summary+=' All four model strategies lose money after the fixed costs.' if negative_net else 'Net strategy outcomes vary; inspect costs, uncertainty and benchmark exposure.'
assessment='\n## Verified scientific assessment\n\n'+summary+' The tabled results provide no basis for claiming robust predictive or trading alpha from these specific model/rule combinations. Small positive cross-sectional ICs alone are insufficient. Negative R², uncertainty, cash-baseline outcomes and net profitability must be considered together. This does not prove every possible architecture or strategy must fail.\n\n'
assessment+='At 15 bp per leg, '+', '.join(f"{row.model}: {row.annualized_return:.2%} annualized, Sharpe {row.sharpe:.2f}" for row in port.itertuples() if row.model in ['ARIMA','RandomForest','LSTM','Transformer','buy_hold'])+'. The benchmark remains subject to survivor and source-price limitations.\n\n'
for label in ['validation','test_1','test_2','test_3']:
    statuses=json.loads((r/'models'/f'{label}_ARIMA.json').read_text())['status']
    assessment+=f"{label}: {sum(s['status']!='fit' for s in statuses)} explicit insufficient-history ARIMA fallbacks, {sum(not s.get('converged',True) for s in statuses)} nonconverged fits, {sum(len(s.get('warnings',[])) for s in statuses)} warnings. "
assessment+='Fallback observations remain in common-sample evaluation. One seed, a short test horizon, universe bias and unverified corporate-action conventions limit inference. A new point-in-time, dividend-aware dataset and untouched test period are needed for further selection or deployment claims.\n'
with (r/'final_report.md').open('a') as handle:handle.write(assessment)
