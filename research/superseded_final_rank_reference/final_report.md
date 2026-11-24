# Expanded local research: diagnosis, validation search and sealed evaluation

**Publication status: LOCAL ONLY. No commit or push.** The verified original redesign is preserved in `../results/` and `../audit/baseline_verified_20261124.zip`. Its inspected 2017–2018 interval is exploratory. See `baseline_diagnosis.md` for the signal, feature, fit and cost diagnosis.

## Experiment design

All work ran locally on Apple M5. PyTorch used MPS; LightGBM/XGBoost used single-thread CPU after a local OpenMP crash before the first completed trial. Native dependencies and exact package versions are recorded. Data is Yahoo Finance through yfinance, development requests ending strictly before 2024. Final data was downloaded only after `selection_lock.json` was written. It is a newly untouched **retrospective** holdout, not a prospective live experiment or unknown market history.

Universe: top 100 original Kaggle names by 2016 median dollar volume. Provider returned 88, and 87 met development-history/completeness rules. Unavailable: FB, AGN, PCLN, CELG, UTX, WBA, TWX, PXD, ATVI, MON, APC, ESRX. Development exclusions: `[{"symbol": "PYPL", "reason": "incomplete development validation calendar or insufficient pre-2015 history", "first": "2015-07-06 00:00:00", "last": "2023-12-29 00:00:00"}]`. Conditioning on current provider availability and complete 2021–2023 history adds survivorship bias. This is explicitly a survivor-conditioned study, not a historical S&P membership backtest. Final names are frozen and never substituted based on future performance. Final missing names/quotes fail loudly.

Training precedes 2021 validation by a conservative cutoff (labels end before 2019-10-01); second-fold training labels end before 2020-10-01, with validation 2022–2023. These gaps exceed the 252-session feature lookback plus 60-session sequence length. Final training labels end before 2022-10-01; evaluation is 2024–2025. No train/evaluation raw-history overlap under the recorded market calendar; assertions and a split audit document the boundary. No final hyperparameter or portfolio tuning. The strategy universe is retrospective and not made unbiased by these temporal safeguards.

55 features include 1/5/10/20/60/120/252-session momentum, 12-minus-1-month momentum, multi-scale volatility, moving-average distance, volume ratios, ranges, intraday/overnight changes, close location, illiquidity, relative momentum and per-date ranks. Features use information only through the prior close. Zero-range close location is neutral zero; the validation bug and superseded trials are retained. Neural inputs use 10 rank/market features and layer normalization; no future-fitted scaler.

Targets are dividend-inclusive open-to-open returns over 1, 5 or 20 sessions, entering after the signal close. Horizons deliberately differ from the original intraday baseline. Multi-day labels overlap **within** folds; their uncertainty cannot be treated as independent observations. Cross-sectional ranks use only the same date's known features; ranking targets use only training future returns. Ranker scores are not return estimates, so their RMSE/MAE/R²/directional accuracy are undefined, not comparable to regression metrics.

32 model specifications × 2 temporal validation folds × 3 portfolio variants = 192 validation rows, representing 96 candidate strategies. LightGBM and XGBoost each use regression and grouped ranking, depths 3/6 and horizons 1/5/20, 120 trees, learning rate .03 and regularization. LSTM/Transformer use 20/60-session windows and return/centered-rank regression, four epochs maximum, AdamW, weight decay, dropout and validation-only epoch choice. Neural rank regression learns centered percentile targets; it is distinct from the boosting pairwise/LambdaRank losses. Simple momentum/reversal/low-volatility controls are retained separately.

All portfolio candidates are re-evaluated on identical validation date/ticker intersections (`common_validation_coverage.json`), retaining original-boundary scores for audit. Predictive comparisons across **different** horizons are different tasks; portfolio performance uses common dates. Selection maximizes mean validation active CAGR, preferring candidates with positive active CAGR and Rank IC in both folds. Gate status: **True**. The large search creates selection bias; validation significance is not claimed. If no candidate passes, the best is explicitly exploratory. The final is evaluated once and cannot be recycled into further tuning.

## Selected configuration and validation

`{"family": "LightGBM", "objective": "ranking", "horizon": 20, "depth": 3}`, portfolio `buffer`, trial `trial_10`.

| fold | rank_ic | annualized_return | benchmark_annualized | active_annualized | sharpe | annualized_turnover |
|---|---|---|---|---|---|---|
| 0 | 0.020634 | 0.44578 | 0.26099 | 0.18479 | 1.6932 | 3.6415 |
| 1 | 0.042049 | 0.13126 | 0.039132 | 0.092131 | 0.60543 | 2.2696 |

Every trial and negative result remains in `validation_trials.csv`; all learning curves, checkpoints, predictions and trades are local. The controls and feature IC tables are descriptive validation diagnostics, not post-hoc reasons to suppress outcomes.

## Accounting

Long-only top 20, 95% invested target, 5% cash buffer; fractional shares; sell before buy; fixed shares between rebalances. Horizons correspond to rebalance schedules. Rank buffer keeps incumbents until outside top 40; inverse-volatility is an alternative tested variant. Costs are 15 bp each way on traded notional, charged only on actual position changes, including final liquidation. Dividends accrue when carried into the next ex-date open. Daily equity equals cash plus marked shares. No borrowing, shorts, interest, tax or realistic auction-capacity model. Fixed costs are assumptions, not measured historical spreads.

Reference prices are Yahoo split-adjusted OHLC with auto-adjust disabled and explicit dividends; adjusted-close is not used. Historical provider revisions, split/dividend conventions and corporate-action completeness are not independently verified. Exact reference opening fills and fractional shares remain approximations. Special dividends, spinoffs and delistings need an institutional action ledger for stronger claims.

Final benchmark is the same fixed universe with 95% equal-weight allocation, same rebalance horizon/costs. Fixed momentum+buffer and simple 5-session reversal controls are also reported. These are survivor-basket benchmarks, not S&P total-return indices. This isolates some portfolio-selection effects while retaining market exposure; no risk-adjusted factor alpha claim is made.

## Untouched final results

41,934 observations over 482 execution dates: 2024-01-02 00:00:00 through 2025-12-02 00:00:00. Labels must finish by 2025-12-31, so the last horizon's dates are deliberately excluded for all final comparisons.

| strategy | annualized_return | total_return | sharpe | max_drawdown | annualized_turnover | transaction_costs | final_equity |
|---|---|---|---|---|---|---|---|
| selected | 0.26329 | 0.56368 | 1.1293 | 0.24276 | 2.7581 | 1022.9 | 1.5637e+05 |
| equal_weight | 0.17934 | 0.37096 | 1.1985 | 0.1651 | 1.6602 | 564.38 | 1.371e+05 |
| momentum | 0.24307 | 0.51616 | 1.1671 | 0.23081 | 3.0736 | 1132.4 | 1.5162e+05 |
| reversal | 0.023923 | 0.046257 | 0.22269 | 0.20159 | 19.568 | 5869.8 | 1.0463e+05 |

Predictive results: `{"n": 41934, "rmse": null, "mae": null, "r2": null, "directional_accuracy": null, "ic": 0.06270767940298749, "rank_ic": 0.04990257707234148, "icir": 0.28849684458130437, "ic_valid_days": 482}`. Zero and historical-mean comparisons: `{"zero": {"n": 41934, "rmse": 0.08490903142169863, "mae": 0.0628139943856389, "r2": -0.032094297566896746, "directional_accuracy": 0.00019077598130395383, "ic": null, "rank_ic": null, "icir": null, "ic_valid_days": 0}, "historical_mean": {"n": 41934, "rmse": 0.08361649915376418, "mae": 0.06201049535599114, "r2": -0.0009112395980206411, "directional_accuracy": 0.5695139981876282, "ic": null, "rank_ic": null, "icir": null, "ic_valid_days": 0}}`. IC/Rank IC are mean daily cross-sectional correlations; ICIR is unannualized mean/sample standard deviation. CAGR uses 252 sessions; Sharpe uses zero risk-free rate; drawdown includes initial capital; turnover is two-sided.

Selected-minus-equal-weight average daily net return ×252: **8.46%**, paired circular 20-session block-bootstrap 95% interval **[-6.97%, 22.53%]**, 5,000 resamples. Resampling whole dates preserves cross-stock dependence; interval is conditional on this selected strategy/universe and fixed block length, not a universal alpha guarantee. The final bootstrap is not used for further selection.

The final evaluation does not establish reliable benchmark-relative economic value: the paired block-bootstrap interval does not lie wholly above zero. Positive absolute returns or a winning validation score do not establish alpha.

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

## Validation-only ablation and seed checks

These checks do not change the frozen candidate. They test feature dependence on validation only. LightGBM uses full row/feature sampling here; seed changes can still affect histogram construction. Identical runs are not independent replication, and the small seed set is only a sensitivity check.

| fold | ablation | rank_ic | annualized_return | total_return | sharpe | max_drawdown | mean_daily_turnover | annualized_turnover | transaction_costs | final_equity | days |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | full_seed7 | 0.020634 | 0.44578 | 0.4041 | 1.6932 | 0.10264 | 0.01445 | 3.6415 | 611.9 | 1.4041e+05 | 232 |
| 0 | full_seed99 | 0.020634 | 0.44578 | 0.4041 | 1.6932 | 0.10264 | 0.01445 | 3.6415 | 611.9 | 1.4041e+05 | 232 |
| 0 | short_only | -0.009209 | 0.31798 | 0.28941 | 1.209 | 0.14271 | 0.01628 | 4.1027 | 666.81 | 1.2894e+05 | 232 |
| 0 | no_cross_section | 0.0083664 | 0.43007 | 0.39004 | 1.7598 | 0.086559 | 0.014984 | 3.7759 | 629.43 | 1.39e+05 | 232 |
| 1 | full_seed7 | 0.040761 | 0.12485 | 0.25178 | 0.58065 | 0.23864 | 0.0090154 | 2.2719 | 706.58 | 1.2518e+05 | 481 |
| 1 | full_seed99 | 0.043386 | 0.13088 | 0.26462 | 0.60978 | 0.24861 | 0.009356 | 2.3577 | 744.87 | 1.2646e+05 | 481 |
| 1 | short_only | 0.040543 | 0.091125 | 0.18111 | 0.4581 | 0.24215 | 0.014018 | 3.5324 | 1060.8 | 1.1811e+05 | 481 |
| 1 | no_cross_section | 0.040134 | 0.14601 | 0.2971 | 0.68222 | 0.23048 | 0.013426 | 3.3834 | 1073.6 | 1.2971e+05 | 481 |

## Fixed cost stress (descriptive, no reselection)

| strategy | cost_one_way | annualized_return | sharpe | transaction_costs |
|---|---|---|---|---|
| selected | 0 | 0.2684 | 1.1469 | 0 |
| equal_weight | 0 | 0.18225 | 1.2153 | 0 |
| reversal | 0 | 0.054343 | 0.39097 | 0 |
| selected | 0.0005 | 0.26669 | 1.1411 | 341.81 |
| equal_weight | 0.0005 | 0.18128 | 1.2097 | 188.38 |
| reversal | 0.0005 | 0.044118 | 0.33488 | 1990.5 |
| selected | 0.0015 | 0.26329 | 1.1293 | 1022.9 |
| equal_weight | 0.0015 | 0.17934 | 1.1985 | 564.38 |
| reversal | 0.0015 | 0.023923 | 0.22269 | 5869.8 |
| selected | 0.003 | 0.25821 | 1.1117 | 2038.1 |
| equal_weight | 0.003 | 0.17644 | 1.1816 | 1126.5 |
| reversal | 0.003 | -0.0057365 | 0.054889 | 11442 |

## Simple controls on aligned validation dates

| objective | horizon | portfolio | active_annualized | rank_ic | annualized_turnover |
|---|---|---|---|---|---|
| low_vol | 1 | buffer | -0.072537 | -0.016533 | 9.6031 |
| low_vol | 1 | equal | -0.17139 | -0.016533 | 41.946 |
| low_vol | 1 | inverse_vol | -0.17419 | -0.016533 | 43.657 |
| low_vol | 5 | buffer | -0.06932 | -0.019621 | 7.3829 |
| low_vol | 5 | equal | -0.11959 | -0.019621 | 21.759 |
| low_vol | 5 | inverse_vol | -0.11617 | -0.019621 | 23.002 |
| low_vol | 20 | buffer | -0.044361 | -0.024922 | 5.3574 |
| low_vol | 20 | equal | -0.041951 | -0.024922 | 11.732 |
| low_vol | 20 | inverse_vol | -0.043405 | -0.024922 | 12.237 |
| momentum | 1 | buffer | -0.050402 | 0.0078323 | 7.0903 |
| momentum | 1 | equal | -0.080939 | 0.0078323 | 32.408 |
| momentum | 1 | inverse_vol | -0.12391 | 0.0078323 | 44.423 |
| momentum | 5 | buffer | -0.011709 | -0.013535 | 5.134 |
| momentum | 5 | equal | -0.052304 | -0.013535 | 14.526 |
| momentum | 5 | inverse_vol | -0.091312 | -0.013535 | 19.674 |
| momentum | 20 | buffer | 0.0094532 | -0.030636 | 4.2077 |
| momentum | 20 | equal | -0.03124 | -0.030636 | 7.6745 |
| momentum | 20 | inverse_vol | -0.050867 | -0.030636 | 9.8647 |
| reversal | 1 | buffer | -0.19746 | 0.0032945 | 83.871 |
| reversal | 1 | equal | -0.26618 | 0.0032945 | 160.21 |
| reversal | 1 | inverse_vol | -0.29853 | 0.0032945 | 179.18 |
| reversal | 5 | buffer | -0.054737 | -0.0081547 | 54.03 |
| reversal | 5 | equal | -0.099565 | -0.0081547 | 74.619 |
| reversal | 5 | inverse_vol | -0.12617 | -0.0081547 | 77.414 |
| reversal | 20 | buffer | 0.089036 | -0.0053559 | 14.059 |
| reversal | 20 | equal | 0.10931 | -0.0053559 | 19.24 |
| reversal | 20 | inverse_vol | 0.080348 | -0.0053559 | 20.088 |

## Calendar-year stability of the chosen validation strategy

| year | net_return | benchmark_return | active_return |
|---|---|---|---|
| 2021 | 0.4041 | 0.23799 | 0.1661 |
| 2022 | 0.010351 | -0.03913 | 0.049481 |
| 2023 | 0.25247 | 0.11984 | 0.13263 |

## Post-reveal descriptive checks — no strategy change

A fixed 95%-invested buy-and-hold basket was calculated after the primary reveal to distinguish scheduled equal-weight rebalancing from literal buy-and-hold. These checks did not change the candidate or its scores.

| annualized_return | total_return | sharpe | max_drawdown | mean_daily_turnover | annualized_turnover | transaction_costs | final_equity | days |
|---|---|---|---|---|---|---|---|---|
| 0.18705 | 0.38815 | 1.2407 | 0.16911 | 0.0039042 | 0.98385 | 336.93 | 1.3882e+05 | 482 |

Rank IC block-bootstrap 95% interval: [-0.019019284353203207, 0.11445267845850823]. Selected-minus-momentum annualized mean daily return: 0.02185670767272416, block-bootstrap interval: [-0.08840528905092279, 0.1314869906472823]. These are descriptive additional comparisons, not fresh independent confirmations.
