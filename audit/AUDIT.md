# Audit of upstream 9b143e407ed878e0694564f953149d0456177433

The original files are preserved byte-for-byte in `original_unverified.zip`, with SHA-256 hashes in `original_manifest.json`. They are evidence, not usable results. All original source modules, both notebooks, tests, requirements, README, CSV/JSON artifacts and report were inspected. All five historical plots were visually inspected and checked against their underlying CSV and claims. Training plots show MSE/RMSE scales that are not mutually consistent; importance plots include features absent from the implemented feature list. The replacement deliberately breaks the unsafe positional-array API.

## Confirmed defects and unsupported claims

| Location in archived version | Finding | Repair |
|---|---|---|
| `results/models/LSTM_best.pth`, `Transformer_best.pth` | 33- and 40-byte literal text placeholders, not PyTorch weights | Genuine trained state dictionaries for validation and each walk-forward fold |
| Modeling notebook cells 3–4 | Test tensors passed as validation for early stopping and scheduling | Dedicated 2016 validation; epochs frozen before test |
| Modeling notebook cell 6 | Pooled predictions paired with a tail slice of ticker-grouped prices; min-length truncation hides mismatch | Explicit symbol, feature start, signal date, execution date and prices on every prediction |
| `data_preprocessing.py` | Row-count split can divide one date across train/test; metadata lost; overlapping windows not purged; no distinct validation | Whole-date boundaries and 40-session raw-history purge |
| `data_preprocessing.py` | Forward-fills missing execution prices; drops rows before windows, compressing gaps; no duplicate-key rejection | Calendar reindexing, invalid windows excluded and counted; no price fills; fail on duplicate keys |
| `models.py` | LSTM itself is unidirectional, contrary to report; return-target output shapes generally correct; Transformer position encoding already fixed upstream | Small documented models; actual epochs/history/device saved |
| Modeling notebook cells 7–9 | Input-weight magnitude mislabeled explanatory importance; SHAP assumes fixed batch=100; parameter sensitivity performed on test; stop-loss parameter not actually changed | Remove unsupported importance and SHAP artifacts; fixed strategy, predeclared cost stress only |
| Modeling notebook final cell | Ends inside an unclosed triple-quoted report string | Executable result-inspection notebook sharing canonical modules |
| EDA notebook | Returns combined by array position rather than date; hardcoded sample TSLA not in source; unseeded plotting samples; full-history feature/target correlations could influence selection | Calendar-aware summaries; descriptive EDA separated from selection |
| `backtest.py` | Same-index close execution with unspecified information time; silently truncates arrays; close-based stop detection/fill not reliably executable | Prior-close signal, next-open entry, scheduled same-close exit; no optimistic intraday stops |
| `PortfolioBacktest` | Assumes dates equal by row count; ignores commission/slippage in rebalance; cannot account for missing quotes or dividends | Cash-constrained daily round trips; explicit costs, shares and P&L |
| `backtest.py` main | Generates random prices/predictions and writes into real results locations | Removed; real-data entry point only |
| `results/backtest_results_lstm.csv` | 504 calendar-day rows, 2022-01-03–2023-05-21, including weekends; starts at $96,043.75, ends $139,600; drawdown 25.755%, not 12.3% | Superseded by actual 2017–2018 session-indexed cash ledger |
| `results/trade_log.csv` | Unsorted 2022–2023 dates, weekend trades; no ticker/cost/cash reconciliation | Chronological keyed round-trip ledger with both execution legs |
| `results/model_comparison.csv` | RMSE and R² imply different target variances (0.0002045–0.0002725); cannot be a common target sample within rounding | Every model evaluated on precisely the same rows; variance identity tested |
| Random Forest / ARIMA historical rows | No corresponding training implementation, fitted checkpoint or predictions found | Actual sklearn forest and statsmodels ARIMA(1,0,0) fits |
| `final_report.md` | Unsupported ATR/CCI/Williams features, RSI/Bollinger trading filters, beta/sector constraints, market-regime analysis, CUDA environment and epoch counts | Replaced with provenance-backed report |
| `tests/test_core.py` | Basic smoke tests only; no test isolation, accounting reconciliation or artifact provenance checks | Leakage mutation tests, split/scaler tests, hand-calculated accounting and metrics, checkpoint replay |
| `requirements.txt`, `src/utils.py` | Broad unpinned/unused dependencies; utils empty | Exact environment lock, small direct dependency list; central metrics |

The placeholder files are conclusively fake checkpoints. The date-incompatible curve and trades cannot be experiments on the stated dataset. Other historical numbers are **unverified**, not automatically proven synthetic merely because provenance is absent. No attempt is made to recover the generating random seed or infer author intent.

## Before-versus-after interpretation

Historical scores are withdrawn. The replacement changes the target from close-to-close to executable next-session open-to-close returns, the architecture size, feature representation, split and strategy. Therefore numerical changes are not a controlled attribution of any one bug's effect. The before/after table documents claims versus verified outcomes, not a like-for-like performance gain.

## Run amendments

The initial validation run was interrupted before test inference to preserve gaps in ARIMA's market calendar. Forest feature subsampling (`sqrt`) was fixed at the same point to control training cost, before any completed validation comparison. Partial outputs remain in `interrupted_validation/`. A command-path error caused a brief restart of panel construction, also stopped before any test inference. No test result was used to make these changes.
