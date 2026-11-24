# Local incremental-alpha run record

- Publication prohibited: no commit/push. Git HEAD remains original `9b143e407ed878e0694564f953149d0456177433`.
- Previous 2024–2025 reveal was treated as exploratory from the start of this study.
- Protocol fixed before 2026 retrieval: four fixed tree target/feature configurations, ten signals, three risk variants, two development folds. No architecture expansion or hyperparameter sweep.
- Initial development run stopped when factor-regression diagnostics mistakenly dropped rows for undefined optional ledger fields. Fixed `dropna` to require only regression columns and reran the same study before selection/holdout. No strategy changes resulted.
- No candidate passed the complete development gate. Residual-no-volume diversified tilt was locked as an exploratory challenger; the gate was not weakened.
- 2026 data downloaded after the selection lock. All 88 reference tickers returned 224 source sessions. Final observations include 87 tradable names and 204 decision dates, with 20-session labels completed by November20.
- Final evaluation used unchanged score construction, risk settings and benchmarks. All four fixed model checkpoints were fitted for reproducible score replay, but only the locked ML strategy was evaluated economically against the fixed traditional comparators. Other final scores were not used for selection.
- Holdout result: negative incremental return versus same-risk momentum; uncertainty includes zero under both20- and60-session blocks. No post-holdout model change or second candidate evaluation.
- All ledgers and checkpoints replayed;21 regression tests passed; notebooks executed; figures visually reviewed.
- Further selection must not reuse2026 as a new holdout. No promise of statistically credible alpha is made.
