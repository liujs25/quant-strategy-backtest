# Local run record

1. Original Kaggle baseline: completed on MPS, 138,894 common test observations; checkpoint replay, portfolio reconciliation, and regression tests passed. Original unsupported results quarantined.
2. User requested local optimization and revoked publication. No commit or push had occurred. Baseline snapshot preserved with SHA-256. Old test declared exploratory.
3. New design set before future-period retrieval: development through 2023; final 2024–2025 withheld; universe fixed from 2016 liquidity. Downloaded development only. 12 names unavailable, PYPL excluded by pre-2015-history rule; 87 retained. Both provider-availability and validation-completeness survivorship conditioning disclosed.
4. First boosting process crashed before a completed trial, exit 139. Installed libomp and used one CPU thread to avoid local OpenMP conflict. Neural MPS retained.
5. Validation accounting rejected a missing held BIIB row caused by zero price range. Defined close-location as zero on zero-range days; retained superseded output and restarted the same bounded grid. No final data involved.
6. Bounded grid: 24 boosted specifications + 8 neural specifications, two validation folds, three portfolio modes. Regression/ranking targets, 1/5/20-day horizons, 20/60 neural lookbacks. Simple reversal/momentum/low-volatility controls retained.
7. Validation portfolio comparison re-aligned to common date/ticker intersection before selection. This corrects unequal horizon-tail evaluation dates without changing predictions; original-boundary scores preserved. Ranking-score regression error fields are undefined.
8. Selection lock precedes final download. Additional seed/feature ablations of the selected model use validation only and do not change the lock. Final evaluation runs once with frozen design. Any final data-quality problem must fail explicitly rather than drop securities.
9. No remote compute, messaging, commit or push. Final publication requires explicit human approval.
