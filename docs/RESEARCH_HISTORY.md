# Research history and evaluation status

The primary publication is the [unified 2026 comparison](../comparison_2026/report.md). Earlier studies are retained as research records; their publication-status wording describes the state at the time of each run.

| Stage | Evaluation status | Evidence | Finding |
|---|---|---|---|
| Upstream audit | Forensic review of commit `9b143e4` | [Audit](../audit/AUDIT.md), original-file hashes and compressed archive | Saved neural weights were text placeholders; historical curves and trades used dates outside the stated dataset. Those claims are withdrawn. |
| Kaggle redesign | 2016 validation; purged expanding 2017–2018 test, later exploratory | [Baseline report](../results/final_report.md), fold predictions/checkpoints/ledgers | All four fitted models had negative R² and lost money after fixed costs. |
| Expanded feature/model research | 2021–2023 selection; 2024–2025 initially sealed, then inspected | [Expanded report](../research/final_report.md), all validation trials, frozen selection | Ranking and lower turnover improved absolute results. A documented post-reveal rank-reference correction reduced performance; ML did not outperform simple momentum. |
| Incremental-alpha research | Data through 2025 exploratory; 2026 initially sealed until selection | [Incremental report](../incremental/final_report.md), gate results and ablations | No candidate passed the full development gate. The frozen exploratory residual challenger failed to add value versus same-risk momentum. |
| Unified comparison | **Retrospective 2026; already inspected data** | [Current report](../comparison_2026/report.md) | Residual ML leads realized return and Sharpe under a common allocator. All pairwise excess-return intervals include zero. |

## Preservation policy

Earlier reports, selection locks, score tables, model checkpoints, negative findings and the original unverified archive are retained. The original artifacts are audit evidence and are not active results. The first 2024–2025 reveal remains in `research/superseded_final_rank_reference/`; its comparison to the corrected calculation is documented in the expanded report. Protocol amendments record interrupted development runs and the zero-range feature repair.

Large rebuildable panels, raw provider CSVs, verbose execution logs and redundant local snapshot ZIPs are excluded from Git. They remain in the local research workspace. Download requests and source fingerprints are retained. A downloaded file that no longer matches its recorded hash is a revised input, not a bitwise reproduction of the original experiment.

The top-level `artifact_manifest.json` is the **historical workspace inventory**, including these local-only files; it is not a promise that every raw or intermediate file is distributed. The publication inventory is `publication_manifest.json`. The primary verifier can audit the published comparison without the excluded caches; its optional `--history` check audits the full original local workspace.

## Interpretation across stages

Different stages changed targets, periods, allocation rules and risk settings. Their scores are not controlled estimates of the effect of a single modeling improvement. The unified comparison standardizes portfolio construction specifically to distinguish frozen model signals from earlier allocation differences. Its results must not be relabeled as fresh holdout evidence.
