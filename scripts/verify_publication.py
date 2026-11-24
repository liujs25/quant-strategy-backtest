"""Verify the publication inventory and exact primary tables against outputs."""
from pathlib import Path
import json,hashlib
import pandas as pd
R=Path(__file__).resolve().parents[1]
manifest=json.loads((R/'publication_manifest.json').read_text())
for name,digest in manifest['sha256'].items():
    assert (R/name).is_file(),name
    assert hashlib.sha256((R/name).read_bytes()).hexdigest()==digest,name
m=pd.read_csv(R/'comparison_2026/metrics.csv')
for doc in ['README.md','comparison_2026/report.md']:
    text=(R/doc).read_text()
    for row in m.itertuples():
        ic='—' if pd.isna(row.rank_ic) else f'{row.rank_ic:.4f}'
        expected=f'| {row.strategy} | {row.annualized_return:.2%} | {row.sharpe:.2f} | {row.max_drawdown:.2%} | {ic} | {row.annualized_turnover:.2f}× |'
        assert expected in text,(doc,row.strategy)
    assert 'not a fresh holdout' in text.lower()
ci=pd.read_csv(R/'comparison_2026/excess_return_intervals.csv');report=(R/'comparison_2026/report.md').read_text()
for row in ci.itertuples():
    expected=f'| {row.strategy} − {row.comparator} | {row.block_sessions} | {row.annualized_mean_excess_return:.2%} | [{row.ci_low:.2%}, {row.ci_high:.2%}] |'
    assert expected in report,expected
print(f"Verified {len(manifest['sha256'])} publication files, every primary table row and all 12 interval rows.")
