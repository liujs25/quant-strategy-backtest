from pathlib import Path
import json,hashlib
import yfinance as yf
R=Path(__file__).resolve().parent
assert (R/'selection_lock.json').exists(),'Holdout sealed'
ref=[p.stem for p in (R.parent/'research/development_raw').glob('*.csv')];folder=R/'holdout_raw';folder.mkdir(exist_ok=True);status=[]
for symbol in sorted(ref):
 p=folder/f'{symbol}.csv'
 if p.exists():raise SystemExit('Do not overwrite frozen holdout; use separate replication copy')
 try:
  d=yf.Ticker(symbol.replace('.','-')).history(start='2026-01-01',end='2026-11-21',auto_adjust=False,repair=False,actions=True)
  if d.empty:raise ValueError('No data')
  d.index=d.index.tz_localize(None);d.to_csv(p);status.append({'symbol':symbol,'rows':len(d),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()});print(symbol,len(d),flush=True)
 except Exception as e:status.append({'symbol':symbol,'error':str(e)});print(symbol,'FAILED',flush=True)
 (R/'holdout_download_manifest.json').write_text(json.dumps({'source':'Yahoo Finance via yfinance','requested_start':'2026-01-01','exclusive_end':'2026-11-21','files':status},indent=2))
