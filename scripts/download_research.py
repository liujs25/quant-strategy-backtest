"""Fetch development data only; final period requires a frozen selection."""
from pathlib import Path
import argparse,json,hashlib
import yfinance as yf
root=Path(__file__).resolve().parents[1];r=root/'research';p=json.loads((r/'protocol.json').read_text())
a=argparse.ArgumentParser();a.add_argument('--final',action='store_true');args=a.parse_args()
if args.final and not (r/'selection_lock.json').exists():raise SystemExit('Final data sealed until selection lock exists')
folder=r/('final_raw' if args.final else 'development_raw');folder.mkdir(exist_ok=True)
start,end=('2023-01-01','2026-01-01') if args.final else ('2010-01-01','2024-01-01')
status=[]
for symbol in p['symbols']:
    path=folder/f'{symbol}.csv';yahoo=symbol.replace('.','-')
    if path.exists():
        status.append({'symbol':symbol,'status':'cached','sha256':hashlib.sha256(path.read_bytes()).hexdigest()});continue
    try:
        d=yf.Ticker(yahoo).history(start=start,end=end,auto_adjust=False,actions=True,repair=False,raise_errors=True)
        if d.empty:raise ValueError('Empty response')
        d.index=d.index.tz_localize(None);d.to_csv(path)
        status.append({'symbol':symbol,'status':'downloaded','rows':len(d),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()});print(symbol,len(d),flush=True)
    except Exception as e:
        status.append({'symbol':symbol,'status':'unavailable','error':str(e)});print(symbol,'UNAVAILABLE',str(e)[:150],flush=True)
    (folder/'download_manifest.json').write_text(json.dumps({'source':'Yahoo Finance via yfinance','auto_adjust':False,'repair':False,'requested_start':start,'exclusive_end':end,'files':status},indent=2))
(folder/'download_manifest.json').write_text(json.dumps({'source':'Yahoo Finance via yfinance','requested_start':start,'exclusive_end':end,'files':status},indent=2))
