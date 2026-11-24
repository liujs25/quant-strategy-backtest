from pathlib import Path
import json
import pandas as pd
root=Path(__file__).resolve().parents[1];r=root/'results'
raw=pd.read_csv(root/'data/raw/all_stocks_5yr.csv',parse_dates=['date'])
invalid=~((raw[['open','high','low','close']]>0).all(axis=1)&(raw.volume>=0))
raw.loc[invalid].to_csv(r/'invalid_raw_rows.csv',index=False)
raw.groupby('Name').agg(first_date=('date','min'),last_date=('date','max'),rows=('date','size')).to_csv(r/'data_coverage.csv')
print('Invalid raw rows:',int(invalid.sum()))
