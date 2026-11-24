"""Causal panel features. No price imputation, clipping, or outcome filtering."""
import numpy as np
import pandas as pd

FEATURES = ['ret1','ret5','ret20','intraday','range','gap','vol20','volume_ratio']
LOOKBACK = 20
RAW_SPAN = 40  # 20 rows of features; longest feature needs 20 previous prices


def features(g):
    g = g.sort_values('date').copy()
    c, o = g.close, g.open
    g['ret1'] = c.pct_change(fill_method=None)
    g['ret5'] = c.pct_change(5, fill_method=None)
    g['ret20'] = c.pct_change(20, fill_method=None)
    g['intraday'] = c/o-1
    g['range'] = (g.high-g.low)/c
    g['gap'] = o/c.shift()-1
    g['vol20'] = g.ret1.rolling(20).std()
    g['volume_ratio'] = g.volume/g.volume.rolling(20).mean()-1
    return g


def build_panel(raw):
    raw = raw.copy()
    raw['date'] = pd.to_datetime(raw.date)
    if raw.duplicated(['Name','date']).any():
        raise ValueError('Duplicate ticker/date')
    dates = np.sort(raw.date.unique())
    xs, records = [], []
    counts = {'raw_rows':len(raw),'tickers':raw.Name.nunique(),'invalid_windows':0}
    for symbol, g in raw.groupby('Name',sort=True):
        # Reindex on the observed market calendar: a gap cannot become a one-day return.
        g = g.set_index('date').reindex(dates).rename_axis('date').reset_index()
        valid = (g[['open','high','low','close']]>0).all(axis=1) & (g.volume>=0)
        g.loc[~valid,['open','high','low','close','volume']] = np.nan
        g = features(g)
        a = g[FEATURES].to_numpy(dtype=np.float32)
        for j in range(RAW_SPAN,len(g)):
            window = a[j-LOOKBACK:j]
            if not np.isfinite(window).all() or not valid.iloc[j]:
                counts['invalid_windows'] += 1
                continue
            xs.append(window)
            records.append((symbol,dates[j-RAW_SPAN],dates[j-1],dates[j],g.open.iloc[j],g.close.iloc[j],g.intraday.iloc[j],g.intraday.iloc[j-1]))
    meta = pd.DataFrame(records,columns=['symbol','feature_start','signal_date','date','open','close','y','last_intraday'])
    order = meta.sort_values(['date','symbol']).index.to_numpy()
    return np.stack(xs)[order], meta.iloc[order].reset_index(drop=True), counts


def split_indices(meta, start, end):
    evaluation = np.flatnonzero((meta.date>=pd.Timestamp(start)) & (meta.date<=pd.Timestamp(end)))
    cutoff = meta.iloc[evaluation].feature_start.min()
    train = np.flatnonzero(meta.date < cutoff)
    assert meta.iloc[train].date.max() < meta.iloc[evaluation].feature_start.min()
    return train, evaluation, cutoff


def fit_scaler(x):
    # Only training observations. Duplicated historical rows in windows are intentional.
    mean = x.mean(axis=(0,1),dtype=np.float64)
    std = x.std(axis=(0,1),dtype=np.float64)
    return mean.astype('float32'), np.maximum(std,1e-6).astype('float32')
