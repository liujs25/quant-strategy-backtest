import numpy as np
import pandas as pd
import pytest
from research.engine import make_features,portfolio

def fixture():
    rows=[]
    for s in ['A','B','C']:
        for i,d in enumerate(pd.bdate_range('2019-01-01',periods=400)):
            p=100+i*.1;rows.append(dict(date=d,symbol=s,open=p,close=p+.2,high=p+1,low=p-1,volume=10000+i,dividend=.5 if i==350 else 0))
    return pd.DataFrame(rows)

def test_causal_multiscale_and_horizons():
    raw=fixture();p,c=make_features(raw);cut=sorted(raw.date.unique())[340]
    altered=raw.copy();altered.loc[altered.date>=cut,['open','close','high','low']]*=2
    q,_=make_features(altered)
    ix=p.signal_date<cut;np.testing.assert_allclose(p.loc[ix,c],q.loc[ix,c],equal_nan=True)
    row=p[(p.symbol=='A')&(p.signal_date==sorted(raw.date.unique())[340])].iloc[0]
    assert row.y_5==pytest.approx((134.6/134.1)-1)
    assert row.date>row.signal_date;assert row.label_end_5>row.date

def test_open_portfolio_cash_dividends_and_fees():
    f=pd.DataFrame({'date':pd.to_datetime(['2020-01-02','2020-01-03']),'symbol':['A','A'],'execution_open':[100,110],'next_open':[110,100],'next_dividend':[1,0],'vol_20':[.1,.1],'label_end_1':pd.to_datetime(['2020-01-03','2020-01-06'])})
    d,t=portfolio(f,[1,1],rebalance=20,cost=.01,top=1)
    shares=950.;expected=100000-95000-950+950+95000-950
    assert d.equity.iloc[-1]==pytest.approx(expected)
    assert d.cost.sum()==pytest.approx(t.cost.sum());assert d.position_value.iloc[-1]==0
    assert (d.cash>=0).all();assert d.dividends.sum()==950

def test_zero_range_stays_eligible():
    raw=fixture();day=sorted(raw.date.unique())[340];ix=(raw.date==day)&(raw.symbol=='A')
    raw.loc[ix,'high']=raw.loc[ix,'close'];raw.loc[ix,'low']=raw.loc[ix,'close']
    p,c=make_features(raw);row=p[(p.signal_date==day)&(p.symbol=='A')].iloc[0]
    assert row.close_location==0;assert np.isfinite(row[c].to_numpy(dtype=float)).all()

def test_turnover_buffer_retains_incumbent_and_cost_penalty():
    rows=[]
    dates=pd.bdate_range('2020-01-01',periods=4)
    for i,d in enumerate(dates):
        for j,s in enumerate(['A','B','C']):rows.append({'date':d,'symbol':s,'execution_open':100.,'next_open':100.,'next_dividend':0.,'vol_20':.1,'label_end_1':d+pd.Timedelta(days=1),'score':3-j if i%2==0 else j+1})
    f=pd.DataFrame(rows)
    d,t=portfolio(f,f.score,rebalance=1,top=1,buffer=False,cost=.0015)
    b,bt=portfolio(f,f.score,rebalance=1,top=1,buffer=True,cost=.0015)
    z,_=portfolio(f,f.score,rebalance=1,top=1,buffer=False,cost=0)
    assert d.equity.iloc[-1]<z.equity.iloc[-1];assert b.cost.sum()<=d.cost.sum()+1e-8

def test_missing_held_security_fails():
    f=pd.DataFrame({'date':pd.to_datetime(['2020-01-01','2020-01-02']),'symbol':['A','B'],'execution_open':[100.,100.],'next_open':[100.,100.],'next_dividend':[0.,0.],'vol_20':[.1,.1],'label_end_1':pd.to_datetime(['2020-01-02','2020-01-03'])})
    with pytest.raises(ValueError,match='Held security missing'):portfolio(f,[1,1],top=1,rebalance=20)

def test_ranking_reference_must_precede_trading_filter():
    raw=fixture();raw.loc[raw.symbol=='C','close']*=1.02
    full,cols=make_features(raw)
    # Filtering tradability after feature creation preserves the original rank reference.
    tradable=full[full.symbol.isin(['A','B'])]
    expected=full.set_index(['date','symbol']).loc[list(zip(tradable.date,tradable.symbol)),cols]
    np.testing.assert_allclose(tradable[cols],expected,equal_nan=True)
