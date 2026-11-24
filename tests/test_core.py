import numpy as np
import pandas as pd
import pytest
from src.data_preprocessing import build_panel,fit_scaler,split_indices
from src.backtest import backtest,buy_hold
from src.utils import predictive_metrics,portfolio_metrics

def raw_panel(n=160):
    out=[]
    for symbol in ['A','B']:
        for i,d in enumerate(pd.bdate_range('2015-01-01',periods=n)):
            o=100+i*.1;c=o*(1+(.01 if i%2 else -.005))
            out.append(dict(Name=symbol,date=d,open=o,close=c,high=max(o,c)+1,low=min(o,c)-1,volume=10000+i))
    return pd.DataFrame(out)

def test_future_mutation():
    raw=raw_panel();x,m,_=build_panel(raw);changed=raw.copy();cut=raw.date.unique()[100]
    changed.loc[changed.date>=cut,['open','high','low','close']]*=5
    x2,m2,_=build_panel(changed);before=m.date<=cut
    np.testing.assert_array_equal(x[before],x2[before]);assert (m.signal_date<m.date).all()
    np.testing.assert_allclose(m.y,m.close/m.open-1)

def test_split_and_scaler():
    x,m,_=build_panel(raw_panel());ti,ei,_=split_indices(m,str(m.date.unique()[75]),str(m.date.max()))
    assert m.iloc[ti].date.max()<m.iloc[ei].feature_start.min()
    assert set(m.iloc[ti].date).isdisjoint(m.iloc[ei].date)
    mean,std=fit_scaler(x[ti]);x[ei]=999;mean2,std2=fit_scaler(x[ti])
    np.testing.assert_array_equal(mean,mean2);np.testing.assert_array_equal(std,std2)

def test_gap():
    raw=raw_panel();missing=raw.date.unique()[80];end=raw.date.unique()[100]
    raw=raw[~((raw.Name=='A')&(raw.date==missing))];_,m,_=build_panel(raw)
    assert not ((m.symbol=='A')&(m.date>missing)&(m.date<=end)).any()

def test_duplicate():
    raw=raw_panel()
    with pytest.raises(ValueError):build_panel(pd.concat([raw,raw.iloc[:1]]))

def frame():
    return pd.DataFrame({'date':pd.to_datetime(['2020-01-02','2020-01-03']),'signal_date':pd.to_datetime(['2020-01-01','2020-01-02']), 'symbol':['A','A'],'open':[100.,100.],'close':[110.,90.],'y':[.1,-.1]})

def test_accounting():
    d,t=backtest(frame(),[.1,.1],cost=.01,slots=1,initial=1000)
    expected=1000*(1.1*.99/1.01)*(.9*.99/1.01)
    assert d.equity.iloc[-1]==pytest.approx(expected)
    assert 1000+t.net_pnl.sum()==pytest.approx(expected)
    assert d.cost.sum()==pytest.approx((t.entry_cost+t.exit_cost).sum())
    assert (d.cash==d.equity).all();assert (d.end_position_value==0).all()
    assert d.turnover.iloc[0]==pytest.approx(2.1/1.01)

def test_alignment_cash():
    f=frame();d,t=backtest(f,[0,0]);assert t.empty;assert (d.equity==100000).all()
    with pytest.raises(ValueError):backtest(f,[0])
    f.signal_date=f.date
    with pytest.raises(ValueError):backtest(f,[.1,.1])

def test_metrics():
    f=frame();m,_=predictive_metrics(f,[.1,-.1]);assert m['rmse']==0;assert m['r2']==1;assert m['directional_accuracy']==1
    m,_=predictive_metrics(f,[0,0]);assert m['r2']==0;assert m['directional_accuracy']==0;assert np.isnan(m['ic'])
    d=pd.DataFrame({'equity':[900.,990.],'net_return':[-.1,.1],'cost':[0.,0.],'turnover':[0.,0.]})
    p=portfolio_metrics(d,1000);assert p['max_drawdown']==pytest.approx(.1);assert p['total_return']==pytest.approx(-.01)

def test_ic():
    f=pd.DataFrame({'date':['a']*3+['b']*3,'y':[1,2,3,3,2,1]})
    m,_=predictive_metrics(f,[2,4,6,6,4,2]);assert m['ic']==pytest.approx(1);assert m['rank_ic']==pytest.approx(1)

def test_benchmark():
    raw=pd.DataFrame({'Name':['A','A'],'date':pd.to_datetime(['2020-01-02','2020-01-03']),'open':[100,120],'close':[110,90]})
    d=buy_hold(raw,pd.DatetimeIndex(raw.date),['A'],cost=.01,initial=1000)
    assert d.equity.iloc[-1]==pytest.approx(1000/1.01*.9*.99)
    assert d.equity.iloc[0]==pytest.approx(1000/1.01*1.1)

def test_models(tmp_path):
    import torch
    from src.models import LSTM,Transformer
    for cls in [LSTM,Transformer]:
        m=cls().eval();x=torch.zeros(3,20,8);a=m(x);assert a.shape==(3,)
        torch.save(m.state_dict(),tmp_path/'m.pt');m2=cls().eval();m2.load_state_dict(torch.load(tmp_path/'m.pt',weights_only=True));torch.testing.assert_close(a,m2(x))
    assert not torch.equal(Transformer().pe[0],Transformer().pe[1])
