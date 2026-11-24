import numpy as np
import pandas as pd
import pytest
from incremental.core import neutralize,optimize,covariance,backtest

def test_residual_orthogonality():
 rng=np.random.default_rng(3);f=rng.normal(size=(100,4));y=2+f@np.arange(4)+rng.normal(size=100);r=neutralize(y,f)
 assert abs(r.mean())<1e-10;np.testing.assert_allclose(f.T@r,0,atol=1e-9)

def test_neutral_optimizer_caps_and_exposures():
 rng=np.random.default_rng(4);n=87;f=rng.normal(size=(n,4));prev=np.full(n,.95/n)
 ref=optimize(f[:,0],prev,.03,1);w=optimize(rng.normal(size=n),prev,.03,1,f,ref)
 assert w.min()>=-1e-9;assert w.max()<=.03+1e-8;assert w.sum()==pytest.approx(.95)
 np.testing.assert_allclose(f.T@w,f.T@ref,atol=1e-6)

def test_covariance_cannot_see_future():
 rng=np.random.default_rng(5);r=pd.DataFrame(rng.normal(0,.01,(100,3)),index=pd.bdate_range('2020-01-01',periods=100),columns=list('ABC'));day=r.index[70]
 a=covariance(r,day,list('ABC'));r.loc[r.index>day]=100
 np.testing.assert_array_equal(a,covariance(r,day,list('ABC')))

def test_neutrality_target_future_labels_not_features():
 rng=np.random.default_rng(6);f=rng.normal(size=(50,4));copy=f.copy();neutralize(rng.normal(size=50),f);np.testing.assert_array_equal(f,copy)

def test_risk_ledger_and_future_score_independence():
 from incremental.core import FACTORS
 rng=np.random.default_rng(8);symbols=[f'S{i:02}' for i in range(40)];dates=pd.bdate_range('2020-01-01',periods=75)
 returns=pd.DataFrame(rng.normal(0,.012,(75,40)),index=dates,columns=symbols);rows=[]
 for i in range(65,70):
  for j,s in enumerate(symbols):
   row={'date':dates[i],'signal_date':dates[i-1],'symbol':s,'execution_open':100.,'next_open':101.,'next_dividend':.1,'label_end_1':dates[i+1]}
   row.update({f:float(rng.normal()) for f in FACTORS});rows.append(row)
 frame=pd.DataFrame(rows);risk={'max_weight':.03,'turnover_penalty':1};score=rng.normal(size=len(frame))
 d,t,w=backtest(frame,score,returns,risk,neutral=True)
 assert d.cash.min()>=0;assert d.position_value.iloc[-1]==0;assert w.weight.max()<=.03+1e-8
 assert d.neutrality_error.max()<1e-6
 np.testing.assert_allclose(d.cash+d.position_value,d.equity)
 np.testing.assert_allclose(t.cost.sum(),d.cost.sum())
 changed=score.copy();changed[frame.date>dates[65]]=100
 d2,t2,w2=backtest(frame,changed,returns,risk,neutral=True)
 np.testing.assert_allclose(w.weight,w2.weight)
