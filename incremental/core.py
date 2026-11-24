from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from research.engine import load_panel,make_features
FACTORS=['rank_mom_12_1','rank_mom_60','rank_vol_20','rank_reversal_5']

def zscore(x):
    a=np.asarray(x,float);s=a.std();return (a-a.mean())/s if s>1e-12 else a*0

def neutralize(y,factors):
    a=np.column_stack([np.ones(len(y)),np.asarray(factors,float)])
    return np.asarray(y)-a@np.linalg.lstsq(a,y,rcond=None)[0]

def data(root,final=False):
    import json
    old=root/'research';dev=load_panel(old/'development_raw');ref=set(dev.symbol)
    later=load_panel(old/'final_raw');raw=pd.concat([dev,later[(later.date>dev.date.max())&later.symbol.isin(ref)]])
    if final:
        new=load_panel(root/'incremental/holdout_raw');raw=pd.concat([raw,new[(new.date>raw.date.max())&new.symbol.isin(ref)]])
    raw=raw.sort_values(['symbol','date']);assert not raw.duplicated(['symbol','date']).any()
    universe=json.loads((old/'selection_lock.json').read_text())['universe']
    panel,cols=make_features(raw);panel=panel[panel.symbol.isin(universe)].dropna(subset=cols+['date','execution_open','next_open']).copy()
    closes=raw[raw.symbol.isin(universe)].pivot(index='date',columns='symbol',values='close').reindex(columns=sorted(universe))
    returns=closes.pct_change(fill_method=None)
    return panel,cols,returns

def covariance(returns,signal_date,symbols):
    hist=returns.loc[:signal_date,symbols].tail(60)
    if len(hist)<60 or hist.isna().any().any():raise ValueError('Incomplete trailing covariance history')
    c=hist.cov().to_numpy()*252;c=.75*c+.25*np.diag(np.diag(c))+np.eye(len(c))*1e-8
    return c

def optimize(score,previous,cap,penalty,factors=None,reference=None):
    n=len(score);score=zscore(score);base=np.full(n,.95/n)
    target=base+.012*score
    if reference is not None:target=reference+.006*score
    def obj(w):return np.sum((w-target)**2)+penalty*np.sum((w-previous)**2)
    def jac(w):return 2*(w-target)+2*penalty*(w-previous)
    constraints=[{'type':'eq','fun':lambda w:w.sum()-.95,'jac':lambda w:np.ones(n)}]
    if factors is not None:
        f=np.asarray(factors);exposure=f.T@reference
        constraints.append({'type':'eq','fun':lambda w:f.T@w-exposure,'jac':lambda w:f.T})
    init=reference.copy() if reference is not None else base
    result=minimize(obj,init,jac=jac,bounds=[(0,cap)]*n,constraints=constraints,method='SLSQP',options={'ftol':1e-11,'maxiter':200})
    if not result.success:raise RuntimeError(result.message)
    w=result.x;assert abs(w.sum()-.95)<1e-7;assert w.max()<=cap+1e-7
    if factors is not None:assert np.max(abs(f.T@(w-reference)))<1e-6
    return w

def backtest(frame,score,returns,risk,neutral=False,cost=.0015):
    frame=frame.copy();frame['score']=np.asarray(score);symbols=sorted(frame.symbol.unique());n=len(symbols)
    cash=100000.;q=np.zeros(n);prev=100000.;rows=[];trades=[];weights=[];groups=list(frame.groupby('date',sort=True))
    prior_unscaled=np.full(n,.95/n);prior_ref=prior_unscaled.copy()
    for i,(date,g) in enumerate(groups):
        g=g.set_index('symbol').reindex(symbols)
        if g[['execution_open','next_open']].isna().any().any():raise ValueError('Incomplete fixed universe')
        op=g.execution_open.to_numpy();nxt=g.next_open.to_numpy();nav=cash+q@op;fees=turn=0.;exante=np.nan;scale=np.nan;neutral_error=np.nan
        if i%20==0:
            f=g[FACTORS].to_numpy();c=covariance(returns,g.signal_date.iloc[0],symbols)
            reference=optimize(g.rank_mom_12_1,prior_ref,risk['max_weight'],risk['turnover_penalty']);prior_ref=reference
            w=optimize(g.score,prior_unscaled,risk['max_weight'],risk['turnover_penalty'],f if neutral else None,reference if neutral else None);prior_unscaled=w
            neutral_error=float(np.max(abs(f.T@(w-reference)))) if neutral else np.nan
            exante=float(np.sqrt(w@c@w));scale=min(1.,.10/max(exante,1e-8));w=w*scale
            target=nav*w/op;delta=target-q
            for j in np.flatnonzero(delta<0):
                value=-delta[j]*op[j];fee=value*cost;cash+=value-fee;q[j]+=delta[j];fees+=fee;turn+=value;trades.append([date,symbols[j],'SELL',-delta[j],op[j],fee])
            for j in np.flatnonzero(delta>0):
                shares=min(delta[j],max(cash,0)/(op[j]*(1+cost)));value=shares*op[j];fee=value*cost;cash-=value+fee;q[j]+=shares;fees+=fee;turn+=value;trades.append([date,symbols[j],'BUY',shares,op[j],fee])
            for j,s in enumerate(symbols):weights.append({'date':date,'symbol':s,'weight':q[j]*op[j]/nav,'target_weight':w[j],'reference_unscaled':reference[j],**{f:float(g.loc[s,f]) for f in FACTORS}})
        dividends=float(q@g.next_dividend.fillna(0).to_numpy());cash+=dividends;end=cash+q@nxt
        if i==len(groups)-1:
            for j in np.flatnonzero(q>0):
                value=q[j]*nxt[j];fee=value*cost;cash+=value-fee;fees+=fee;turn+=value;trades.append([g.label_end_1.iloc[0],symbols[j],'FINAL_SELL',q[j],nxt[j],fee])
            q*=0;end=cash
        assert cash>=-1e-6
        rows.append({'date':date,'equity':end,'net_return':end/prev-1,'cash':cash,'position_value':end-cash,'dividends':dividends,'cost':fees,'turnover':turn/nav,'exante_before_scaling':exante,'vol_scale':scale,'neutrality_error':neutral_error});prev=end
    return pd.DataFrame(rows),pd.DataFrame(trades,columns=['date','symbol','side','shares','price','cost']),pd.DataFrame(weights)
