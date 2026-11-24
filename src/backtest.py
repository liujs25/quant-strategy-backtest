"""Cash-constrained long-only, next-open entry / same-close liquidation.
Orders are selected before the open. Fractional shares; costs on both legs.
"""
import numpy as np
import pandas as pd

def backtest(frame,prediction,threshold=0.003,cost=0.0015,initial=100000,slots=20):
    if len(frame)!=len(prediction):raise ValueError('Prediction alignment mismatch')
    if frame.duplicated(['date','symbol']).any():raise ValueError('Duplicate keys')
    if not (frame.signal_date<frame.date).all():raise ValueError('Signal must precede execution')
    if not np.isfinite(prediction).all():raise ValueError('Non-finite forecast')
    frame=frame.copy();frame['prediction']=prediction
    cash=initial;days=[];trades=[]
    for date,g in frame.groupby('date',sort=True):
        start=cash
        chosen=g[g.prediction>threshold].sort_values(['prediction','symbol'],ascending=[False,True]).head(slots)
        # Fixed 1/slots maximum allocation including entry costs. Unused slots stay cash.
        budget=start/slots
        costs=notional=0.0
        for row in chosen.itertuples():
            shares=budget/(row.open*(1+cost));buy=shares*row.open;fee=buy*cost
            cash-=buy+fee;sell=shares*row.close;sellfee=sell*cost;cash+=sell-sellfee
            costs+=fee+sellfee;notional+=buy+sell
            trades.append({'date':date,'signal_date':row.signal_date,'symbol':row.symbol,'prediction':row.prediction,
                           'shares':shares,'buy_price':row.open,'sell_price':row.close,'buy_notional':buy,'sell_notional':sell,
                           'entry_cost':fee,'exit_cost':sellfee,'net_pnl':sell-buy-fee-sellfee})
        assert cash>=0
        days.append({'date':date,'equity':cash,'cash':cash,'end_position_value':0,'net_return':cash/start-1,
                     'cost':costs,'turnover':notional/start,'positions':len(chosen)})
    return pd.DataFrame(days),pd.DataFrame(trades,columns=['date','signal_date','symbol','prediction','shares','buy_price','sell_price','buy_notional','sell_notional','entry_cost','exit_cost','net_pnl'])

def buy_hold(raw,dates,symbols,cost=0.0015,initial=100000):
    """Fixed basket selected before test; no selection using future availability.
    Missing marks use the last known close. Missing terminal quotes fail loudly.
    """
    p=raw[raw.Name.isin(symbols)].copy();p['date']=pd.to_datetime(p.date)
    opens=p.pivot(index='date',columns='Name',values='open').reindex(index=dates,columns=symbols)
    closes=p.pivot(index='date',columns='Name',values='close').reindex(index=dates,columns=symbols)
    if opens.iloc[0].isna().any() or closes.iloc[-1].isna().any():raise ValueError('Untradeable benchmark boundary; specify a missing-security policy')
    shares=initial/len(symbols)/(1+cost)/opens.iloc[0];entry=float((shares*opens.iloc[0]).sum()*cost)
    values=closes.ffill().mul(shares).sum(axis=1);exit_cost=float(values.iloc[-1]*cost)
    values.iloc[-1]-=exit_cost
    rows=[];prev=initial
    for i,(date,value) in enumerate(values.items()):
        fee=(entry if i==0 else 0)+(exit_cost if i==len(values)-1 else 0)
        turnover=(initial/(1+cost)/prev if i==0 else 0)+((value+exit_cost)/prev if i==len(values)-1 else 0)
        rows.append({'date':date,'equity':value,'net_return':value/prev-1,'cost':fee,'turnover':turnover});prev=value
    return pd.DataFrame(rows)
