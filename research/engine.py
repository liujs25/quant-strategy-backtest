"""Local research: causal multi-scale features and dividend-aware open-to-open ledger."""
from pathlib import Path
import numpy as np
import pandas as pd


def load_panel(folder):
    parts=[]
    for p in sorted(Path(folder).glob('*.csv')):
        g=pd.read_csv(p,parse_dates=['Date']).rename(columns={'Date':'date','Open':'open','High':'high','Low':'low','Close':'close','Volume':'volume','Dividends':'dividend'})
        g['symbol']=p.stem;parts.append(g[['date','symbol','open','high','low','close','volume','dividend']])
    return pd.concat(parts,ignore_index=True).sort_values(['symbol','date'])


def make_features(raw):
    calendar=pd.DatetimeIndex(sorted(raw.date.unique()));parts=[];names=[]
    for symbol,g in raw.groupby('symbol'):
        g=g.set_index('date').reindex(calendar).rename_axis('date').reset_index();g['symbol']=symbol
        c=g.close;o=g.open;ret=c.pct_change(fill_method=None)
        feats={}
        for w in [1,5,10,20,60,120,252]:feats[f'mom_{w}']=c/c.shift(w)-1
        feats['mom_12_1']=c.shift(20)/c.shift(252)-1
        for w in [5,20,60]:
            feats[f'vol_{w}']=ret.rolling(w).std()
            feats[f'ma_gap_{w}']=c/c.rolling(w).mean()-1
            feats[f'volume_ratio_{w}']=g.volume/g.volume.rolling(w).mean()-1
            feats[f'range_{w}']=((g.high-g.low)/c).rolling(w).mean()
        feats['intraday']=c/o-1;feats['overnight']=o/c.shift()-1
        feats['close_location']=((2*c-g.high-g.low)/(g.high-g.low).replace(0,np.nan)).where(g.high!=g.low,0.)
        feats['illiquidity']=(ret.abs()/(c*g.volume).replace(0,np.nan)).rolling(20).mean()
        feats['reversal_5']=-feats['mom_5'];feats['reversal_1']=-ret
        for name,series in feats.items():g[name]=series
        names=list(feats)
        g['signal_date']=g.date;g['date']=g.date.shift(-1)
        g['execution_open']=o.shift(-1);g['next_open']=o.shift(-2);g['next_dividend']=g.dividend.shift(-2).fillna(0)
        daily=(o.shift(-1)+g.dividend.shift(-1).fillna(0))/o
        for h in [1,5,20]:
            product=pd.Series(1.,index=g.index)
            for k in range(1,h+1):product*=daily.shift(-k)
            g[f'y_{h}']=product-1;g[f'label_end_{h}']=g.signal_date.shift(-(h+1))
        parts.append(g)
    panel=pd.concat(parts,ignore_index=True).sort_values(['date','symbol']).reset_index(drop=True)
    panel[names]=panel[names].replace([np.inf,-np.inf],np.nan)
    # Cross-sectional ranks use only features observable at signal time.
    rankcols=[]
    for name in names:
        r='rank_'+name;panel[r]=panel.groupby('date')[name].rank(pct=True)-.5;rankcols.append(r)
    panel['market_ret']=panel.groupby('date').mom_1.transform('mean')
    panel['market_vol']=panel.groupby('date').vol_20.transform('mean')
    panel['relative_mom20']=panel.mom_20-panel.groupby('date').mom_20.transform('mean')
    return panel,names+rankcols+['market_ret','market_vol','relative_mom20']


def portfolio(frame,scores,rebalance=5,buffer=False,inverse_vol=False,cost=.0015,top=20):
    """Signals pre-open; trade reference open, cash dividend next ex-date open.
    Fixed shares between rebalances. Sell-first cash-constrained sizing; final liquidation cost.
    Common complete panel required by the caller; no missing-security silent drops.
    """
    f=frame.copy();f['score']=np.asarray(scores);cash=100000.;positions={};rows=[];trades=[];prior=100000
    groups=list(f.groupby('date',sort=True))
    for i,(date,g) in enumerate(groups):
        g=g.set_index('symbol');prices=g.execution_open
        if any(s not in g.index for s in positions):raise ValueError('Held security missing; explicit policy required')
        nav=cash+sum(q*prices[s] for s,q in positions.items());totalcost=turn=0.
        if i%rebalance==0:
            ranks=g.score.rank(ascending=False,method='first');chosen=list(ranks.nsmallest(top).index)
            if buffer:
                keep=[s for s in positions if ranks[s]<=2*top];chosen=(keep+[s for s in chosen if s not in keep])[:top]
            weights=np.ones(len(chosen))
            if inverse_vol:weights=1/np.maximum(g.loc[chosen,'vol_20'].to_numpy(),.005)
            weights=weights/weights.sum()*.95
            target={s:nav*w/prices[s] for s,w in zip(chosen,weights)}
            for s in sorted(positions):
                delta=min(0.,target.get(s,0)-positions[s])
                if delta<0:
                    value=-delta*prices[s];fee=value*cost;cash+=value-fee;positions[s]+=delta;turn+=value;totalcost+=fee
                    trades.append([date,s,'SELL',-delta,prices[s],fee])
            positions={s:q for s,q in positions.items() if q>1e-10}
            for s in chosen:
                delta=max(0.,target[s]-positions.get(s,0));delta=min(delta,max(0,cash)/(prices[s]*(1+cost)))
                if delta>0:
                    value=delta*prices[s];fee=value*cost;cash-=value+fee;positions[s]=positions.get(s,0)+delta;turn+=value;totalcost+=fee
                    trades.append([date,s,'BUY',delta,prices[s],fee])
        dividends=sum(q*g.loc[s,'next_dividend'] for s,q in positions.items());cash+=dividends
        end=cash+sum(q*g.loc[s,'next_open'] for s,q in positions.items())
        if i==len(groups)-1:
            for s,q in positions.items():
                value=q*g.loc[s,'next_open'];fee=value*cost;cash+=value-fee;turn+=value;totalcost+=fee
                trades.append([g.loc[s,'label_end_1'],s,'FINAL_SELL',q,g.loc[s,'next_open'],fee])
            positions={};end=cash
        assert cash>=-1e-6
        rows.append({'date':date,'equity':end,'net_return':end/prior-1,'cash':cash,'position_value':end-cash,'dividends':dividends,'cost':totalcost,'turnover':turn/nav});prior=end
    return pd.DataFrame(rows),pd.DataFrame(trades,columns=['date','symbol','side','shares','price','cost'])
