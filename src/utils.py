import numpy as np
import pandas as pd
from scipy.stats import rankdata

def corr(a,b):
    return float(np.corrcoef(a,b)[0,1]) if len(a)>2 and np.std(a)>1e-14 and np.std(b)>1e-14 else np.nan

def predictive_metrics(frame,prediction):
    y=frame.y.to_numpy();p=np.asarray(prediction);error=p-y
    daily=[]
    for date,ix in frame.groupby('date').indices.items():
        daily.append({'date':date,'ic':corr(y[ix],p[ix]),'rank_ic':corr(rankdata(y[ix]),rankdata(p[ix]))})
    daily=pd.DataFrame(daily);s=daily.ic.dropna()
    result={'n':len(y),'rmse':float(np.sqrt(np.mean(error**2))),'mae':float(np.mean(abs(error))),
            'r2':float(1-np.sum(error**2)/np.sum((y-y.mean())**2)),
            'directional_accuracy':float(np.mean(np.sign(y)==np.sign(p))),
            'ic':float(s.mean()),'rank_ic':float(daily.rank_ic.mean()),
            'icir':float(s.mean()/s.std(ddof=1)) if len(s)>1 and s.std()>0 else np.nan,
            'ic_valid_days':len(s)}
    return result,daily

def portfolio_metrics(daily,initial=100000):
    r=daily.net_return.to_numpy(); equity=np.r_[initial,daily.equity.to_numpy()]
    dd=equity/np.maximum.accumulate(equity)-1
    return {'annualized_return':float((equity[-1]/initial)**(252/len(r))-1),
            'total_return':float(equity[-1]/initial-1),'sharpe':float(np.mean(r)/np.std(r,ddof=1)*np.sqrt(252)) if np.std(r)>1e-14 else np.nan,
            'max_drawdown':float(-dd.min()),'mean_daily_turnover':float(daily.turnover.mean()),
            'annualized_turnover':float(daily.turnover.mean()*252),'transaction_costs':float(daily.cost.sum()),
            'final_equity':float(equity[-1]),'days':len(r)}
