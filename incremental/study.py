"""Bounded incremental-alpha experiments; never accesses holdout before lock."""
from pathlib import Path
import sys,json,time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import joblib
import lightgbm as lgb
import statsmodels.api as sm
from incremental.core import data,FACTORS,zscore,neutralize,backtest
from src.utils import predictive_metrics,portfolio_metrics,corr
R=Path(__file__).resolve().parent;ROOT=R.parent

def factor_returns(frame):
    rows=[]
    for date,g in frame.groupby('date'):
        y=g.y_1.to_numpy();row={'date':date,'market':float(y.mean())}
        for f in FACTORS:
            w=zscore(g[f]);w=w/max(np.sum(abs(w)),1e-12);row[f]=float(w@y)
        rows.append(row)
    return pd.DataFrame(rows).set_index('date')

def hac(daily,factors):
    d=daily.set_index('date').join(factors).dropna(subset=['net_return']+list(factors.columns));fit=sm.OLS(d.net_return,sm.add_constant(d[factors.columns])).fit(cov_type='HAC',cov_kwds={'maxlags':20})
    return {'n':len(d),'r2':fit.rsquared,'annualized_intercept':fit.params['const']*252,'intercept_t':fit.tvalues['const'],'intercept_p':fit.pvalues['const'],'coefficients':fit.params.to_dict(),'hac_standard_errors':fit.bse.to_dict()}

def fit_models(train,cols,dest,params):
    models={};features={};y=train.y_20.to_numpy();res=np.empty(len(train))
    for _,ix in train.groupby('date').indices.items():res[ix]=neutralize(y[ix],train.iloc[ix][FACTORS])
    for name in ['raw_return','residual_all','residual_no_momentum','residual_no_volume']:
        selected=cols.copy()
        if name=='residual_no_momentum':selected=[c for c in cols if not any(s in c for s in ['mom','ma_gap','relative'])]
        if name=='residual_no_volume':selected=[c for c in cols if not any(s in c for s in ['volume','illiquidity'])]
        m=lgb.LGBMRegressor(**params,verbosity=-1);m.fit(train[selected].to_numpy(),(y if name=='raw_return' else res)*100)
        models[name]=m;features[name]=selected;joblib.dump({'model':m,'features':selected,'target':name},dest/f'{name}.joblib',compress=3)
    return models,features

def scores(frame,models,features):
    s=pd.DataFrame(index=frame.index)
    s['equal_weight']=0.;s['momentum']=frame.rank_mom_12_1.to_numpy()
    s['multi_momentum']=(frame.rank_mom_12_1+frame.rank_mom_60+frame.rank_mom_120).to_numpy()/3
    s['low_vol']=-frame.rank_vol_20.to_numpy()
    s['traditional_combo']=(frame.rank_mom_12_1+frame.rank_mom_60-frame.rank_vol_20+frame.rank_reversal_5).to_numpy()/4
    for name,m in models.items():s[name]=m.predict(frame[features[name]].to_numpy())/100
    for _,ix in frame.groupby('date').indices.items():
        for col in ['residual_all','residual_no_momentum','residual_no_volume']:
            s.iloc[ix,s.columns.get_loc(col)]=neutralize(s.iloc[ix][col].to_numpy(),frame.iloc[ix][FACTORS])
    s['raw_ml']=s.raw_return
    s['residual_tilt']=s.residual_all;s['residual_no_momentum_tilt']=s.residual_no_momentum;s['residual_no_volume_tilt']=s.residual_no_volume
    s['ml_momentum_50']=s.groupby(frame.date).residual_all.transform(lambda x:zscore(x))*.5+s.groupby(frame.date).momentum.transform(lambda x:zscore(x))*.5
    return s

def main():
    if (R/'selection_lock.json').exists():raise SystemExit('Already frozen; replicate in separate directory')
    protocol=json.loads((R/'protocol.json').read_text());panel,cols,returns=data(ROOT);panel=panel.sort_values(['date','symbol']).reset_index(drop=True)
    panel.to_pickle(R/'development_panel.pkl');returns.to_pickle(R/'risk_returns.pkl');(R/'features.json').write_text(json.dumps(cols))
    old=pd.read_csv(ROOT/'research/final_predictions.csv.gz',parse_dates=['date']);old_daily=pd.read_csv(ROOT/'research/final_selected_daily.csv',parse_dates=['date'])
    f=panel[panel.date.isin(old_daily.date)].copy();fr=factor_returns(f);exposure=hac(old_daily,fr)
    joined=f.merge(old[['date','symbol','score']],on=['date','symbol'],validate='one_to_one');score_diag=[]
    for date,g in joined.groupby('date'):
        row={'date':str(date)}
        for factor in FACTORS:row[factor]=corr(g.score.rank(),g[factor].rank())
        row['residual_rank_ic']=corr(pd.Series(neutralize(g.score,g[FACTORS])).rank(),pd.Series(neutralize(g.y_20,g[FACTORS])).rank());score_diag.append(row)
    pd.DataFrame(score_diag).to_csv(R/'prior_score_exposures.csv',index=False)
    (R/'prior_strategy_factor_regression.json').write_text(json.dumps(exposure,indent=2))
    rows=[];predictive=[];regressions=[]
    for fold,(start,end,cutoff) in enumerate(protocol['development_folds']):
        train=panel[(panel.label_end_20<pd.Timestamp(cutoff))&panel.y_20.notna()].reset_index(drop=True)
        ev=panel[panel.date.between(start,end)&(panel.label_end_20<=pd.Timestamp(end))].reset_index(drop=True)
        assert ev.groupby('date').symbol.nunique().min()==87
        dest=R/f'models_fold{fold}';dest.mkdir(exist_ok=True)
        models,features=fit_models(train,cols,dest,protocol['model_params']);ss=scores(ev,models,features)
        saved=ev.copy()
        for c in ss:saved['score_'+c]=ss[c].to_numpy()
        saved.to_pickle(R/f'predictions_fold{fold}.pkl')
        factors=factor_returns(ev)
        for name in protocol['strategies']:
            target=ev.copy();target['y']=ev.y_20
            met,ic=predictive_metrics(target,ss[name]);predictive.append({'fold':fold,'strategy':name,**met})
            for risk in protocol['risk_variants']:
                d,t,w=backtest(ev,ss[name],returns,risk,neutral=name.startswith('residual_'))
                key=f'fold{fold}_{name}_{risk["name"]}';d.to_csv(R/f'{key}_daily.csv',index=False);t.to_csv(R/f'{key}_trades.csv.gz',index=False);w.to_csv(R/f'{key}_weights.csv.gz',index=False)
                rows.append({'fold':fold,'strategy':name,'risk':risk['name'],**portfolio_metrics(d),'rank_ic':met['rank_ic']})
                regressions.append({'fold':fold,'strategy':name,'risk':risk['name'],**hac(d,factors)})
                print(key,'Sharpe',rows[-1]['sharpe'],'return',rows[-1]['annualized_return'],flush=True)
            pd.DataFrame(rows).to_csv(R/'development_results.csv',index=False)
    pd.DataFrame(predictive).to_csv(R/'predictive_diagnostics.csv',index=False);(R/'development_factor_regressions.json').write_text(json.dumps(regressions,indent=2))
    print('STUDY COMPLETE',flush=True)
if __name__=='__main__':main()
