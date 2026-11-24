"""Run validation first, freeze choices, then expanding-window prequential test."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import time
import warnings
import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestRegressor
from statsmodels.tsa.arima.model import ARIMA
from .data_preprocessing import build_panel, fit_scaler, split_indices
from .models import choose_device, train, predict
from .utils import predictive_metrics, portfolio_metrics
from .backtest import backtest, buy_hold

MODELS=['zero','historical_mean','ARIMA','RandomForest','LSTM','Transformer']

def write_json(path,obj):
    def sanitize(v):
        if isinstance(v,dict):return {str(k):sanitize(x) for k,x in v.items()}
        if isinstance(v,(list,tuple)):return [sanitize(x) for x in v]
        if isinstance(v,(np.integer,)):return int(v)
        if isinstance(v,(float,np.floating)):return float(v) if np.isfinite(v) else None
        if isinstance(v,(pd.Timestamp,np.datetime64)):return str(v)
        return v
    Path(path).write_text(json.dumps(sanitize(obj),indent=2,allow_nan=False))

def fit_arima(meta,train_ix,eval_ix,destination):
    tr=meta.iloc[train_ix];ev=meta.iloc[eval_ix];pooled=float(tr.y.mean())
    predictions=np.full(len(ev),pooled);status=[];params={}
    groups=ev.groupby('symbol').indices
    for symbol,indices in groups.items():
        series=tr.loc[tr.symbol==symbol].set_index('date').y
        calendar=pd.DatetimeIndex(sorted(tr.date.unique()))
        y=series.reindex(calendar).to_numpy()*100
        if np.isfinite(y).sum()<100:
            status.append({'symbol':symbol,'status':'insufficient_history_pooled_mean','n':len(y)});continue
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            fitted=ARIMA(y,order=(1,0,0),trend='c').fit()
        c,phi,sigma=map(float,fitted.params)
        predictions[indices]=(c+phi*(ev.iloc[indices].last_intraday.to_numpy()*100-c))/100
        params[symbol]={'mean_percent':c,'ar1':phi,'sigma2':sigma}
        status.append({'symbol':symbol,'n':int(np.isfinite(y).sum()),'missing_calendar_sessions':int(np.isnan(y).sum()),'status':'fit','converged':bool(fitted.mle_retvals.get('converged',False)),
                       'warnings':[str(w.message) for w in caught]})
    write_json(destination,{'order':[1,0,0],'params':params,'fallback_mean':pooled,'status':status})
    return predictions

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',default='results');parser.add_argument('--data',default='data/raw/all_stocks_5yr.csv');args=parser.parse_args()
    out=Path(args.output)
    if (out/'run_manifest.json').exists():raise SystemExit('Use a fresh --output directory; completed runs are immutable')
    out.mkdir(exist_ok=True,parents=True);(out/'models').mkdir(exist_ok=True)
    protocol=json.loads(Path('protocol.json').read_text());write_json(out/'protocol.json',protocol)
    torch.set_num_threads(6)
    device=choose_device();print('DEVICE',device,flush=True)
    started=time.time();raw=pd.read_csv(args.data)
    raw['date']=pd.to_datetime(raw.date)
    audit={'rows':len(raw),'symbols':raw.Name.nunique(),'start':str(raw.date.min()),'end':str(raw.date.max()),
           'missing':raw.isna().sum().to_dict(),'duplicate_keys':int(raw.duplicated(['Name','date']).sum())}
    x,meta,counts=build_panel(raw);audit.update(counts)
    print('PANEL',x.shape,audit,flush=True);write_json(out/'data_quality.json',audit)
    selections={};history=[];timings=[];fold_frames=[];split_audit=[]
    folds=[('validation',*protocol['validation'])]+[(f'test_{i+1}',*dates) for i,dates in enumerate(protocol['test_folds'])]
    for label,start,end in folds:
        ti,ei,cutoff=split_indices(meta,start,end);validation=label=='validation'
        print('FOLD',label,'train',len(ti),'evaluate',len(ei),'cutoff',cutoff,flush=True)
        split_audit.append({'fold':label,'n_train':len(ti),'n_evaluate':len(ei),'max_train_target':meta.iloc[ti].date.max(),
                            'min_evaluation_feature_start':meta.iloc[ei].feature_start.min(),'evaluation_start':meta.iloc[ei].date.min(),
                            'evaluation_end':meta.iloc[ei].date.max()})
        mean,std=fit_scaler(x[ti]);np.savez(out/'models'/f'{label}_scaler.npz',mean=mean,std=std)
        xt=(x[ti]-mean)/std;xe=(x[ei]-mean)/std;yt=meta.iloc[ti].y.to_numpy();ye=meta.iloc[ei].y.to_numpy()
        frame=meta.iloc[ei].copy().reset_index(drop=True);frame['fold']=label
        frame['zero']=0.;frame['historical_mean']=yt.mean()
        write_json(out/'models'/f'{label}_mean.json',{'mean':float(yt.mean()),'n':len(yt)})
        t=time.time();frame['ARIMA']=fit_arima(meta,ti,ei,out/'models'/f'{label}_ARIMA.json')
        timings.append({'fold':label,'model':'ARIMA','seconds':time.time()-t})
        rf_train=xt.reshape(len(xt),-1);rf_eval=xe.reshape(len(xe),-1)
        candidates=protocol['rf_candidates'] if validation else [selections['RandomForest']]
        best_loss=float('inf');best_rf=None;best_pred=None
        for c in candidates:
            t=time.time();rf=RandomForestRegressor(n_estimators=protocol['rf_n_estimators'],random_state=42,n_jobs=6,max_features='sqrt',**c)
            rf.fit(rf_train,yt);p=rf.predict(rf_eval)
            timings.append({'fold':label,'model':'RandomForest','config':c,'seconds':time.time()-t})
            if validation:
                loss=float(np.mean((ye-p)**2));history.append({'fold':label,'model':'RandomForest',**c,'validation_mse':loss})
            else:loss=0
            if loss<best_loss:
                best_loss=loss;best_rf=rf;best_pred=p
                if validation:selections['RandomForest']=c
            print('RF',label,c,'done',flush=True)
        joblib.dump(best_rf,out/'models'/f'{label}_RandomForest.joblib',compress=3);frame['RandomForest']=best_pred
        for name in ['LSTM','Transformer']:
            epochs=protocol['neural_max_epochs'] if validation else selections[name]['epochs']
            model,h,best_epoch,seconds=train(name,xt,yt,device,epochs,(xe,ye) if validation else None)
            history.extend([{'fold':label,'model':name,**row} for row in h]);timings.append({'fold':label,'model':name,'seconds':seconds})
            if validation:selections[name]={'epochs':best_epoch}
            frame[name]=predict(model,xe,device)
            torch.save({'state_dict':{k:v.cpu() for k,v in model.state_dict().items()},'model':name,'epochs':best_epoch,
                        'n_features':8,'lookback':20,'seed':42},out/'models'/f'{label}_{name}.pth')
            del model
        assert np.isfinite(frame[MODELS].to_numpy()).all()
        frame.to_csv(out/f'predictions_{label}.csv.gz',index=False)
        fold_frames.append(frame)
        if validation:
            write_json(out/'selection_frozen.json',{'selections':selections,'protocol_sha256':hashlib.sha256(Path('protocol.json').read_bytes()).hexdigest(),
                       'frozen_before_test':True,'strategy_threshold':protocol['threshold']})
        write_json(out/'split_audit.json',split_audit);write_json(out/'training_history.json',history);write_json(out/'timings.json',timings)
        del xt,xe,rf_train,rf_eval,best_rf
    test=pd.concat(fold_frames[1:],ignore_index=True);test.to_csv(out/'predictions_test.csv.gz',index=False)
    metrics=[];portfolio=[]
    for label,frame in [('validation',fold_frames[0])]+[(f'test_{i+1}',f) for i,f in enumerate(fold_frames[1:])]+[('test',test)]:
        for name in MODELS:
            m,ic=predictive_metrics(frame,frame[name]);metrics.append({'split':label,'model':name,**m})
            ic.to_csv(out/f'ic_{label}_{name}.csv',index=False)
            daily,trades=backtest(frame,frame[name],threshold=protocol['threshold'],cost=protocol['one_way_cost'])
            portfolio.append({'split':label,'model':name,**portfolio_metrics(daily)})
            daily.to_csv(out/f'portfolio_{label}_{name}.csv',index=False)
            trades.to_csv(out/f'trades_{label}_{name}.csv.gz',index=False)
    # Entire 2016 only: basket fixed before test begins.
    v=raw[(raw.date>='2016-01-01')&(raw.date<='2016-12-31')].copy();v['dollar_volume']=v.close*v.volume
    symbols=v.groupby('Name').dollar_volume.median().nlargest(20).index.tolist()
    dates=pd.DatetimeIndex(sorted(test.date.unique()));bh=buy_hold(raw,dates,symbols)
    bh.to_csv(out/'portfolio_test_buy_hold.csv',index=False);write_json(out/'benchmark_basket.json',symbols)
    portfolio.append({'split':'test','model':'buy_hold',**portfolio_metrics(bh)})
    pd.DataFrame(metrics).to_csv(out/'model_comparison.csv',index=False);pd.DataFrame(portfolio).to_csv(out/'portfolio_comparison.csv',index=False)
    # Prespecified cost stress test; never select thresholds/models from these test outcomes.
    sensitivity=[]
    for cost in [0.,0.0005,0.0015,0.003]:
        for name in MODELS:
            daily,_=backtest(test,test[name],threshold=protocol['threshold'],cost=cost)
            sensitivity.append({'model':name,'one_way_cost':cost,**portfolio_metrics(daily)})
    pd.DataFrame(sensitivity).to_csv(out/'cost_sensitivity.csv',index=False)
    rng=np.random.default_rng(42);uncertainty=[]
    for name in MODELS[1:]:
        diff=(test.y**2-(test.y-test[name])**2).groupby(test.date).mean().to_numpy();n=len(diff)
        draws=[]
        for _ in range(2000):
            starts=rng.integers(0,n,size=(n+19)//20);ix=np.concatenate([(s+np.arange(20))%n for s in starts])[:n]
            draws.append(diff[ix].mean())
        uncertainty.append({'model':name,'mean_daily_mse_improvement_over_zero':float(diff.mean()),'ci_2.5':float(np.quantile(draws,.025)),'ci_97.5':float(np.quantile(draws,.975))})
    pd.DataFrame(uncertainty).to_csv(out/'uncertainty.csv',index=False)
    write_json(out/'key_results.json',{'predictive':[m for m in metrics if m['split']=='test'],'portfolio':[m for m in portfolio if m['split']=='test']})
    manifest={'completed':True,'device':device,'torch':torch.__version__,'python':platform.python_version(),'platform':platform.platform(),
              'seconds':time.time()-started,'data_sha256':hashlib.sha256(Path(args.data).read_bytes()).hexdigest(),
              'protocol_sha256':hashlib.sha256(Path('protocol.json').read_bytes()).hexdigest(),
              'source_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in Path('src').glob('*.py')}}
    write_json(out/'run_manifest.json',manifest);print('COMPLETE',manifest,flush=True)

if __name__=='__main__':main()
