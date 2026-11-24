"""One-time sealed evaluation, no final-period selection."""
from pathlib import Path
import sys,json,time,hashlib
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import joblib
import torch
import lightgbm as lgb
import xgboost as xgb
from research.engine import load_panel,make_features,portfolio
from research.search import RankNet
from src.utils import predictive_metrics,portfolio_metrics
R=Path(__file__).resolve().parent

def main():
    if (R/'final_results.json').exists():raise SystemExit('Final already evaluated; no repeated selection')
    lock=json.loads((R/'selection_lock.json').read_text());spec=lock['spec'];h=spec['horizon'];cols=json.loads((R/'feature_names.json').read_text())
    dev=load_panel(R/'development_raw');future=load_panel(R/'final_raw');future=future[future.date>dev.date.max()]
    raw=pd.concat([dev,future]).sort_values(['symbol','date']);reference=set(dev.symbol);raw=raw[raw.symbol.isin(reference)]
    # Refuse to delete a missing security after seeing the final period.
    expected=set(lock['universe']);observed=set(future.symbol);missing=expected-observed
    if missing:raise ValueError(f'Final security coverage failure; do not substitute: {missing}')
    panel,_=make_features(raw);panel=panel[panel.symbol.isin(lock['universe'])];panel=panel.dropna(subset=cols+['date','execution_open','next_open'])
    tr=panel[(panel[f'label_end_{h}']<'2022-10-01')&panel[f'y_{h}'].notna()].sort_values(['date','symbol'])
    ev=panel[panel.date.between(lock['final_start'],lock['final_end'])&(panel[f'label_end_{h}']<=lock['final_end'])].sort_values(['date','symbol']).copy()
    counts=ev.groupby('date').symbol.nunique()
    if (counts!=len(expected)).any():raise ValueError('Incomplete final panel; missing-security policy required, not silent complete-case selection')
    yt=tr[f'y_{h}'].to_numpy();rank=spec['objective']=='ranking'
    if rank:yt=np.minimum(4,(tr.groupby('date')[f'y_{h}'].rank(pct=True).to_numpy()*5).astype(int))
    start=time.time()
    if spec['family'] in ['LightGBM','XGBoost']:
        common={'n_estimators':120,'max_depth':spec['depth'],'learning_rate':.03,'n_jobs':1,'random_state':42}
        if spec['family']=='LightGBM':
            cls=lgb.LGBMRanker if rank else lgb.LGBMRegressor;model=cls(**common,num_leaves=2**spec['depth'],min_child_samples=200,reg_lambda=10,verbosity=-1)
        else:
            cls=xgb.XGBRanker if rank else xgb.XGBRegressor;model=cls(**common,objective='rank:pairwise' if rank else 'reg:squarederror',min_child_weight=100,reg_lambda=10,subsample=.8,colsample_bytree=.8,tree_method='hist')
        model.fit(tr[cols].to_numpy(),yt,**({'group':tr.groupby('date').size().to_numpy()} if rank else {}));score=model.predict(ev[cols].to_numpy());joblib.dump(model,R/'models/final_selected.joblib',compress=3)
    else:
        ck=torch.load(R/'models'/f"{lock['trial']}_fold1.pth",map_location='cpu',weights_only=True);seqcols=ck['columns'];length=spec['lookback'];seqmap={}
        for _,g in panel.sort_values(['symbol','date']).groupby('symbol'):
            a=g[seqcols].to_numpy(dtype='float32');indices=g.index.to_numpy()
            for j in range(length-1,len(g)):seqmap[indices[j]]=a[j-length+1:j+1]
        tr=tr.loc[tr.index.isin(seqmap)];ev=ev.loc[ev.index.isin(seqmap)];xt=np.stack([seqmap[i] for i in tr.index]);xe=np.stack([seqmap[i] for i in ev.index]);del seqmap
        yt=tr[f'y_{h}'].to_numpy()*100
        if spec['objective']=='rank_regression':yt=tr.groupby('date')[f'y_{h}'].rank(pct=True).to_numpy()-.5
        torch.manual_seed(42);torch.set_num_threads(6);device='mps' if torch.backends.mps.is_available() else 'cpu';model=RankNet(spec['family'],len(seqcols),length).to(device)
        opt=torch.optim.AdamW(model.parameters(),lr=.0003,weight_decay=.01);rng=np.random.default_rng(42)
        for epoch in range(lock['epochs']):
            model.train();order=rng.permutation(len(xt))
            for j in range(0,len(order),512):
                ix=order[j:j+512];a=torch.from_numpy(xt[ix]).to(device);b=torch.tensor(yt[ix],dtype=torch.float32,device=device);opt.zero_grad();loss=((model(a)-b)**2).mean();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),.5);opt.step()
        model.eval();pred=[]
        with torch.no_grad():
            for j in range(0,len(xe),512):pred.append(model(torch.from_numpy(xe[j:j+512]).to(device)).cpu().numpy())
        score=np.concatenate(pred);score=score/100 if spec['objective']=='regression' else score
        torch.save({'state_dict':model.cpu().state_dict(),'spec':spec,'columns':seqcols,'epochs':lock['epochs']},R/'models/final_selected.pth')
    ev['score']=score;ev['y']=ev[f'y_{h}'];ev.to_csv(R/'final_predictions.csv.gz',index=False)
    metric,ic=predictive_metrics(ev.reset_index(drop=True),score)
    if spec['objective']!='regression':
        for key in ['rmse','mae','r2','directional_accuracy']:metric[key]=None
    ic.to_csv(R/'final_ic.csv',index=False)
    results={};daily_map={}
    for name,scores,top,mode in [('selected',score,20,lock['portfolio']),('equal_weight',np.zeros(len(ev)),len(expected),'equal'),('momentum',ev.mom_12_1.to_numpy(),20,'buffer'),('reversal',ev.reversal_5.to_numpy(),20,'equal')]:
        d,t=portfolio(ev,scores,rebalance=h,buffer=mode=='buffer',inverse_vol=mode=='inverse_vol',top=top)
        d.to_csv(R/f'final_{name}_daily.csv',index=False);t.to_csv(R/f'final_{name}_trades.csv.gz',index=False);results[name]=portfolio_metrics(d);daily_map[name]=d
    # Exact same observations: zero and training-mean forecast baselines for regression interpretation.
    baselines={}
    for name,p in [('zero',np.zeros(len(ev))),('historical_mean',np.full(len(ev),tr[f'y_{h}'].mean()))]:baselines[name]=predictive_metrics(ev.reset_index(drop=True),p)[0]
    delta=daily_map['selected'].net_return.to_numpy()-daily_map['equal_weight'].net_return.to_numpy();n=len(delta);rng=np.random.default_rng(42);draws=[]
    for _ in range(5000):
        starts=rng.integers(0,n,size=(n+19)//20);ix=np.concatenate([(s+np.arange(20))%n for s in starts])[:n];draws.append(delta[ix].mean()*252)
    final={'selection':lock,'predictive':metric,'baseline_forecasts':baselines,'portfolio':results,'active_mean_return_annualized':float(delta.mean()*252),'active_block_bootstrap_95':np.quantile(draws,[.025,.975]).tolist(),'days':n,'observations':len(ev),'fit_seconds':time.time()-start,'period':[str(ev.date.min()),str(ev.date.max())],'final_data_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (R/'final_raw').glob('*.csv')}}
    def clean(v):
        if isinstance(v,dict):return {k:clean(x) for k,x in v.items()}
        if isinstance(v,list):return [clean(x) for x in v]
        if isinstance(v,(float,np.floating)) and not np.isfinite(v):return None
        return v
    (R/'final_results.json').write_text(json.dumps(clean(final),indent=2));print(json.dumps(clean(final),indent=2))
if __name__=='__main__':main()
