"""Bounded validation-only search. No final raw directory access."""
import sys,json,time,hashlib,copy
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import torch
from torch import nn
import joblib
import lightgbm as lgb
import xgboost as xgb
from research.engine import load_panel,make_features,portfolio
from src.utils import predictive_metrics,portfolio_metrics
R=Path(__file__).resolve().parent

class RankNet(nn.Module):
    def __init__(self,kind,n,length):
        super().__init__();self.kind=kind;self.norm=nn.LayerNorm(n)
        if kind=='LSTM':self.enc=nn.LSTM(n,48,2,batch_first=True,dropout=.2);self.head=nn.Sequential(nn.LayerNorm(48),nn.Dropout(.2),nn.Linear(48,1))
        else:
            self.proj=nn.Linear(n,48);self.pos=nn.Parameter(torch.zeros(1,length,48));nn.init.normal_(self.pos,std=.02)
            self.enc=nn.TransformerEncoder(nn.TransformerEncoderLayer(48,4,96,.2,batch_first=True,norm_first=True),2,enable_nested_tensor=False)
            self.head=nn.Sequential(nn.LayerNorm(48),nn.Dropout(.2),nn.Linear(48,1))
    def forward(self,x):
        x=self.norm(x)
        h=self.enc(x)[0][:,-1] if self.kind=='LSTM' else self.enc(self.proj(x)+self.pos).mean(dim=1)
        return self.head(h).squeeze(-1)

def neural_fit(kind,length,xt,yt,xv,yv,objective,epochs=4):
    torch.manual_seed(42);np.random.seed(42);torch.set_num_threads(6);device='mps' if torch.backends.mps.is_available() else 'cpu'
    m=RankNet(kind,xt.shape[-1],length).to(device);opt=torch.optim.AdamW(m.parameters(),lr=.0003,weight_decay=.01);rng=np.random.default_rng(42)
    history=[];best=None;bestloss=np.inf
    def predict(x):
        out=[];m.eval()
        with torch.no_grad():
            for j in range(0,len(x),512):out.append(m(torch.from_numpy(x[j:j+512]).to(device)).cpu().numpy())
        return np.concatenate(out)
    for epoch in range(1,epochs+1):
        m.train();ix=rng.permutation(len(xt));losses=[]
        for j in range(0,len(ix),512):
            ids=ix[j:j+512];a=torch.from_numpy(xt[ids]).to(device);b=torch.from_numpy(yt[ids].astype('float32')).to(device)
            opt.zero_grad();pred=m(a);loss=((pred-b)**2).mean();loss.backward();nn.utils.clip_grad_norm_(m.parameters(),.5);opt.step();losses.append(loss.item())
        pv=predict(xv);vl=float(np.mean((pv-yv)**2));history.append({'epoch':epoch,'train_loss':float(np.mean(losses)),'validation_loss':vl})
        if vl<bestloss:bestloss=vl;best=copy.deepcopy({k:v.cpu() for k,v in m.state_dict().items()});bestepoch=epoch
        print(kind,length,objective,history[-1],flush=True)
    m.load_state_dict(best);return m,predict(xv),history,bestepoch

def prepare():
    raw=load_panel(R/'development_raw');panel,cols=make_features(raw)
    # Eligibility fixed using development history only; explicitly report survivor conditioning.
    eligible=[];excluded=[]
    dates=sorted(raw.loc[raw.date.between('2021-01-01','2023-12-31'),'date'].unique())
    for s,g in raw.groupby('symbol'):
        seen=g[g.date.between('2021-01-01','2023-12-31')]
        if len(seen)==len(dates) and seen[['open','close']].notna().all().all() and g.date.min()<pd.Timestamp('2015-01-01'):eligible.append(s)
        else:excluded.append({'symbol':s,'reason':'incomplete development validation calendar or insufficient pre-2015 history','first':str(g.date.min()),'last':str(g.date.max())})
    panel=panel[panel.symbol.isin(eligible)].copy();panel=panel.dropna(subset=cols+['date','execution_open','next_open'])
    (R/'eligibility.json').write_text(json.dumps({'eligible':eligible,'excluded':excluded,'warning':'Conditioning on 2021-2023 completeness adds development survivor bias; final universe fixed before reveal'},indent=2))
    panel.to_pickle(R/'development_panel.pkl');(R/'feature_names.json').write_text(json.dumps(cols,indent=2));return panel,cols

def main():
    if (R/'selection_lock.json').exists():raise SystemExit('Completed selection exists. Replicate in a separate local copy; do not overwrite frozen research.')
    (R/'models').mkdir(exist_ok=True);panel,cols=prepare();print('PANEL',len(panel),'FEATURES',len(cols),'SYMBOLS',panel.symbol.nunique(),flush=True)
    # 2021 first validation, 2022-2023 second. Fixed training cutoff before entire raw lookback (252+60).
    specs=[]
    for family in ['LightGBM','XGBoost']:
        for objective in ['regression','ranking']:
            for h in [1,5,20]:
                for depth in [3,6]:specs.append({'family':family,'objective':objective,'horizon':h,'depth':depth})
    for kind in ['LSTM','Transformer']:
        for length in [20,60]:
            for objective in ['regression','rank_regression']:specs.append({'family':kind,'objective':objective,'horizon':5,'lookback':length})
    (R/'trial_plan.json').write_text(json.dumps(specs,indent=2))
    results=[];predictions=[];histories=[]
    for fold,(start,end,cutoff) in enumerate([('2021-01-01','2021-12-31','2019-10-01'),('2022-01-01','2023-12-31','2020-10-01')]):
        for num,spec in enumerate(specs):
            ident=f"trial_{num:02}";h=spec['horizon'];t0=time.time()
            tr=panel[(panel[f'label_end_{h}']<cutoff)&panel[f'y_{h}'].notna()].copy()
            va=panel[panel.date.between(start,end)&(panel[f'label_end_{h}']<=end)&panel[f'y_{h}'].notna()].copy()
            # full available validation dates for this horizon; comparisons reported within horizon.
            tr=tr.sort_values(['date','symbol']);va=va.sort_values(['date','symbol'])
            yt=tr[f'y_{h}'].to_numpy();yv=va[f'y_{h}'].to_numpy()
            if spec['family'] in ['LightGBM','XGBoost']:
                xt=tr[cols].to_numpy();xv=va[cols].to_numpy()
                rank=spec['objective']=='ranking';groups=tr.groupby('date',sort=True).size().to_numpy()
                if rank:yt=np.minimum(4,(tr.groupby('date')[f'y_{h}'].rank(pct=True).to_numpy()*5).astype(int))
                common={'n_estimators':120,'max_depth':spec['depth'],'learning_rate':.03,'n_jobs':1,'random_state':42}
                if spec['family']=='LightGBM':
                    cls=lgb.LGBMRanker if rank else lgb.LGBMRegressor
                    model=cls(**common,num_leaves=2**spec['depth'],min_child_samples=200,reg_lambda=10,verbosity=-1)
                    model.fit(xt,yt,**({'group':groups} if rank else {}))
                else:
                    cls=xgb.XGBRanker if rank else xgb.XGBRegressor
                    model=cls(**common,objective='rank:pairwise' if rank else 'reg:squarederror',min_child_weight=100,reg_lambda=10,subsample=.8,colsample_bytree=.8,tree_method='hist')
                    model.fit(xt,yt,**({'group':groups} if rank else {}))
                pv=model.predict(xv);joblib.dump(model,R/'models'/f'{ident}_fold{fold}.joblib',compress=3)
            else:
                length=spec['lookback'];seqcols=['rank_mom_1','rank_mom_5','rank_mom_20','rank_mom_60','rank_vol_20','rank_volume_ratio_20','rank_intraday','rank_overnight','market_ret','market_vol']
                seqmap={}
                for symbol,g in panel.sort_values(['symbol','date']).groupby('symbol'):
                    values=g[seqcols].to_numpy(dtype='float32');indices=g.index.to_numpy()
                    for j in range(length-1,len(g)):seqmap[indices[j]]=values[j-length+1:j+1]
                tr=tr.loc[tr.index.isin(seqmap)];va=va.loc[va.index.isin(seqmap)]
                xt=np.stack([seqmap[i] for i in tr.index]);xv=np.stack([seqmap[i] for i in va.index]);del seqmap
                yt=tr[f'y_{h}'].to_numpy()*100;yv=va[f'y_{h}'].to_numpy()*100
                if spec['objective']=='rank_regression':
                    yt=tr.groupby('date')[f'y_{h}'].rank(pct=True).to_numpy()-.5;yv=va.groupby('date')[f'y_{h}'].rank(pct=True).to_numpy()-.5
                model,pv,history,bestepoch=neural_fit(spec['family'],length,xt,yt,xv,yv,spec['objective'])
                histories.append({'trial':ident,'fold':fold,'history':history});
                torch.save({'state_dict':model.cpu().state_dict(),'spec':spec,'columns':seqcols,'epochs':bestepoch},R/'models'/f'{ident}_fold{fold}.pth')
                if spec['objective']=='regression':pv=pv/100
                del xt,xv,model
            # No portfolio comparison can mix horizons silently; full ledger retained for each trial.
            scored=va.copy();scored['score']=pv;scored['y']=scored[f'y_{h}']
            met,ic=predictive_metrics(scored.reset_index(drop=True),pv)
            for mode in ['equal','buffer','inverse_vol']:
                daily,trades=portfolio(scored,pv,rebalance=h,buffer=mode=='buffer',inverse_vol=mode=='inverse_vol')
                pm=portfolio_metrics(daily)
                # Same dates and same universe, 95% equal-weight comparator, same rebalance horizon/cost.
                bench,_=portfolio(scored,np.zeros(len(scored)),rebalance=h,top=scored.symbol.nunique())
                bm=portfolio_metrics(bench)
                row={'trial':ident,'fold':fold,'portfolio':mode,**spec,**met,**pm,'benchmark_annualized':bm['annualized_return'],'active_annualized':pm['annualized_return']-bm['annualized_return'],'fit_seconds':time.time()-t0}
                results.append(row)
                daily.to_csv(R/f'{ident}_fold{fold}_{mode}_daily.csv',index=False)
                if mode=='buffer':trades.to_csv(R/f'{ident}_fold{fold}_buffer_trades.csv.gz',index=False)
            scored[['date','signal_date','symbol',f'y_{h}','score']].to_csv(R/f'{ident}_fold{fold}_predictions.csv.gz',index=False)
            pd.DataFrame(results).to_csv(R/'validation_trials.csv',index=False);(R/'neural_history.json').write_text(json.dumps(histories,indent=2))
            print('DONE',ident,fold,spec,'rank_ic',met['rank_ic'],flush=True)
    print('VALIDATION SEARCH COMPLETE',flush=True)
if __name__=='__main__':main()
