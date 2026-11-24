"""Small, explicitly specified neural models and deterministic training."""
import copy
import time
import numpy as np
import torch
from torch import nn

class LSTM(nn.Module):
    def __init__(self,n_features=8):
        super().__init__()
        self.encoder=nn.LSTM(n_features,32,batch_first=True)
        self.head=nn.Linear(32,1)
    def forward(self,x):
        return self.head(self.encoder(x)[0][:,-1]).squeeze(-1)

class Transformer(nn.Module):
    def __init__(self,n_features=8):
        super().__init__()
        self.proj=nn.Linear(n_features,32)
        pos=torch.arange(20).float()[:,None]
        div=torch.exp(torch.arange(0,32,2).float()*(-np.log(10000)/32))
        pe=torch.zeros(20,32);pe[:,0::2]=torch.sin(pos*div);pe[:,1::2]=torch.cos(pos*div)
        self.register_buffer('pe',pe)
        self.encoder=nn.TransformerEncoder(nn.TransformerEncoderLayer(32,4,64,0.1,batch_first=True),1,enable_nested_tensor=False)
        self.head=nn.Linear(32,1)
    def forward(self,x):
        # All tokens precede execution; bidirectional attention inside this past window is causal.
        return self.head(self.encoder(self.proj(x)+self.pe)[ :,-1]).squeeze(-1)

def choose_device():
    if torch.cuda.is_available():return 'cuda'
    if torch.backends.mps.is_available():return 'mps'
    return 'cpu'

def predict(model,x,device,batch=2048):
    model.eval(); out=[]
    with torch.no_grad():
        for i in range(0,len(x),batch):
            out.append(model(torch.from_numpy(x[i:i+batch]).to(device)).cpu().numpy()/100)
    return np.concatenate(out)

def train(name,x,y,device,epochs,validation=None):
    torch.manual_seed(42);np.random.seed(42)
    model=globals()[name]().to(device)
    optimizer=torch.optim.Adam(model.parameters(),lr=0.001)
    rng=np.random.default_rng(42);history=[];best=None;best_loss=float('inf');best_epoch=epochs
    start=time.perf_counter()
    for epoch in range(1,epochs+1):
        model.train();order=rng.permutation(len(x));total=0
        for i in range(0,len(x),1024):
            ix=order[i:i+1024]
            bx=torch.from_numpy(x[ix]).to(device);by=torch.from_numpy((y[ix]*100).astype('float32')).to(device)
            optimizer.zero_grad();loss=((model(bx)-by)**2).mean();loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),1);optimizer.step()
            total+=loss.item()*len(ix)
        row={'epoch':epoch,'train_mse':total/len(x)/10000}
        if validation is not None:
            vx,vy=validation;pred=predict(model,vx,device);val=float(np.mean((pred-vy)**2));row['validation_mse']=val
            if val<best_loss:
                best_loss=val;best_epoch=epoch;best=copy.deepcopy({k:v.cpu() for k,v in model.state_dict().items()})
        history.append(row);print(name,row,flush=True)
    if best is not None:model.load_state_dict(best)
    return model,history,best_epoch,time.perf_counter()-start
