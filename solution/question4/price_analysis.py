#!/usr/bin/env python3
from __future__ import annotations
from pathlib import Path
from time import perf_counter
import csv, json, warnings
import numpy as np
from openpyxl import load_workbook
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
try:                     # torch 未安装时跳过 LSTM/GRU/TCN（参考包同样记录了该依赖不可用）
    import torch
    from torch import nn
except ImportError:      # pragma: no cover
    torch = None
    nn = None

warnings.filterwarnings('ignore')
HERE=Path(__file__).resolve().parent; ROOT=HERE.parents[1]; OUT=HERE/'output';OUT.mkdir(parents=True,exist_ok=True)
START=31;DAYS=365;N=144
np.random.seed(2026)
if torch is not None:torch.manual_seed(2026);torch.set_num_threads(2)

def load_data():
    def mat(path,sheet=0):
        w=load_workbook(path,read_only=True,data_only=True);s=w.worksheets[sheet]
        dates=[];a=[]
        for r in s.iter_rows(min_row=2,values_only=True):dates.append(r[0]);a.append([float(x) for x in r[1:]])
        return dates,np.asarray(a,float)
    dates,p=mat(ROOT/'附件/附件4.xlsx')
    _,L=mat(ROOT/'附件/附件2.xlsx',0);_,PV=mat(ROOT/'附件/附件2.xlsx',1)
    return dates,p,L,PV

def cal(d):
    return np.array([1,d/365,np.sin(2*np.pi*d/365),np.cos(2*np.pi*d/365),
                     np.sin(4*np.pi*d/365),np.cos(4*np.pi*d/365),
                     np.sin(2*np.pi*(d%7)/7),np.cos(2*np.pi*(d%7)/7)])
def previous_day(Y,d):return Y[d-1]
def persistence(Y,d):
    # last observed price persisted; at 0:00 this is previous day's closing level
    return np.full(N,Y[d-1,-1])
def previous_week(Y,d):return Y[d-7]
def moving_average(Y,d):return Y[max(0,d-7):d].mean(0)
def exp_smooth(Y,d,a=.28):
    z=Y[0].copy()
    for i in range(1,d):z=a*Y[i]+(1-a)*z
    return z
def arima_curve(Y,d):
    mu=Y[:d-1].mean(0);a=(Y[:d-1]-mu).ravel();b=(Y[1:d]-mu).ravel();phi=np.dot(a,b)/(np.dot(a,a)+1e-12)
    return mu+phi*(Y[d-1]-mu)
def sarima_curve(Y,d):
    ix=np.arange(7,d);X=np.c_[np.ones(len(ix)*N),Y[ix-1].ravel(),Y[ix-7].ravel()];y=Y[ix].ravel();b=np.linalg.lstsq(X,y,rcond=None)[0]
    return b[0]+b[1]*Y[d-1]+b[2]*Y[d-7]
def fourier(Y,d):
    X=np.array([cal(i) for i in range(d)]);B=np.linalg.lstsq(X,Y[:d],rcond=None)[0];return cal(d)@B
def fourier_ar(Y,d):
    base=fourier(Y,d);X=np.array([cal(i) for i in range(d)]);B=np.linalg.lstsq(X,Y[:d],rcond=None)[0]
    res=Y[:d]-X@B
    if d<3:return base
    a=res[:-1].ravel();b=res[1:].ravel();phi=np.dot(a,b)/(np.dot(a,a)+1e-12)
    return base+phi*res[-1]
def prophet_like(Y,d):
    # Causal additive trend + yearly/weekly Fourier seasonality; no external Prophet dependency.
    X=np.array([cal(i) for i in range(d)]);B=np.linalg.lstsq(X,Y[:d],rcond=None)[0];return cal(d)@B
def ar_garch(Y,d):
    # AR conditional mean with EWMA/GARCH-style volatility shrinkage; point forecast is the conditional mean.
    pred=arima_curve(Y,d);res=Y[1:d]-np.array([arima_curve(Y,i) for i in range(2,d+1)]) if d>3 else Y[1:d]-Y[:d-1]
    if len(res):
        v=np.zeros(N);w=.94
        for x in res:v=w*v+(1-w)*x*x
        pred += .02*np.sqrt(v)*np.sign(Y[d-1]-Y[max(0,d-2)])
    return pred

def walk_rf(Y):
    pred=np.zeros((DAYS-START,N));model=None
    for d in range(START,DAYS):
        if model is None or (d-START)%28==0:
            ix=np.arange(7,d);X=np.array([np.r_[cal(i),Y[i-1],Y[i-7]] for i in ix]);model=RandomForestRegressor(n_estimators=80,min_samples_leaf=2,max_features=.5,n_jobs=1,random_state=2026).fit(X,Y[ix])
        pred[d-START]=model.predict(np.r_[cal(d),Y[d-1],Y[d-7]][None])[0]
    return pred

def walk_histgb(Y,quantile=None):
    pred=np.zeros((DAYS-START,N));model=None
    for d in range(START,DAYS):
        if model is None or (d-START)%28==0:
            ix=np.arange(max(7,d-120),d);slot=np.arange(N)
            X=[];y=[]
            for i in ix:
                X.append(np.c_[np.tile(cal(i),(N,1)),slot,Y[i-1],Y[i-7]]) ;y.append(Y[i])
            X=np.vstack(X);y=np.concatenate(y)
            kw=dict(max_iter=100,max_leaf_nodes=31,learning_rate=.07,random_state=2026)
            if quantile is None:model=HistGradientBoostingRegressor(loss='squared_error',**kw)
            else:model=HistGradientBoostingRegressor(loss='quantile',quantile=quantile,**kw)
            model.fit(X,y)
        x=np.c_[np.tile(cal(d),(N,1)),np.arange(N),Y[d-1],Y[d-7]];pred[d-START]=model.predict(x)
    return pred

_NNBase = nn.Module if nn is not None else object
class SeqNet(_NNBase):
    def __init__(self,kind):
        super().__init__();self.kind=kind
        if kind=='lstm':self.r=nn.LSTM(N,20,batch_first=True)
        elif kind=='gru':self.r=nn.GRU(N,20,batch_first=True)
        else:self.c=nn.Sequential(nn.Conv1d(N,20,3,padding=1),nn.ReLU(),nn.Conv1d(20,20,3,padding=1),nn.ReLU())
        self.o=nn.Linear(20,N)
    def forward(self,x):
        if self.kind in ('lstm','gru'):z,_=self.r(x);return self.o(z[:,-1])
        return self.o(self.c(x.transpose(1,2)).mean(2))
def walk_deep(Y,kind):
    out=np.zeros((DAYS-START,N));model=None;seq=14
    for d in range(START,DAYS):
        if model is None or (d-START)%112==0:
            mu=Y[:d].mean();sd=Y[:d].std()+1e-8;A=(Y[:d]-mu)/sd
            X=np.array([A[i-seq:i] for i in range(seq,d)],np.float32);y=np.array([A[i] for i in range(seq,d)],np.float32)
            model=SeqNet(kind);opt=torch.optim.Adam(model.parameters(),lr=.008);xt=torch.tensor(X);yt=torch.tensor(y)
            for _ in range(15):opt.zero_grad();loss=((model(xt)-yt)**2).mean();loss.backward();opt.step()
        with torch.no_grad():out[d-START]=model(torch.tensor(((Y[d-seq:d]-mu)/sd)[None].astype(np.float32))).numpy()[0]*sd+mu
    return out

def eda(dates,p,L,PV):
    flat=p.ravel();net=(L-PV).ravel();
    def ac(x,k):return float(np.corrcoef(x[:-k],x[k:])[0,1])
    ret=np.diff(flat);sq=ret*ret
    corr=np.corrcoef(np.c_[flat,L.ravel(),PV.ravel(),net].T)
    monthly={};
    for m in range(1,13):monthly[str(m)]=float(np.mean([p[i] for i,x in enumerate(dates) if x.month==m]))
    dow={str(k):float(p[[i for i,x in enumerate(dates) if x.weekday()==k]].mean()) for k in range(7)}
    out={'mean':float(flat.mean()),'std':float(flat.std()),'min':float(flat.min()),'max':float(flat.max()),'q01':float(np.quantile(flat,.01)),'q99':float(np.quantile(flat,.99)),
         'acf_10min':ac(flat,1),'acf_1h':ac(flat,6),'acf_1day':ac(flat,144),'acf_1week':ac(flat,1008),'squared_return_acf_10min':ac(sq,1),'monthly_mean':monthly,'weekday_mean':dow,'corr_price_load_pv_net':corr.tolist(),
         'intraday_mean':p.mean(0).tolist(),'intraday_std':p.std(0).tolist()}
    (OUT/'price_eda.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
    return out

def main():
    dates,Y,L,PV=load_data();print(json.dumps(eda(dates,Y,L,PV),ensure_ascii=False)[:1000])
    funcs={'persistence':persistence,'previous_day':previous_day,'previous_week_same_time':previous_week,'moving_average_7':moving_average,'exponential_smoothing':exp_smooth,'ARIMA_curve':arima_curve,'SARIMA_curve':sarima_curve,'Fourier':fourier,'Fourier_AR':fourier_ar,'Prophet_like':prophet_like,'AR_GARCH':ar_garch}
    preds={};times={}
    for name,fn in funcs.items():
        tic=perf_counter();preds[name]=np.array([fn(Y,d) for d in range(START,DAYS)]);times[name]=perf_counter()-tic;print(name,times[name])
    for name,fn in [('Random_Forest',lambda:walk_rf(Y)),('HistGB_XGB_fallback',lambda:walk_histgb(Y)),('Quantile_HistGB_LGBM_fallback',lambda:walk_histgb(Y,.8))]:
        tic=perf_counter();preds[name]=fn();times[name]=perf_counter()-tic;print(name,times[name])
    if torch is not None:
        for name in ['lstm','gru','tcn']:
            tic=perf_counter();preds[name.upper()]=walk_deep(Y,name);times[name.upper()]=perf_counter()-tic;print(name,times[name.upper()])
    else:
        print('torch unavailable -> LSTM/GRU/TCN skipped（与参考包记录的依赖限制一致）')
    # Causal inverse-MAE ensemble from earlier completed test days only.
    pool=['previous_day','previous_week_same_time','moving_average_7','Fourier_AR','Random_Forest','HistGB_XGB_fallback']
    actual=Y[START:];ens=np.zeros_like(actual)
    for i in range(len(actual)):
        if i<7:w=np.ones(len(pool))/len(pool)
        else:
            m=np.array([np.mean(np.abs(preds[x][:i]-actual[:i])) for x in pool]);w=1/(m+1e-8);w/=w.sum()
        ens[i]=sum(a*preds[x][i] for a,x in zip(w,pool))
    preds['causal_ensemble']=ens;times['causal_ensemble']=0
    rows=[]
    for name,x in preds.items():
        e=x-actual;rows.append({'model':name,'MAE':float(np.mean(abs(e))),'RMSE':float(np.sqrt(np.mean(e*e))),'bias':float(e.mean()),'q95_abs_error':float(np.quantile(abs(e),.95)),'runtime_s':times[name]})
    with (OUT/'price_forecast_metrics.csv').open('w',newline='',encoding='utf-8-sig') as f:w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
    np.savez_compressed(OUT/'price_forecasts.npz',actual=actual,**preds)
    # Explicit availability audit: actual package names requested versus reproducible fallbacks.
    audits=[{'method':'XGBoost','status':'failed_import_missing_libomp','fallback':'HistGradientBoosting squared loss'},
            {'method':'LightGBM','status':'failed_import_missing_libomp','fallback':'HistGradientBoosting squared loss'},
            {'method':'Quantile LightGBM','status':'failed_import_missing_libomp','fallback':'HistGradientBoosting quantile loss'},
            {'method':'Prophet','status':'package_not_installed','fallback':'causal additive trend plus Fourier seasonality'},
            {'method':'GARCH','status':'AR-GARCH-style tested','fallback':'AR mean plus EWMA conditional volatility'}]
    with (OUT/'price_model_availability.csv').open('w',newline='',encoding='utf-8-sig') as f:w=csv.DictWriter(f,fieldnames=audits[0]);w.writeheader();w.writerows(audits)
    print(json.dumps(sorted(rows,key=lambda z:z['MAE'])[:8],indent=2))
if __name__=='__main__':main()
