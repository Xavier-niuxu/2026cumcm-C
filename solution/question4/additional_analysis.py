#!/usr/bin/env python3
from pathlib import Path
import csv,json,sys
import numpy as np
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];OUT=HERE/'output'
sys.path.insert(0,str(HERE));import q4_models as q

L,P,netall=q.load_data();prices_all,dates=q.load_prices_dates();prices=prices_all[q.START:]
z=np.load(OUT/'price_forecasts.npz');net=np.asarray(netall[q.START:]);net_hat=np.array([q.fourier(netall,d) for d in range(q.START,365)])

# Same deterministic planner/controller, differing only in causal price forecast.
econ_path=OUT/'price_forecast_economic.csv'
if econ_path.exists():
    with econ_path.open(encoding='utf-8-sig') as f:rows=[{k:(v if k=='model' else float(v)) for k,v in r.items()} for r in csv.DictReader(f)]
else:
    rows=[]
    for model in [x for x in z.files if x!='actual']:
        r,_=q.q42_backtest('price_forecast_screen',net,net_hat,prices,z[model],mode='joint_block')
        rows.append({'model':model,'total_cost':r['total_cost'],'plan_cost':r['plan_cost'],'emergency_cost':r['emergency_cost'],'daily_q95':r['daily_q95']})
    with econ_path.open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)

# Residual dependence: separate load, PV and price errors using strictly causal forecasts.
Lhat=np.array([q.fourier(L,d) if d>=7 else L[:max(d,1)].mean(0) for d in range(365)])
_,F=q.load_all();pvhat=np.array([q.pv_curve(P,F,d,0) for d in range(q.START,365)])
erL=(L[q.START:]-Lhat[q.START:]).ravel();erPV=(P[q.START:]-pvhat).ravel();erPrice=(prices-z['HistGB_XGB_fallback']).ravel()
corr=np.corrcoef(np.c_[erL,erPV,erPrice].T)
with (OUT/'residual_correlation.csv').open('w',newline='',encoding='utf-8-sig') as f:
    w=csv.writer(f);w.writerow(['','load_error','PV_error','price_error'])
    for name,row in zip(['load_error','PV_error','price_error'],corr):w.writerow([name,*row])

# Harmonized counterfactual: reuse original fixed-price Q2/Q3 decisions and settle at actual variable prices.
# 交付包不含 q2/q3 的固定价决策数组（final_daily_arrays.npz / final_days.npz），
# 缺失时跳过这两项反事实重结算，只报告本地可复现的部分。
q2cost=q3_finalnet=None
q43=np.load(OUT/'final_q43.npz')
try:
    q2=np.load(ROOT/'q2/output/final_daily_arrays.npz');q3=np.load(ROOT/'q3/output/final_days.npz')
    q2cost=float(np.sum(q2['g']*prices)+np.sum(5*q2['emergency']*prices))
    q3_finalnet=float(np.sum(q3['g0']*prices)+np.sum((1.5*np.maximum(q3['gfinal']-q3['g0'],0)-.5*np.maximum(q3['g0']-q3['gfinal'],0))*prices)+np.sum(5*q3['emergency']*prices))
except FileNotFoundError as exc:
    print('skip fixed-price repricing counterfactual:', exc)
q43_finalnet=float(np.sum(q43['g0']*prices)+np.sum((1.5*np.maximum(q43['gfinal']-q43['g0'],0)-.5*np.maximum(q43['g0']-q43['gfinal'],0))*prices)+np.sum(5*q43['e']*prices))
summary={'price_forecast_best_by_economic_cost':min(rows,key=lambda x:x['total_cost']),'fixed_price_Q2_repriced':q2cost,'fixed_price_Q3_final_net_repriced':q3_finalnet,'dynamic_price_Q43_final_net_repriced':q43_finalnet,'residual_correlation':corr.tolist()}
(OUT/'additional_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(summary,ensure_ascii=False,indent=2))
