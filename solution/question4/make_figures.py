#!/usr/bin/env python3
from pathlib import Path
import sys,json
import numpy as np,pandas as pd
import matplotlib.pyplot as plt
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent;OUT=HERE/'output';FIG=OUT/'figures';FIG.mkdir(exist_ok=True)
sys.path.insert(0,str(HERE));import q4_models as q
plt.rcParams['font.sans-serif']=['Arial Unicode MS','SimHei','DejaVu Sans'];plt.rcParams['axes.unicode_minus']=False
prices,dates=q.load_prices_dates();p=prices[q.START:];t=np.arange(q.N)/6
econ=pd.read_csv(OUT/'economic_experiments.csv');qscan=pd.read_csv(OUT/'q42_quantile_scan.csv');fm=pd.read_csv(OUT/'price_forecast_metrics.csv');fe=pd.read_csv(OUT/'price_forecast_economic.csv');pz=np.load(OUT/'price_forecasts.npz');q42=np.load(OUT/'final_q42.npz');q43=np.load(OUT/'final_q43.npz')

def save(name):plt.tight_layout();plt.savefig(FIG/name,dpi=180,bbox_inches='tight');plt.close()

plt.figure(figsize=(11,5));plt.imshow(prices,aspect='auto',cmap='turbo',vmin=np.quantile(prices,.01),vmax=np.quantile(prices,.99));plt.colorbar(label='元/kWh');plt.xlabel('10 min时段');plt.ylabel('日期序号');plt.title('2025年实时电价热力图');save('01_price_heatmap.png')

daily_mean=prices.mean(1);ixs=[int(np.argmin(daily_mean)),int(np.argsort(daily_mean)[len(daily_mean)//2]),int(np.argmax(daily_mean))]
plt.figure(figsize=(11,5))
for i in ixs:plt.plot(t,prices[i],label=f'{dates[i]:%Y-%m-%d}，均价{daily_mean[i]:.3f}')
plt.xlabel('时刻/h');plt.ylabel('元/kWh');plt.legend();plt.title('低价、典型与高价日实时电价');save('02_representative_prices.png')

i=int(np.argmax(np.mean(np.abs(p-pz['HistGB_XGB_fallback']),1)))
plt.figure(figsize=(11,5));plt.plot(t,p[i],lw=2,label='实际')
for m in ['previous_week_same_time','HistGB_XGB_fallback','causal_ensemble','persistence']:plt.plot(t,pz[m][i],label=m,alpha=.85)
plt.xlabel('时刻/h');plt.ylabel('元/kWh');plt.title(f'价格预测比较：{dates[q.START+i]:%Y-%m-%d}');plt.legend(ncol=2);save('03_price_forecasts.png')

c=pd.read_csv(OUT/'residual_correlation.csv',index_col=0).values;labs=['Load error','PV error','Price error'];plt.figure(figsize=(6,5));im=plt.imshow(c,vmin=-1,vmax=1,cmap='coolwarm');plt.colorbar(im)
plt.xticks(range(3),labs,rotation=20);plt.yticks(range(3),labs)
for a in range(3):
 for b in range(3):plt.text(b,a,f'{c[a,b]:.3f}',ha='center',va='center')
plt.title('预测残差相关矩阵');save('04_residual_correlation.png')

# Representative joint scenarios from the final 90 completed days.
L,P,netall=q.load_data();net_hat=np.array([q.fourier(netall,d) for d in range(q.START,365)]);en=netall[q.START:-1]-net_hat[:-1];ep=p[:-1]-pz['HistGB_XGB_fallback'][:-1];ne,pe,w=q.reduce_joint(en[-90:],ep[-90:],5,'joint_block')
fig,ax=plt.subplots(2,1,figsize=(11,7),sharex=True)
for k in range(len(w)):ax[0].plot(t,net_hat[-1]+ne[k],alpha=.75,label=f's{k+1}({w[k]:.0%})');ax[1].plot(t,np.maximum(pz['HistGB_XGB_fallback'][-1]+pe[k],0),alpha=.75)
ax[0].set_ylabel('净负荷/kW');ax[1].set_ylabel('价格/元·kWh⁻¹');ax[1].set_xlabel('时刻/h');ax[0].legend(ncol=5,fontsize=8);fig.suptitle('联合净负荷—价格代表场景');save('05_joint_scenarios.png')

q2=econ[econ.problem=='Q4-2'];sel=pd.concat([q2[q2.name.isin(['deterministic','robust','DRO_tail'])],q2[(q2.name=='joint_structure')&q2.scenario.isin(['independent','joint_block','multivariate_gaussian'])],q2[(q2.name=='CVaR')&(q2.risk_lambda==.1)&(q2.alpha==.95)]])
names=['Deterministic' if x=='deterministic' else ('CVaR' if x=='CVaR' else (s if x in ['joint_structure'] else x)) for x,s in zip(sel.name,sel.scenario)]
names.append('Q2 q80 inherited');vals=list(sel.total_cost/1e6)+[float(qscan.loc[qscan.total_cost.idxmin(),'total_cost']/1e6)]
plt.figure(figsize=(11,5));plt.bar(names,vals,color=['#4c78a8']*(len(vals)-1)+['#59a14f']);plt.ylabel('全年费用/百万元');plt.xticks(rotation=25);plt.title('Q4-2方法费用比较');save('06_q42_costs.png')

q3=econ[(econ.problem=='Q4-3')&econ.name.isin(['deterministic_dynamic_price','stochastic_dynamic_price','risk_aware_MPC'])];q3=q3[~((q3.name=='stochastic_dynamic_price')&(q3.scenario=='independent'))]
names=['Deterministic' if 'deterministic' in n else ('Joint stochastic' if 'stochastic' in n else f'CVaR λ={l}') for n,l in zip(q3.name,q3.risk_lambda)]
plt.figure(figsize=(10,5));plt.bar(names,q3.total_cost/1e6,color='#59a14f');plt.ylabel('全年费用/百万元');plt.xticks(rotation=20);plt.title('Q4-3方法费用比较');save('07_q43_costs.png')

main42=qscan.loc[[qscan.total_cost.idxmin()]].copy();main42['adjustment_cost']=0
rows=pd.concat([q2[q2.name=='deterministic'],main42,econ[(econ.problem=='Q4-3')&(econ.name=='stochastic_dynamic_price')&(econ.scenario=='joint_block')]],ignore_index=True)
names=['Q4-2确定性','Q4-2主模型','Q4-3主模型'];bottom=np.zeros(3);plt.figure(figsize=(9,5))
for col,label,color in [('plan_cost','计划','#4c78a8'),('adjustment_cost','调整','#f28e2b'),('emergency_cost','紧急','#e15759')]:
 vals=np.nan_to_num(rows[col].values)/1e6 if col in rows else np.zeros(3);plt.bar(names,vals,bottom=bottom,label=label,color=color);bottom+=vals
plt.ylabel('百万元');plt.legend();plt.title('费用构成');save('08_cost_decomposition.png')

imax=int(np.argmax(q43['cost']));fig,ax=plt.subplots(3,1,figsize=(11,8),sharex=True);ax[0].plot(t,p[imax],color='#e15759');ax[0].set_ylabel('价格');ax[1].plot(t,(L[q.START+imax]-P[q.START+imax])/6,label='净需求');ax[1].plot(t,q43['gfinal'][imax],label='最终购电');ax[1].legend();ax[1].set_ylabel('kWh');ax[2].plot(np.arange(145)/6,q43['soc'][imax]);ax[2].set_ylabel('SOC/kWh');ax[2].set_xlabel('时刻/h');fig.suptitle(f'极端费用日案例：{dates[q.START+imax]:%Y-%m-%d}');save('09_extreme_day.png')

add=json.loads((OUT/'additional_summary.json').read_text());vals=[add['fixed_price_Q3_final_net_repriced'],add['dynamic_price_Q43_final_net_repriced']];plt.figure(figsize=(7,5));plt.bar(['固定价Q3策略\n按动态价重结算','动态价Q4-3策略'],np.array(vals)/1e6,color=['#bab0ac','#4c78a8']);plt.ylabel('全年费用/百万元');plt.title('固定价格与动态价格策略比较（统一最终净额口径）');save('10_fixed_vs_variable.png')
print(FIG)
