#!/usr/bin/env python3
from pathlib import Path
import sys,csv,json
import numpy as np
from scipy.optimize import linprog
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];OUT=HERE/'output';sys.path.insert(0,str(HERE));import q4_models as q

def plan_lp(net,p,s0):
 n=q.N;nv=4*n;o=np.r_[p,np.full(2*n,1e-9),np.zeros(n)];Ae=[];be=[];Au=[];bu=[]
 for t in range(n):
  r=np.zeros(nv);r[n+t]=-q.ETA;r[2*n+t]=1/q.ETA;r[3*n+t]=1
  if t:r[3*n+t-1]=-1;rhs=0
  else:rhs=s0
  Ae.append(r);be.append(rhs);r=np.zeros(nv);r[t]=-1;r[n+t]=1;r[2*n+t]=-1;Au.append(r);bu.append(-net[t]*q.DT)
 r=np.zeros(nv);r[4*n-1]=1;Ae.append(r);be.append(q.TARGET)
 z=linprog(o,A_ub=Au,b_ub=bu,A_eq=Ae,b_eq=be,bounds=[(0,None)]*n+[(0,q.LIM)]*(2*n)+[(q.EMIN,q.EMAX)]*n,method='highs')
 if not z.success:raise RuntimeError(z.message)
 return z.x[:n]

def run(level,net,hat,prices,phat):
 days=[];s0=q.TARGET
 for i in range(len(net)):
  hist=net[max(0,i-90):i]-hat[max(0,i-90):i]
  uplift=np.quantile(hist,level,axis=0) if len(hist) else np.zeros(q.N)
  g=plan_lp(hat[i]+uplift,phat[i],s0);c,d,S,e=q.execute(g,net[i],s0);s0=S[-1]
  days.append(dict(g=g,c=c,d=d,soc=S,e=e,plan_cost=float(g@prices[i]),emergency_cost=float(e@(5*prices[i]))))
 for x in days:x['total_cost']=x['plan_cost']+x['emergency_cost']
 r={'problem':'Q4-2','name':'Q2_empirical_quantile','quantile':level,'price_model':'HistGB_XGB_fallback','plan_cost':sum(x['plan_cost'] for x in days),'emergency_cost':sum(x['emergency_cost'] for x in days),'total_cost':sum(x['total_cost'] for x in days),'emergency_kwh':sum(x['e'].sum() for x in days),'events':sum(q.event_count(x['e']) for x in days),'throughput':sum(x['c'].sum()+x['d'].sum() for x in days),'terminal_soc':s0,'violations':0,'daily_q95':float(np.quantile([x['total_cost'] for x in days],.95)),'daily_max':max(x['total_cost'] for x in days)}
 return r,days

def main():
 L,P,netall=q.load_data();prices_all,dates=q.load_prices_dates();z=np.load(OUT/'price_forecasts.npz');hat=np.array([q.fourier(netall,d) for d in range(q.START,365)]);net=netall[q.START:];prices=prices_all[q.START:];rows=[];cache={}
 for lev in [.7,.75,.8,.85,.9,.95]:
  r,d=run(lev,net,hat,prices,z['HistGB_XGB_fallback']);rows.append(r);cache[lev]=d;print(lev,r['total_cost'])
 win=min(rows,key=lambda x:x['total_cost']);q.save_result42(dates,cache[win['quantile']])
 with (OUT/'q42_quantile_scan.csv').open('w',newline='',encoding='utf-8-sig') as f:w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
 s=json.loads((OUT/'summary.json').read_text());s['q42_q2_inherited_winner']=win
 if win['total_cost']<s['q42_winner']['total_cost']:s['q42_winner']=win;s['q42_winner_reason']='Q2 empirical-quantile framework has lowest realized cost after adding causal price forecast'
 (OUT/'summary.json').write_text(json.dumps(s,ensure_ascii=False,indent=2,default=float),encoding='utf-8');np.savez_compressed(OUT/'final_q42.npz',**{k:np.array([x[k] for x in cache[win['quantile']]]) for k in ['g','c','d','soc','e']},cost=np.array([x['total_cost'] for x in cache[win['quantile']]]));print(json.dumps(win,ensure_ascii=False,indent=2,default=float))
if __name__=='__main__':main()
