#!/usr/bin/env python3
from pathlib import Path
import csv
HERE=Path(__file__).resolve().parent;OUT=HERE/'output'
sources=[('economic',(OUT/'economic_experiments.csv')),('q42_quantile',(OUT/'q42_quantile_scan.csv')),('price_forecast',(OUT/'price_forecast_metrics.csv')),('price_economic',(OUT/'price_forecast_economic.csv')),('availability',(OUT/'price_model_availability.csv'))]
rows=[];fields=['experiment_group']
for group,path in sources:
 with path.open(encoding='utf-8-sig') as f:
  for x in csv.DictReader(f):
   x={'experiment_group':group,**x};rows.append(x)
   for k in x:
    if k not in fields:fields.append(k)
extra=[
 {'experiment_group':'method_catalog','method':'joint historical block bootstrap','status':'implemented/selected Q4-3'},
 {'experiment_group':'method_catalog','method':'multivariate Gaussian low-rank','status':'implemented'},
 {'experiment_group':'method_catalog','method':'Gaussian Copula','status':'implemented'},
 {'experiment_group':'method_catalog','method':'Vine Copula approximation','status':'implemented; full 288D rejected for sample size'},
 {'experiment_group':'method_catalog','method':'conditional regime scenario','status':'implemented'},
 {'experiment_group':'method_catalog','method':'CVaR','status':'9 Q4-2 + 3 Q4-3 settings tested'},
]
for x in extra:
 rows.append(x)
 for k in x:
  if k not in fields:fields.append(k)
with (OUT/'experiments_all.csv').open('w',newline='',encoding='utf-8-sig') as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
print(OUT/'experiments_all.csv')
