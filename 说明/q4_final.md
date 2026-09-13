# 问题 4：实时波动电价下的购电策略（Q4-2 / Q4-3）

代码：`solution/question4/`（主引擎 `q4_models.py`；价格分析 `price_analysis.py`；
Q4-2 分位框架 `q42_q2_inheritance.py`）　结果：`solution/question4/output/result4-{2,3}.xlsx`

---

## 0. 主结果

| 指标 | Q4-2（沿用问题 2 框架） | Q4-3（沿用问题 3 框架） |
| --- | ---: | ---: |
| 计划购电费 | 14,128,327.70 元 | 13,393,518.67 元 |
| 调整费用（1.5p·u − 0.5p·v） | — | 588,619.51 元 |
| 紧急购电费 | 605,428.46 元 | 405,886.43 元 |
| **全年总费用** | **14,733,756.16 元** | **14,388,024.61 元** |
| 紧急购电量 | 93,421.23 kWh | 73,703.80 kWh |
| 调整电量 | — | 1,546,375.27 kWh |
| 年末 SOC | 7,009.19 kWh | 6,915.57 kWh |

参考包记录值：Q4-2 14,720,508.71 元、Q4-3 14,376,895.52 元；本实现相差
**+0.090%** 与 **+0.077%**（差异来自下面第 5 节说明的负荷预报模块替换）。

---

## 1. 问题与数据

问题 4 是问题 2、问题 3 在**实时波动电价**下的重算：新增状态是附件 4 给出的 365×144
真实交易时刻电价；购电与紧急购电分别按 $p_{d,t}$ 与 $5p_{d,t}$ 结算；Q4-3 沿用问题 3
的逐次调整口径（调增 $1.5p$、调减 $-0.5p$）。

附件 4 **没有**官方日前电价预报，因此本实现从历史价格构造**严格因果**的预测，
而不是假定 0:00 能看到当天真实价格。

## 2. 价格分析与预测

`price_analysis.py` 先做价格 EDA（自相关、月度/星期均值、与负荷/光伏的相关性，
写入 `output/price_eda.json`），再逐日滚动预测 2025-02-01 ~ 12-31 的价格曲线，
候选模型包括：persistence、前一日、上周同日、7 日滑动平均、指数平滑、ARIMA 曲线、
SARIMA 曲线、Fourier、Fourier_AR、Prophet-like、AR-GARCH、随机森林、
HistGradientBoosting（均值/0.8 分位）、因果逆 MAE 集成。

预测精度（MAE，元/kWh）：**HistGB 0.0432** < 因果集成 0.0443 < SARIMA 0.0457 <
上周同日 0.0472 < 随机森林 0.0480。正文采用 `HistGB_XGB_fallback`。

有意思的是**按经济性筛选**时最优的是最简单的 `persistence`（全年 15,154,576.61 元）——
说明"价格预测精度更高"不等价于"购电成本更低"，因此正文同时报告
`output/price_forecast_economic.csv` 的经济性对比。

## 3. Q4-2：沿用问题 2 的经验分位框架

每天 0:00 只决策一次，日内只做因果储能响应与紧急购电：

1. 用**因果价格预测** $\hat p_{d,t}$ 作为 LP 目标里的电价；
2. 对净负荷点预报加上过去 90 天的**残差分位数** $q$（修正 5 倍紧急电价带来的不对称：
   缺电成本 $5p$、超购成本 $p$，理论临界分位 $5/6$）；
3. 解确定性日前 LP 得到 144 段计划 $g$，执行时储能按真实缺额因果响应；
4. **结算用附件 4 的真实价格**（计划、紧急分别按 $p$、$5p$）。

分位数扫描（全年总费用）：

| 分位 q | 0.70 | 0.75 | **0.80** | 0.85 | 0.90 | 0.95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 总费用/元 | 14,903,750 | 14,777,436 | **14,733,756** | 14,799,339 | 15,005,249 | 15,467,813 |

$q=0.8$ 最优，与参考包一致。

## 4. Q4-3：沿用问题 3 的滚动随机 MPC

0:00 产生初始承诺，6:00/12:00/18:00 只调整尚未执行的区间。与问题 3 的差别：

* **场景是"净负荷误差 × 价格误差"的联合块**：把过去 90 天同发布时刻的净负荷误差向量
  与价格误差向量标准化后拼接，用确定性 NumPy K-means 压成 5 条联合代表路径
  （另有 independent / multivariate_gaussian / gaussian_copula / vine_copula /
  conditional_regime 等构造作为对照）；
* LP 目标里的电价是**该场景的价格路径**（预测 + 价格误差场景），紧急购电按 $5p$ 计；
* 可选 CVaR（Rockafellar–Uryasev 线性化，$\lambda$ 与 $\alpha$ 网格搜索）；
* 价格信息在日内按因果方式更新：时刻 $h$ 只用 $[0,h)$ 已实现价格对剩余曲线做
  水平残差修正（`price_update`）。

### 信息集消融（是否引入日内更新）

| PV 更新 | 价格更新 | 全年总费用 | 紧急购电费 |
| --- | --- | ---: | ---: |
| 否 | 否 | 14,488,502.12 元 | 543,136.11 元 |
| **是** | 否 | 14,387,152.63 元 | 405,945.82 元 |
| 否 | 是 | 14,488,671.53 元 | 542,746.17 元 |
| 是 | 是（主模型） | 14,388,024.61 元 | 405,886.43 元 |

结论：**日内更新的价值几乎全部来自光伏（价格）预报**——只更新 PV 就把费用降了 10.13 万元，
而单独更新价格几乎没有收益（−0.17 万元）；两者同时更新与只更新 PV 基本相同。
原因是价格的可预测性弱（MAE 0.043 元/kWh 相对于价格波动幅度很小），
而 PV 预报误差直接决定净负荷缺口。

## 5. 与参考包的两处工程差异（算法不变）

参考包的 `q4_models.py` 从 `../q2/forecast_benchmark.py` 与 `../q3/forecast_analysis.py`
导入 `load_data/fourier` 与 `load_all/downscale/historical_shape/HOURS`，
**这两个模块不在交付包里**。本实现用本地等价模块替代：

| 参考包导入 | 本地替代 | 说明 |
| --- | --- | --- |
| `forecast_benchmark.load_data/fourier` | `q2_bridge.py` | 用仓库问题 2 的傅里叶模型对负荷/净负荷做因果预报（问题 3 已验证等价） |
| `forecast_analysis.*` | `pv_tools.py` | 直接取用问题 3 参考包的同一文件，仅修正目录定位 |

此外，参考包提到但本机不可用的依赖（torch / xgboost / lightgbm / prophet / statsmodels）
已按其自述方式处理：用 HistGB、Prophet-like、AR-GARCH 风格实现替代，
`price_analysis.py` 在缺 torch 时自动跳过 LSTM/GRU/TCN 并记录到
`output/price_model_availability.csv`。

差异来源与问题 3 相同：负荷预报模块的实现细节不同，导致 0.08%~0.09% 的全年费用漂移；
Q4-3 的联合场景、结算口径、执行层均与参考包逐项一致。

## 6. 复核与时间一致性审计

```bash
python solution/question4/verify_results_q4.py        # 结果复核 25 项
python solution/question4/timeline_audit_q4.py        # 时间一致性（价格不泄漏）审计 6 项
```

* **结果复核 25/25 通过**：两个结果文件的表结构、144 列之和、SOC 区间与日末递推、
  功率上限；并用求解器重跑两个小问的最优方案，计划量、调整量、紧急量、总费用与表内一致
  （Q4-2 14,733,756.16 元、Q4-3 14,388,024.61 元）。
* **时间一致性审计 6/6 通过**：污染未来日期的价格/预报/负荷/光伏 → 当天决策与执行逐位不变；
  **污染当天 6:00/12:00/18:00 之后的真实价格 → 该时刻之前的承诺与执行逐位不变**
  （价格不泄漏的核心检查）；污染更晚发布时刻的 PV 预报 → 0:00 计划不变；
  污染后续日的真实价格 → 首日 0:00 计划不变（只影响结算）。

## 7. 复现命令与依赖

```bash
python solution/question4/price_analysis.py      # 价格 EDA + 因果价格预测（需 joblib 线程池）
python solution/question4/q4_models.py           # Q4-2/Q4-3 实验矩阵 + 写 result4-2/4-3.xlsx
python solution/question4/q42_q2_inheritance.py  # Q2 经验分位框架的动态价格扩展（刷新 Q4-2 最优）
python solution/question4/additional_analysis.py # 价格模型经济筛选、残差相关
python solution/question4/compile_experiments.py # 汇总 output/experiments_all.csv
```

依赖：`solution/question4/requirements.txt`（numpy / scipy / scikit-learn / pandas / openpyxl）；
本机版本 numpy 2.5.3、scipy 1.18.1、scikit-learn 1.9.1。

## 8. 文件清单

| 文件 | 作用 |
| --- | --- |
| `price_analysis.py` | 价格 EDA 与 14 类因果价格预测（含依赖可用性审计） |
| `q4_models.py` | 联合场景构造、Q4-2 随机版本、Q4-3 滚动 MPC、CVaR、信息集消融、出表 |
| `q42_q2_inheritance.py` | Q4-2 的经验分位框架（分位扫描 + 出表） |
| `additional_analysis.py` | 价格模型经济性筛选、负荷/PV/价格残差相关 |
| `compile_experiments.py` | 汇总全部实验到 `output/experiments_all.csv` |
| `q2_bridge.py` / `pv_tools.py` | 补齐参考包缺失的负荷预报与 PV 降尺度模块 |
| `verify_results_q4.py` / `timeline_audit_q4.py` | 结果复核与时间一致性（价格不泄漏）审计 |
| `ref_docs/` | 参考包自带的说明文档（README、interpretation、decision_log、q4_report） |
| `output/` | 结果表、价格预测、实验表、summary.json |
