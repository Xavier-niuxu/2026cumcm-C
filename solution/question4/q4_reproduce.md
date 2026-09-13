# 问题 4：复现与实现说明

本目录按参考包 `solution/question4_ref/q4.zip` 实现问题 4（波动电价下的 Q4-2 与 Q4-3）。

## 1. 方法概要

| 部分 | 方法 |
| --- | --- |
| 价格分析 | 附件 4 价格 EDA（自相关、月度/星期均值、与负荷/光伏的相关性），并做**严格因果**的价格预测 |
| 价格预测候选 | persistence、前一日、上周同日、7 日滑动平均、指数平滑、ARIMA 曲线、SARIMA 曲线、Fourier、Fourier_AR、Prophet-like、AR-GARCH、随机森林、HistGB（分位数/均值）、因果逆 MAE 集成（参考包另测 LSTM/GRU/TCN） |
| Q4-2 | 沿用问题 2：0:00 用"点预报 + 历史残差分位数"制定全天 144 段计划，日内只做因果储能响应与紧急购电；**计划用预测电价优化、结算用附件 4 真实价格** |
| Q4-3 | 沿用问题 3：0:00 初始承诺，6:00/12:00/18:00 只调整未执行区间（$1.5p\,u-0.5p\,v$），联合"净负荷误差 × 价格误差"场景做滚动随机 MPC（含 CVaR 选项与信息集消融） |

**防价格泄漏**：回测日 $d$ 的 0:00 价格模型只使用 $r<d$ 的完整价格；Q4-3 在时刻 $h$ 只额外使用当日 $[0,h)$ 已实现价格，对剩余时段做水平残差更新；$[h,24{:}00)$ 的真实价格从不进入该时刻的优化，最终费用一律用真实价格重算。

## 2. 与参考包的两处工程差异（均为补齐缺失模块）

参考包的 `q4_models.py` 从 `../q2/forecast_benchmark.py` 与 `../q3/forecast_analysis.py` 导入
`load_data/fourier` 与 `load_all/downscale/historical_shape/HOURS`，这两个模块**不在交付包里**。
本实现用本地等价模块替代，算法不变：

| 参考包导入 | 本地替代 | 说明 |
| --- | --- | --- |
| `forecast_benchmark.load_data/fourier` | `q2_bridge.py` | 用仓库问题 2 的傅里叶模型对负荷/净负荷序列做因果预报（问题 3 已验证二者等价） |
| `forecast_analysis.load_all/downscale/historical_shape/HOURS` | `pv_tools.py` | 直接取自问题 3 参考包的同一文件（仅修正目录定位） |

## 3. 依赖限制（与参考包自述一致）

参考包记录了以下依赖在本机不可用，并给出了可复现替代：

| 期望方法 | 状态 | 替代 |
| --- | --- | --- |
| XGBoost / LightGBM / 分位 LightGBM | 缺 libomp，导入失败 | HistGradientBoosting（均值/分位损失） |
| Prophet | 未安装 | 因果"趋势 + Fourier 季节"（Prophet-like） |
| GARCH | 无 statsmodels | AR 均值 + EWMA 条件波动（AR-GARCH 风格） |
| LSTM / GRU / TCN（PyTorch） | 未安装 torch | **本实现自动跳过**，并在 `output/price_model_availability.csv` 中记录 |

因此正文使用的价格模型为 `HistGB_XGB_fallback`（与参考包一致）。

## 4. 运行顺序

```bash
python solution/question4/price_analysis.py         # 价格 EDA + 因果价格预测 -> output/price_forecasts.npz
python solution/question4/q4_models.py              # Q4-2/Q4-3 实验矩阵 + 写 result4-2/4-3.xlsx
python solution/question4/q42_q2_inheritance.py     # Q2 经验分位框架的动态价格扩展（可能刷新 Q4-2 最优）
python solution/question4/additional_analysis.py    # 价格模型经济筛选、残差相关、反事实重结算
python solution/question4/compile_experiments.py    # 汇总 output/experiments_all.csv
```

> 注：`price_analysis.py` 的随机森林/HistGB 训练会用到 joblib 线程池，在受限沙箱内会因
> 无法创建进程/管道而失败，需要在正常权限下运行。

## 5. 输出

| 文件 | 内容 |
| --- | --- |
| `output/result4-2.xlsx` | Q4-2 最终结果（按附件 5 模板：计划购电量 + 充放电量 + 紧急购电量） |
| `output/result4-3.xlsx` | Q4-3 最终结果（计划购电量 + 调整购电量 + 充放电量 + 紧急购电量） |
| `output/price_forecasts.npz` | 各价格模型对 2025-02-01~12-31 的因果预测 |
| `output/price_forecast_metrics.csv` | 价格预测精度（MAE/RMSE/bias/q95） |
| `output/price_eda.json` | 价格统计特征 |
| `output/economic_experiments.csv` | Q4-2/Q4-3 场景结构与风险参数的经济回测 |
| `output/q42_quantile_scan.csv` | Q4-2 分位数扫描 |
| `output/summary.json` | 两个小问的最优方案与信息价值分解 |

## 6. 参考包记录的目标值（供对照）

| 指标 | Q4-2（Q2 经验分位 q=0.8） | Q4-3（联合块场景滚动 MPC） |
| --- | ---: | ---: |
| 计划购电费 | 14,126,584.34 元 | 13,397,219.09 元 |
| 调整费用 | — | 576,361.80 元 |
| 紧急购电费 | 593,924.37 元 | 403,314.63 元 |
| **全年总费用** | **14,720,508.71 元** | **14,376,895.52 元** |
| 紧急购电量 | 91,876.77 kWh | 72,345.56 kWh |
