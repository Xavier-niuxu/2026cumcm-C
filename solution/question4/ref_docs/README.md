# Q4交付说明

- `Q4完整研究报告.docx`：Q4-2/Q4-3完整论文式报告，含10幅图、模型、实验与结论。
- `q4_report.md`：可编辑报告源文件。
- `output/result4-2.xlsx`、`output/result4-3.xlsx`：按附件5模板生成的最终结果。
- `interpretation.md`：随机价格信息集、无泄漏原则和结算口径。
- `decision_log.md`：所有采用、未采用、失败和依赖受限方法。
- `output/experiments_all.csv`：预测、联合场景、风险参数、经济回测的统一实验表。
- `output/figures/`：论文所用10幅图。
- `price_analysis.py`：价格探索和严格滚动预测。
- `q4_models.py`：联合场景、Q4-2/Q4-3及CVaR回测。
- `q42_q2_inheritance.py`：Q2经验分位框架的动态价格扩展。
- `additional_analysis.py`：价格模型经济筛选、残差相关和固定/动态价重结算。
- `make_figures.py`：图表生成。

最终Q4-2采用Q2经验80%分位采购与动态价格预测；最终Q4-3采用联合历史块场景滚动随机MPC。所有回测覆盖2025-02-01至12-31，日内未来真实负荷、PV和价格均未进入决策信息集。
