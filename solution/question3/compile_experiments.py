#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把各组实验 CSV 合并，并附上方法清单（移植自参考解 compile_experiments.py）。"""

from pathlib import Path
import csv

HERE = Path(__file__).resolve().parent
OUT = HERE / "output"
source = OUT / "experiments.csv"
target = OUT / "experiments_all.csv"


def collect_rows():
    """优先读 experiments.csv；若不存在则合并各分组的 experiments_<group>.csv。"""
    files = [source] if source.exists() else sorted(OUT.glob("experiments_*.csv"))
    rows = []
    for path in files:
        if path.name == "experiments_all.csv":
            continue
        with path.open(encoding="utf-8-sig", newline="") as f:
            rows.extend(csv.DictReader(f))
    return rows

extra = [
    {"name": "method_catalog", "strategy": "no_adjustment", "status": "implemented", "role": "S0 baseline"},
    {"name": "method_catalog", "strategy": "deterministic_reoptimization", "status": "implemented", "role": "deterministic MPC"},
    {"name": "method_catalog", "strategy": "scenario_stochastic_MPC", "status": "selected", "role": "main model"},
    {"name": "method_catalog", "strategy": "multi_stage_scenario_tree", "status": "theoretical benchmark", "role": "not selected: exponential tree growth"},
    {"name": "method_catalog", "strategy": "robust_MPC", "status": "implemented", "role": "too conservative"},
    {"name": "method_catalog", "strategy": "chance_constrained_MPC", "status": "implemented approximation", "role": "90% residual quantile"},
    {"name": "method_catalog", "strategy": "distributionally_robust_MPC", "status": "implemented approximation", "role": "mean plus 2 sd ambiguity stress"},
    {"name": "method_catalog", "strategy": "event_triggered_adjustment", "status": "implemented", "role": "thresholds 0/100/500/2000 CNY"},
    {"name": "method_catalog", "strategy": "approximate_dynamic_programming", "status": "theoretical transfer from Q1", "role": "not selected: explicit scenario LP handles adjustment prices directly"},
    {"name": "scenario_catalog", "strategy": "full_day_residual_block", "status": "selected", "role": "preserves temporal dependence"},
    {"name": "scenario_catalog", "strategy": "KMeans_reduction", "status": "selected", "role": "5 representative residual paths"},
    {"name": "scenario_catalog", "strategy": "tail_weighting", "status": "implemented", "role": "lower emergency energy but higher total cost"},
]


def main():
    rows = collect_rows()
    fields = []
    for row in rows + extra:
        for key in row:
            if key not in fields:
                fields.append(key)
    with target.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows + extra)
    print(target, len(rows), "experiment rows")


if __name__ == "__main__":
    main()
