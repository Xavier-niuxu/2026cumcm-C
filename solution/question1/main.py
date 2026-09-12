# -*- coding: utf-8 -*-
"""问题1主程序：读取附件1 -> LP求解 -> 输出result1.xlsx。"""

from __future__ import annotations

from pathlib import Path

from LP import solve_from_file
from output import print_summary, save_result1


def main() -> None:
    # 项目目录假定为：
    # project/
    # ├─ 附件/
    # │  └─ 附件1.xlsx
    # └─ solution/
    #    └─ question1/
    #       ├─ data_loader.py
    #       ├─ LP.py
    #       ├─ output.py
    #       └─ main.py
    #
    # 因此 result1.xlsx 自动写入 project/附件5/result1.xlsx。
    project_root = Path(__file__).resolve().parents[2]
    data_file = project_root / "附件" / "附件1.xlsx"
    output_file = project_root / "附件5" / "result1.xlsx"

    if not data_file.exists():
        raise FileNotFoundError(
            f"未找到附件1：{data_file}\n"
            "请将题目附件1.xlsx放到项目根目录的“附件”文件夹中。"
        )

    print(f"读取数据：{data_file}")
    ans = solve_from_file(data_file)

    print_summary(ans)

    # 输出分时段储能信息
    print("\n分时段储能充放电量（六个4小时区间）：")
    print(f"{'时间段':<15}{'充电量/kWh':>16}{'放电量/kWh':>16}")
    for k in range(6):
        lo, hi = k * 24, (k + 1) * 24
        time_label = f"{k * 4}:00-{(k + 1) * 4}:00"
        print(
            f"{time_label:<15}"
            f"{ans.charge[lo:hi].sum():>16.4f}"
            f"{ans.discharge[lo:hi].sum():>16.4f}"
        )
    print(f"{'0:00储电量':<15}{ans.soc_start[0]:>16.4f}{'':>16}")
    print(f"{'24:00储电量':<15}{ans.soc_end[-1]:>16.4f}{'':>16}")

    saved = save_result1(ans, output_file)
    print(f"\n完整计划购电策略已保存：{saved}")


if __name__ == "__main__":
    main()
