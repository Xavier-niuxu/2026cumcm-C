#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""问题 3：滚动随机 MPC（移植自参考解 q3/rolling_q3.py）

算法链条（与参考解一致）：

1. 负荷预报：``q2_bridge.fourier(L, d)`` —— 只由 d 日前数据拟合的傅里叶日型模型；
   光伏预报：附件 3 在发布时刻 h 的整点预报，用非负线性插值降尺度到 10 分钟。
2. 场景：过去 90 天、同一发布时刻的**整日净负荷误差向量**（实际净负荷 − 预报净负荷），
   用 K-means 压成 5 个带权场景（保留 144 时段内的时序相关性）。
3. 每个发布时刻解一个两阶段随机 LP：对所有场景共享的购电承诺 q（以及调整量 u/v），
   逐场景的储能 c,d,S 与紧急购电 e；目标 = 调整成本 + 期望紧急购电成本。
4. 只执行到下一个发布时刻：实际执行层每 10 分钟观察真实缺额，先由储能响应，
   剩余缺口才紧急购电（因果执行）。
5. 结算：主口径为**逐次取消结算**（调增 1.5p、调减退回原价并承担 0.5p 违约成本），
   另做"计划沉没+罚金"和"仅最终净额"的敏感性。

与参考解代码的唯一改动：目录定位、把缺失的 ``forecast_benchmark`` 换成 ``q2_bridge``、
结果同时写回 附件5/result3.xlsx。
"""

from __future__ import annotations

from pathlib import Path
from time import perf_counter
from shutil import copyfile
import argparse
import csv
import json
import sys

import numpy as np
from scipy.optimize import linprog
from sklearn.cluster import KMeans
from openpyxl import load_workbook

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
ATT = ROOT / "附件"
OUT = HERE / "output"
OUT.mkdir(exist_ok=True)
sys.path.insert(0, str(HERE))

from forecast_analysis import load_all, downscale, historical_shape, HOURS  # noqa: E402
from q2_bridge import load_data, fourier  # noqa: E402

DT = 1 / 6
ETA = .9
LIM = 5000 * DT
EMIN = 1200.
EMAX = 10800.
TARGET = 6000.
START = 31                     # 2025-02-01 起（题目要求回测区间）
N = 144
LAST_DAY = 365                 # 到 12-31


def labels():
    out = []
    for t in range(N):
        a = t * 10
        b = (t + 1) * 10
        f = lambda x: '0:00+1' if x == 1440 else f'{x // 60}:{x % 60:02d}'
        out.append(f'{f(a)}-{f(b)}')
    return out


def price_dates():
    w = load_workbook(ATT / '附件1.xlsx', read_only=True, data_only=True)
    p = np.array([r[1] for r in w.active.iter_rows(min_row=2, values_only=True)], float)
    w = load_workbook(ATT / '附件2.xlsx', read_only=True, data_only=True)
    dates = [r[0] for r in w.worksheets[0].iter_rows(min_row=2, values_only=True)]
    return p, dates


def pv_curve(P, F, d, j, method='linear'):
    """发布时刻 j 对第 d 天剩余时段的光伏曲线（kW，10 分钟粒度）。"""
    issue = HOURS[j]
    nh = 24 - issue
    shape = historical_shape(P, d, issue, nh)
    return np.maximum(downscale(F[d, j, :nh], method, shape), 0)


def histories(L, P, F, Lhat, d, j, method='linear', window=90):
    """过去 window 天、同一发布时刻的整日净负荷误差向量。"""
    issue = HOURS[j]
    arr = []
    for r in range(max(0, d - window), d):
        ph = pv_curve(P, F, r, j, method)
        pred = Lhat[r, issue * 6:] - ph
        act = L[r, issue * 6:] - P[r, issue * 6:]
        arr.append(act - pred)
    return np.asarray(arr)


def reduce_scen(hist, k=5, tail_weight=False):
    """K-means 把历史误差向量压成 k 个代表场景，簇频率作为权重。"""
    if len(hist) == 0:
        return np.zeros((1, 0)), np.ones(1)
    if len(hist) <= k:
        return hist, np.ones(len(hist)) / len(hist)
    km = KMeans(k, random_state=2026, n_init=10).fit(hist)
    w = np.bincount(km.labels_, minlength=k) / len(hist)
    if tail_weight:
        severity = np.maximum(km.cluster_centers_, 0).sum(1)
        w = w * (1 + severity / (severity.mean() + 1e-9))
        w /= w.sum()
    return km.cluster_centers_, w


def optimize_commit(point, p, s0, prev=None, scenarios=None, weights=None,
                    settlement='refund50'):
    """单个发布时刻的两阶段随机 LP：共享购电承诺 + 逐场景储能/紧急购电。"""
    n = len(point)
    if scenarios is None:
        scenarios = np.zeros((1, n))
        weights = np.ones(1)
    K = len(scenarios)
    adjust = prev is not None
    base = n + (2 * n if adjust else 0)
    nv = base + K * 4 * n
    o = np.zeros(nv)
    if adjust:
        if settlement == 'refund50':
            o[n:2 * n] = 1.5 * p
            o[2 * n:3 * n] = -.5 * p
        elif settlement == 'sunk_plus_penalty':
            o[n:2 * n] = 1.5 * p
            o[2 * n:3 * n] = .5 * p
        else:
            raise KeyError(settlement)
    else:
        o[:n] = p
    for k in range(K):
        o[base + k * 4 * n + 3 * n:base + (k + 1) * 4 * n] = weights[k] * 5 * p

    Ae, be, Au, bu = [], [], [], []
    if adjust:
        for t in range(n):
            r = np.zeros(nv)
            r[t] = 1
            r[n + t] = -1
            r[2 * n + t] = 1
            Ae.append(r)
            be.append(prev[t])
    for k in range(K):
        off = base + k * 4 * n
        net = point + scenarios[k]
        for t in range(n):
            r = np.zeros(nv)
            r[off + t] = -ETA
            r[off + n + t] = 1 / ETA
            r[off + 2 * n + t] = 1
            if t:
                r[off + 2 * n + t - 1] = -1
                rhs = 0
            else:
                rhs = s0
            Ae.append(r)
            be.append(rhs)
            r = np.zeros(nv)
            r[t] = -1
            r[off + t] = 1
            r[off + n + t] = -1
            r[off + 3 * n + t] = -1
            Au.append(r)
            bu.append(-net[t] * DT)
        r = np.zeros(nv)
        r[off + 3 * n - 1] = 1
        Ae.append(r)
        be.append(TARGET)

    bds = [(0, None)] * n
    if adjust:
        bds += [(0, None)] * (2 * n)
    for _ in range(K):
        bds += [(0, LIM)] * (2 * n) + [(EMIN, EMAX)] * n + [(0, None)] * n

    z = linprog(o, A_ub=np.array(Au), b_ub=np.array(bu),
                A_eq=np.array(Ae), b_eq=np.array(be), bounds=bds, method='highs')
    if not z.success:
        raise RuntimeError(z.message)
    q = z.x[:n]
    return q, float(z.fun)


def execute_stage(q, actual, start, end, S):
    """因果执行：逐 10 分钟用储能响应真实缺额，剩余缺口紧急购电。"""
    c = np.zeros(end - start)
    d = np.zeros(end - start)
    e = np.zeros(end - start)
    soc = np.empty(end - start + 1)
    soc[0] = S
    for z, t in enumerate(range(start, end)):
        gap = actual[t] * DT - q[t]
        if gap > 0:
            d[z] = min(gap, LIM, (soc[z] - EMIN) * ETA)
        else:
            c[z] = min(-gap, LIM, (EMAX - soc[z]) / ETA)
        soc[z + 1] = soc[z] + ETA * c[z] - d[z] / ETA
        e[z] = max(0, gap + c[z] - d[z])
    return c, d, e, soc


def predicted_keep_cost(prev, point, p, S):
    """因果估计：保持原承诺时的预期紧急购电成本（用于事件触发判断）。"""
    _, _, e, _ = execute_stage(prev, point, 0, len(point), S)
    return float(e @ (5 * p))


def run_day(d, L, P, F, Lhat, p, updates=(0, 6, 12, 18), strategy='stochastic',
            down='linear', settlement='refund50', event_threshold=0, s0=TARGET):
    actual = L[d] - P[d]
    g0 = None
    current = None
    final = np.zeros(N)
    cAll = np.zeros(N)
    dAll = np.zeros(N)
    eAll = np.zeros(N)
    soc = np.empty(N + 1)
    soc[0] = s0
    adj_cost = 0
    adj_kwh = 0
    stage_info = []
    release = [h for h in HOURS if h in updates]
    for z, h in enumerate(release):
        j = HOURS.index(h)
        start = h * 6
        end = release[z + 1] * 6 if z + 1 < len(release) else N
        point = Lhat[d, start:] - pv_curve(P, F, d, j, down)
        hist = histories(L, P, F, Lhat, d, j, down)
        if strategy == 'deterministic':
            sc = np.zeros((1, len(point)))
            w = np.ones(1)
        elif strategy == 'stochastic':
            sc, w = reduce_scen(hist, 5)
        elif strategy == 'tail_stochastic':
            sc, w = reduce_scen(hist, 5, True)
        elif strategy == 'robust':
            sc = np.quantile(hist, .99, axis=0)[None, :] if len(hist) else np.zeros((1, len(point)))
            w = np.ones(1)
        elif strategy == 'chance':
            sc = np.quantile(hist, .90, axis=0)[None, :] if len(hist) else np.zeros((1, len(point)))
            w = np.ones(1)
        elif strategy == 'DRO':
            sc = (hist.mean(0) + 2 * hist.std(0))[None, :] if len(hist) else np.zeros((1, len(point)))
            w = np.ones(1)
        else:
            raise KeyError(strategy)

        prev = None if h == 0 else current[start:]
        q, obj = optimize_commit(point, p[start:], soc[start], prev, sc, w, settlement)
        if h == 0:
            g0 = np.r_[np.zeros(start), q]
            current = g0.copy()
            plan_cost = float(q @ p[start:])
            accepted = True
        else:
            up = np.maximum(q - prev, 0)
            dn = np.maximum(prev - q, 0)
            inc = float(1.5 * up @ p[start:] +
                        (-.5 if settlement == 'refund50' else .5) * dn @ p[start:])
            keep = predicted_keep_cost(prev, point, p[start:], soc[start])
            accepted = (keep - obj) > event_threshold
            if accepted:
                current[start:] = q
                adj_cost += inc
                adj_kwh += float(up.sum() + dn.sum())
        final[start:end] = current[start:end]
        cc, dd, ee, S = execute_stage(current, actual, start, end, soc[start])
        cAll[start:end] = cc
        dAll[start:end] = dd
        eAll[start:end] = ee
        soc[start:end + 1] = S
        stage_info.append({'hour': h, 'accepted': accepted, 'objective': obj})

    emergency_cost = float(eAll @ (5 * p))
    total = plan_cost + adj_cost + emergency_cost
    return {'g0': g0, 'gfinal': final, 'c': cAll, 'd': dAll, 'soc': soc,
            'emergency': eAll, 'plan_cost': plan_cost, 'adjustment_cost': adj_cost,
            'emergency_cost': emergency_cost, 'total_cost': total,
            'adjustment_kwh': adj_kwh, 'events': event_count(eAll), 'stage': stage_info}


def event_count(e):
    return sum(e[i] > 5e-5 and (i == 0 or e[i - 1] <= 5e-5) for i in range(len(e)))


def backtest(name, L, P, F, Lhat, p, updates, strategy, down='linear',
             settlement='refund50', event_threshold=0, continuous=True,
             start_day=START, last_day=LAST_DAY):
    tic = perf_counter()
    days = []
    s = TARGET
    for d in range(start_day, last_day):
        r = run_day(d, L, P, F, Lhat, p, updates, strategy, down, settlement,
                    event_threshold, s if continuous else TARGET)
        days.append(r)
        if continuous:
            s = float(r['soc'][-1])
    out = {
        'name': name, 'updates': str(updates), 'strategy': strategy,
        'downscale': down, 'settlement': settlement, 'continuous': continuous,
        'plan_cost': sum(x['plan_cost'] for x in days),
        'adjustment_cost': sum(x['adjustment_cost'] for x in days),
        'emergency_cost': sum(x['emergency_cost'] for x in days),
        'total_cost': sum(x['total_cost'] for x in days),
        'emergency_kwh': sum(x['emergency'].sum() for x in days),
        'adjustment_kwh': sum(x['adjustment_kwh'] for x in days),
        'events': sum(x['events'] for x in days),
        'throughput': sum(x['c'].sum() + x['d'].sum() for x in days),
        'terminal_soc': float(days[-1]['soc'][-1]),
        'violations': sum(x['soc'].min() < EMIN - 1e-6 or x['soc'].max() > EMAX + 1e-6
                          for x in days),
        'runtime_s': perf_counter() - tic,
    }
    return out, days


def save_result(dates, days):
    """按附件 5 模板写出 result3.xlsx（同时写回 附件5/ 交付位置）。"""
    src = ATT / '附件5/result3.xlsx'
    dst = OUT / 'result3.xlsx'
    wb = load_workbook(src)
    labs = labels()
    for sn, key in [('计划购电量', 'g0'), ('调整购电量', 'gfinal')]:
        ws = wb[sn]
        for t, l in enumerate(labs, 2):
            ws.cell(1, t, l)
        for i, (day, r) in enumerate(zip(dates[START:LAST_DAY], days), 2):
            ws.cell(i, 1, day)
            x = r[key]
            for t in range(N):
                ws.cell(i, t + 2, round(float(x[t]), 4))
            ws.cell(i, 146, round(float(x.sum()), 4))
            ws.cell(i, 147, round(float(r['total_cost']), 4))
    ws = wb['充放电量']
    ws.delete_rows(2, ws.max_row)
    row = 2
    spans = ['0:00-4:00', '4:00-8:00', '8:00-12:00',
             '12:00-16:00', '16:00-20:00', '20:00-24:00']
    for day, r in zip(dates[START:LAST_DAY], days):
        for k, sp in enumerate(spans):
            ws.cell(row, 1, day if k == 0 else None)
            ws.cell(row, 2, sp)
            ws.cell(row, 3, round(float(r['c'][k * 24:(k + 1) * 24].sum()), 4))
            ws.cell(row, 4, round(float(r['d'][k * 24:(k + 1) * 24].sum()), 4))
            if k == 0:
                ws.cell(row, 5, '0:00')
                ws.cell(row, 6, round(float(r['soc'][0]), 4))
            if k == 1:
                ws.cell(row, 5, '24:00')
                ws.cell(row, 6, round(float(r['soc'][-1]), 4))
            row += 1
    ws = wb['紧急购电量']
    ws.delete_rows(2, ws.max_row)
    row = 2
    for day, r in zip(dates[START:LAST_DAY], days):
        e = r['emergency']
        i = 0
        first = True
        while i < N:
            if e[i] <= 5e-5:
                i += 1
                continue
            j = i + 1
            while j < N and e[j] > 5e-5:
                j += 1
            ws.cell(row, 1, day if first else None)
            ws.cell(row, 2, labs[i].split('-')[0] + '-' + labs[j - 1].split('-')[1])
            ws.cell(row, 3, round(float(e[i:j].sum()), 4))
            row += 1
            first = False
            i = j
        if first:
            ws.cell(row, 1, day)
            ws.cell(row, 2, '无')
            ws.cell(row, 3, 0)
            row += 1
    wb.save(dst)
    copyfile(dst, ATT / '附件5/result3.xlsx')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--start-day', type=int, default=START)
    ap.add_argument('--last-day', type=int, default=LAST_DAY)
    ap.add_argument('--smoke', action='store_true',
                    help='只跑一次主策略回测，不做实验矩阵')
    ap.add_argument('--group', default='all',
                    choices=['all', 'main', 'methods', 'event', 'downscale', 'settlement'],
                    help='只跑某一组实验（便于并行），结果写 experiments_<group>.csv')
    ap.add_argument('--no-save', action='store_true', help='不写 result3.xlsx')
    args = ap.parse_args()

    L, P, _ = load_data()
    P2, F = load_all()
    assert np.allclose(P, P2)
    p, dates = price_dates()
    Lhat = np.array([fourier(L, d) if d >= 7 else L[:max(d, 1)].mean(0)
                     for d in range(LAST_DAY)])

    if args.smoke:
        r, days = backtest('smoke', L, P, F, Lhat, p, (0, 6, 12, 18), 'stochastic',
                           start_day=args.start_day, last_day=args.last_day)
        print(json.dumps(r, ensure_ascii=False, indent=2, default=float))
        return

    rows = []
    cache = {}
    sets = [(0,), (0, 6), (0, 6, 12), (0, 6, 12, 18)]
    grp = args.group

    # 1) 信息集消融（含主模型所用配置）
    if grp in ('all', 'main'):
        for u in sets:
            r, d = backtest('update_ablation', L, P, F, Lhat, p, u, 'stochastic')
            rows.append(r)
            cache[('stochastic', u)] = d
            print('ABL', u, r['total_cost'])
    # 2) 方法对比
    if grp in ('all', 'methods'):
        for st in ['deterministic', 'tail_stochastic', 'robust', 'chance', 'DRO']:
            r, d = backtest('method_search', L, P, F, Lhat, p, sets[-1], st)
            rows.append(r)
            cache[(st, sets[-1])] = d
            print(st, r['total_cost'])
    # 3) 事件触发阈值
    if grp in ('all', 'event'):
        for th in [0, 100, 500, 2000]:
            r, d = backtest('event_trigger', L, P, F, Lhat, p, sets[-1], 'stochastic',
                            event_threshold=th)
            r['event_threshold'] = th
            rows.append(r)
            cache[('event', th)] = d
            print('event', th, r['total_cost'])
    # 4) 降尺度方法的经济性比较
    if grp in ('all', 'downscale'):
        for dm in ['step', 'pchip', 'historical_shape', 'linear_mean_preserving']:
            r, d = backtest('downscale_search', L, P, F, Lhat, p, sets[-1], 'stochastic',
                            down=dm)
            rows.append(r)
            print('down', dm, r['total_cost'])
    # 5) 结算口径敏感性
    if grp in ('all', 'settlement'):
        r, d = backtest('settlement_sensitivity', L, P, F, Lhat, p, sets[-1], 'stochastic',
                        settlement='sunk_plus_penalty')
        rows.append(r)
        cache[('settlement', 'strict')] = d
        print('settlement strict', r['total_cost'])

    # 竞赛口径下的"主模型"：在已算出的随机更新集合里取最低实际总费用
    elig = [x for x in rows if x['name'] in ('update_ablation', 'event_trigger')]
    if not elig:
        print('（本组不含 update_ablation，未选出主模型、不写 result3.xlsx）')
        _dump(rows, grp)
        return
    winner = min(elig, key=lambda x: x['total_cost'])
    if winner['name'] == 'update_ablation':
        main_days = cache[('stochastic', eval(winner['updates']))]
    else:
        main_days = cache[('event', winner['event_threshold'])]

    # 仅最终净额口径的敏感性（用主模型决策轨迹重算）
    base = sum(x['plan_cost'] for x in main_days)
    adj = 0
    for x in main_days:
        up = np.maximum(x['gfinal'] - x['g0'], 0)
        dn = np.maximum(x['g0'] - x['gfinal'], 0)
        adj += float(1.5 * up @ p - .5 * dn @ p)
    rr = dict(rows[3])
    rr.update(name='settlement_sensitivity', settlement='final_net_only',
              adjustment_cost=adj, total_cost=base + adj + rr['emergency_cost'])
    rows.append(rr)
    if not args.no_save:
        save_result(dates, main_days)
    np.savez_compressed(
        OUT / 'final_days.npz',
        g0=np.array([x['g0'] for x in main_days]),
        gfinal=np.array([x['gfinal'] for x in main_days]),
        c=np.array([x['c'] for x in main_days]),
        d=np.array([x['d'] for x in main_days]),
        soc=np.array([x['soc'] for x in main_days]),
        emergency=np.array([x['emergency'] for x in main_days]),
        total=np.array([x['total_cost'] for x in main_days]),
    )
    _dump(rows, grp)
    if grp in ('all', 'main'):
        costs = [next(x for x in rows
                      if x['name'] == 'update_ablation' and x['updates'] == str(u))['total_cost']
                 for u in sets]
        summary = {
            'winner': winner,
            'ablation': dict(zip(map(str, sets), costs)),
            'V6': costs[0] - costs[1],
            'V12': costs[1] - costs[2],
            'V18': costs[2] - costs[3],
        }
        (OUT / 'summary.json').write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, default=float),
            encoding='utf-8')
        print(json.dumps(summary, ensure_ascii=False, indent=2, default=float))


def _dump(rows, grp):
    """把本组实验行写成 CSV（all 组写 experiments.csv，其余写 experiments_<group>.csv）。"""
    if not rows:
        return
    fields = []
    for x in rows:
        for k in x:
            if k not in fields:
                fields.append(k)
    name = 'experiments.csv' if grp == 'all' else f'experiments_{grp}.csv'
    with open(OUT / name, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f'-> {name} ({len(rows)} rows)')


if __name__ == '__main__':
    main()
