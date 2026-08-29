#!/usr/bin/env python3
"""Sweep the two pure-filter thresholds together, reporting the four numbers
the account owner actually asks for: win rate, profit, stop count, drawdown.

Both SMC_BOS_MIN_VOLUME and EFF_RATIO_MIN are single reject lines on columns the
export already carries, so ONE run with both at their loosest reproduces every
combination exactly by filtering. Anchor it: the shipped pair must reproduce the
live book to the decimal.

    BT_EXPORT_RAW=1 SMC_BOS_MIN_VOLUME=1.15 EFF_RATIO_MIN=0.0 \
        python backtest.py --candles 18000 --end-date 2023-07-31 --quiet \
        --export-trades ex23.csv
    python tools_combo.py ex23.csv
"""
from __future__ import annotations

import csv
import sys

from tools_gates import apply_gates, risk_profile


def fld(r, k, m=0.0):
    v = r.get(k)
    if v is None or v == "":
        return m
    try:
        return float(v)
    except (TypeError, ValueError):
        return m


VOL = (1.15, 1.20, 1.30, 1.40, 1.50)
ERT = (0.00, 0.10, 0.12, 0.15, 0.20)


def line(label, rows):
    kept = apply_gates(rows, cooldown_h=3.0, per_scan=3, per_dir=6, kill=5)
    kept.sort(key=lambda r: fld(r, "entry_time"))
    rs = [fld(r, "net_r") for r in kept]
    if not rs:
        print(f"  {label:<22} АВАРИЯ: пусто")
        return
    net = sum(rs)
    n = len(kept)
    stops = sum(1 for r in kept if r.get("outcome") == "SL")
    # win rate in MONEY: a trail exit that closes flat is a loss after costs
    wins = sum(1 for x in rs if x > 0)
    rp = risk_profile(rs)
    print(f"  {label:<22} {n:>5}сд  ВР {100*wins/n:>5.1f}%  {net:>+8.2f}R  "
          f"стопов {stops:>4} ({100*stops/n:>4.1f}%)  просадка {rp['max_dd']:>6.2f}R  "
          f"ulcer {rp['ulcer']:>5.2f}")


for path in sys.argv[1:]:
    rows = list(csv.DictReader(open(path, newline="", encoding="utf-8")))
    if not rows:
        print(f"АВАРИЯ: {path} пуст")
        raise SystemExit(1)
    lo_v = min(fld(r, "volume_ratio", 9.9) for r in rows)
    lo_e = min(fld(r, "eff_ratio", 9.9) for r in rows)
    print("")
    print(f"--- {path}: сырых {len(rows)}, минимум объём {lo_v:.3f} / чистота {lo_e:.3f} ---")
    if lo_v > 1.16 or lo_e > 0.01:
        print("  АВАРИЯ: выгрузка снята не на самых мягких порогах")
        raise SystemExit(1)
    for v in VOL:
        for e in ERT:
            sub = [r for r in rows
                   if fld(r, "volume_ratio") >= v and fld(r, "eff_ratio") >= e]
            tag = f"объём {v:.2f} чист {e:.2f}"
            if (abs(v - 1.30) < 1e-9) and (abs(e - 0.12) < 1e-9):
                tag += " ⬅"
            line(tag, sub)
