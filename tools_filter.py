#!/usr/bin/env python3
"""Sweep any PURE setup filter offline from one export taken at its loosest value.

Several of this bot's gates are a single line of the form
`if ind[column] < THRESHOLD: reject`. For those, the parameter does nothing else
— it does not change structure detection, scoring or sizing — so filtering an
export taken at the loosest setting down to `column >= X` reproduces the book at
X EXACTLY. A fifteen-run sweep becomes three runs plus arithmetic.

Two conditions before trusting this on a new parameter, both cheap to check:
  1. grep it. If it appears once, in a reject line, it qualifies. If it also
     feeds a score or a tier, it does not (SMC_BOS_MIN_VOLUME also feeds a tier,
     but that tier is max(X*1.35, 2.0) and pins at 2.0 across its whole range).
  2. anchor it. An export taken at value V must contain nothing below V in that
     column, and re-filtering to V must reproduce the known book to the decimal.

Generalised 2026-08-29 from cmp_bosvol.py, which found SMC_BOS_MIN_VOLUME 1.4
was on the wrong side of a clean peak at 1.30.

    BT_EXPORT_RAW=1 EFF_RATIO_MIN=0.0 python backtest.py --candles 18000 \
        --end-date 2023-07-31 --quiet --export-trades er_2023.csv
    python tools_filter.py eff_ratio 0.0,0.10,0.12,0.15,0.20 er_2023.csv
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


def main() -> int:
    if len(sys.argv) < 4:
        print("tools_filter.py <колонка> <пороги через запятую> <выгрузка...>")
        return 1
    col = sys.argv[1]
    thresholds = [float(x) for x in sys.argv[2].split(",")]
    for path in sys.argv[3:]:
        rows = list(csv.DictReader(open(path, newline="", encoding="utf-8")))
        if not rows:
            print(f"АВАРИЯ: {path} пуст")
            return 1
        if col not in rows[0]:
            print(f"АВАРИЯ: в {path} нет колонки {col}")
            return 1
        filled = [fld(r, col, None) for r in rows]
        filled = [v for v in filled if v is not None]
        if not filled:
            print(f"АВАРИЯ: колонка {col} в {path} пуста")
            return 1
        lo = min(filled)
        print("")
        print(f"--- {path}: сырых {len(rows)}, минимальный {col} {lo:.4f} ---")
        seen = set()
        for thr in thresholds:
            if thr < lo - 1e-9:
                # Below what the export holds: filtering cannot invent trades,
                # so this row would silently repeat the loosest one available.
                continue
            sub = [r for r in rows if fld(r, col, -1e9) >= thr]
            kept = apply_gates(sub, cooldown_h=3.0, per_scan=3, per_dir=6, kill=5)
            kept.sort(key=lambda r: fld(r, "entry_time"))
            rs = [fld(r, "net_r") for r in kept]
            net = sum(rs)
            rp = risk_profile(rs)
            ww, ul = rp["worst_windows"], rp["ulcer"]
            wins = sum(1 for x in rs if x > 0)
            wr_t = f"{net/ww:>6.1f}" if ww > 0 else "   н-д"
            flag = "  ← совпало с предыдущим (ручка не достаёт)" if len(kept) in seen else ""
            seen.add(len(kept))
            print(f"  {thr:>6.3f}  {len(kept):>5}сд  WR {100*wins/max(1,len(kept)):>5.1f}%  "
                  f"net {net:>+8.2f}R  худш {ww:>6.2f} ({wr_t})  ulcer {ul:>5.2f} "
                  f"({net/ul:>6.1f}){flag}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
