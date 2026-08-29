#!/usr/bin/env python3
"""Sweep SIZE_MULT_MAX offline from ONE uncapped export per window.

The ceiling is a pure clamp at the end of the multiplier stack (backtest.py
`return min(m, SIZE_MULT_MAX)`), so it changes no trade and no outcome — only
how big each position is. Export once with SIZE_MULT_MAX set absurdly high and
every candidate ceiling can then be computed exactly:

    unit R      = net_r / size_mult            (size_mult here is UNCAPPED)
    net at cap  = unit R * min(size_mult, cap)

That turns a nine-run sweep into three runs plus arithmetic. Written 2026-08-29
to re-examine a decision that was made against inflated risk figures: the
kill-switch replay had been peeking at future outcomes, so every worst-windows
and ulcer ratio the original sweep compared was three to four times too kind.

    BT_EXPORT_RAW=1 SIZE_MULT_MAX=100 python backtest.py --candles 18000 \
        --end-date 2023-07-31 --quiet --export-trades unc_2023.csv
    python tools_ceiling.py unc_2023.csv unc_2024.csv unc_2026.csv
"""
from __future__ import annotations

import csv
import sys

from tools_gates import apply_gates, risk_profile


def fld(row: dict, key: str, missing: float) -> float:
    v = row.get(key)
    if v is None or v == "":
        return missing
    try:
        return float(v)
    except (TypeError, ValueError):
        return missing


CAPS = (1.5, 1.75, 2.0, 2.5, 3.0, 4.0, 99.0)


def main() -> int:
    paths = sys.argv[1:]
    if not paths:
        print("укажи выгрузки без потолка: tools_ceiling.py unc_2023.csv ...")
        return 1
    for path in paths:
        with open(path, newline="", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        if not rows:
            print(f"АВАРИЯ: {path} пуст")
            return 1
        kept = apply_gates(rows, cooldown_h=3.0, per_scan=3, per_dir=6, kill=5)
        kept.sort(key=lambda r: fld(r, "entry_time", 0.0))
        raw_sm = [fld(r, "size_mult", 1.0) for r in kept]
        if max(raw_sm) <= 2.01:
            print(f"АВАРИЯ: {path} — размер зажат ({max(raw_sm):.2f}), "
                  f"выгрузка сделана БЕЗ SIZE_MULT_MAX=100")
            return 1
        units = [fld(r, "net_r", 0.0) / (sm or 1.0) for r, sm in zip(kept, raw_sm)]
        label = path.replace("unc_", "").replace(".csv", "")
        share = 100.0 * sum(1 for x in raw_sm if x > 2.0) / len(raw_sm)
        print(f"\n--- {label}: {len(kept)}сд, выше потолка 2.0 стоят {share:.1f}% "
              f"(макс {max(raw_sm):.2f}) ---")
        for cap in CAPS:
            rs = [u * min(sm, cap) for u, sm in zip(units, raw_sm)]
            net = sum(rs)
            rp = risk_profile(rs)
            ww, ul = rp["worst_windows"], rp["ulcer"]
            wr_txt = f"{net/ww:>6.1f}" if ww > 0 else "   н-д"
            print(f"  потолок {cap:>5}  net {net:>+8.2f}R  худш {ww:>6.2f} ({wr_txt})"
                  f"  ulcer {ul:>5.2f} ({net/ul:>6.1f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
