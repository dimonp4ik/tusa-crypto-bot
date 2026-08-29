"""Sweep SMC_BOS_MIN_VOLUME offline from ONE export taken at the loosest value.

At 1.3 vs 1.4 the parameter's only effect is the setup filter in
signal_filter.py line 757 (`volume_ratio < SMC_BOS_MIN_VOLUME` -> reject). The
volume TIER that also reads it is max(X*1.35, 2.0), which pins at 2.0 for every
value in this range, so the tier is untouched. Filtering an export taken at 1.3
down to volume_ratio >= X therefore reproduces the book at X exactly — verified
against the 1.4 export, whose minimum volume_ratio is 1.400 with nothing below.

Written 2026-08-29 to re-open a rejection made against inflated risk figures.
"""
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


THRESHOLDS = (1.15, 1.20, 1.25, 1.30, 1.35, 1.40, 1.50, 1.60)

for path in sys.argv[1:]:
    rows = list(csv.DictReader(open(path, newline="", encoding="utf-8")))
    if not rows:
        print(f"АВАРИЯ: {path} пуст")
        raise SystemExit(1)
    lo = min(fld(r, "volume_ratio", 9.9) for r in rows)
    print("")
    print(f"--- {path}: сырых {len(rows)}, минимальный volume_ratio {lo:.3f} ---")
    for thr in THRESHOLDS:
        if thr < lo - 1e-9:
            # Below what this export holds: filtering cannot invent trades,
            # so the row would silently repeat the loosest one available.
            continue
        sub = [r for r in rows if fld(r, "volume_ratio") >= thr]
        kept = apply_gates(sub, cooldown_h=3.0, per_scan=3, per_dir=6, kill=5)
        kept.sort(key=lambda r: fld(r, "entry_time"))
        rs = [fld(r, "net_r") for r in kept]
        net = sum(rs)
        rp = risk_profile(rs)
        ww, ul = rp["worst_windows"], rp["ulcer"]
        wins = sum(1 for x in rs if x > 0)
        wr_t = f"{net/ww:>6.1f}" if ww > 0 else "   н-д"
        print(f"  порог {thr:.2f}  {len(kept):>5}сд  WR {100*wins/max(1,len(kept)):>5.1f}%  "
              f"net {net:>+8.2f}R  худш {ww:>6.2f} ({wr_t})  ulcer {ul:>5.2f} ({net/ul:>6.1f})")
