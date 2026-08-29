#!/usr/bin/env python3
"""Re-check every shipped size multiplier on an honestly-gated book.

Written 2026-08-29, right after the kill-switch replay stopped peeking at
future outcomes. Every one of these rules was fitted against profit/worst and
profit/ulcer ratios that the peek had inflated three- to fourfold, so the
levels they were judged against were wrong even where the direction was right.

Two traps this tool exists to avoid, both of which have already cost a night:

  * net_r in the export is ALREADY multiplied by size_mult. Comparing a
    subset's net_r to the book's therefore credits a rule with whatever boost
    it happens to overlap. Everything here is per-trade UNIT R, net_r divided
    back out by size_mult.
  * a missing field is not a zero. Each rule declares the same substitute the
    bot uses in _fld(), so "no accel_ratio recorded" reads as 1.0 (rule does
    not fire) rather than as 0.0.

  python tools_gates.py raw_2026.csv --per-dir 6 --kill 5 --out honest_2026.csv
  python tools_sizing.py honest_2023.csv honest_2024.csv honest_2026.csv
"""
from __future__ import annotations

import csv
import sys

import config as C


def fld(row: dict, key: str, missing: float) -> float:
    """Numeric read that substitutes ONLY when the field is genuinely absent."""
    v = row.get(key)
    if v is None or v == "":
        return missing
    try:
        return float(v)
    except (TypeError, ValueError):
        return missing


def sess(row: dict) -> str:
    return str(row.get("session") or "").upper()


# (name, shipped multiplier, predicate) — predicates mirror backtest.py exactly,
# including the substitute each _fld() call uses for an absent field.
RULES = [
    ("overlap_calm", C.OVERLAP_CALM_SIZE_MULT,
     lambda r: sess(r) == "OVERLAP" and fld(r, "volume_ratio", 99.0) < C.OVERLAP_VOL_MAX),
    ("chop", C.CHOP_SIZE_MULT,
     lambda r: (fld(r, "volume_ratio", 0.0) >= C.CHOP_VOL_MIN
                and fld(r, "eff_ratio", 99.0) < C.CHOP_EFF_MAX
                and fld(r, "vol_atr_pct", 0.0) >= C.CHOP_ATR_MIN)),
    ("open_space", C.OPEN_SPACE_SIZE_MULT,
     lambda r: r.get("room_atr") not in (None, "") and fld(r, "room_atr", -1.0) >= C.OPEN_SPACE_ROOM_MIN),
    ("parabolic", C.PARABOLIC_SIZE_MULT,
     lambda r: fld(r, "accel_ratio", 1.0) >= C.PARABOLIC_ACCEL_MIN),
    ("dead_thin", C.DEAD_THIN_SIZE_MULT,
     lambda r: sess(r) == "DEAD_ZONE" and fld(r, "volume_ratio", 99.0) < C.DEAD_THIN_VOL_MAX),
    # The export column is "extension_atr"; the setup key backtest.py reads is
    # "bos_extension_atr". Using the setup name here made the subset come back
    # empty in all three windows, which read as "the rule never fires" — the
    # same export-vs-setup rename that already bit room_atr / overhead_atr.
    ("extension", C.EXTENSION_SIZE_MULT,
     lambda r: fld(r, "extension_atr", 0.0) > C.EXTENSION_ATR_THRESHOLD),
    ("btc", C.SYMBOL_SIZE_MULT.get("BTCUSDT", 1.0),
     lambda r: str(r.get("symbol") or "").upper() == "BTCUSDT"),
]


def unit_r(row: dict) -> float:
    """Per-trade R at UNIT size — the only comparable quantity across rules."""
    sm = fld(row, "size_mult", 1.0)
    if sm <= 0:
        sm = 1.0
    return fld(row, "net_r", 0.0) / sm


def main() -> int:
    paths = sys.argv[1:]
    if not paths:
        print("укажи честные книги: tools_sizing.py honest_2023.csv ...")
        return 1
    books = []
    for p in paths:
        with open(p, newline="", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        if not rows:
            print(f"АВАРИЯ: {p} пуст")
            return 1
        books.append((p.replace("honest_", "").replace(".csv", ""), rows))

    for name, shipped, pred in RULES:
        print(f"\n=== {name}  (сейчас x{shipped}) ===")
        for label, rows in books:
            sub = [r for r in rows if pred(r)]
            rest = [r for r in rows if not pred(r)]
            if not sub:
                print(f"  {label:<6} подмножество ПУСТО")
                continue
            su = sum(unit_r(r) for r in sub) / len(sub)
            ru = sum(unit_r(r) for r in rest) / max(1, len(rest))
            share = 100.0 * len(sub) / len(rows)
            verdict = "ВЫШЕ книги" if su > ru else "НИЖЕ книги"
            print(f"  {label:<6} {len(sub):>4}сд ({share:>4.1f}%)  "
                  f"единичный R {su:>+6.3f}  прочее {ru:>+6.3f}  "
                  f"разрыв {su - ru:>+6.3f}  {verdict}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
