#!/usr/bin/env python3
"""
Compare two or more exported backtest CSV files.

Example:
  python compare_backtests.py old.csv new.csv --labels old,new
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


WIN_OUTCOMES = {"TP1", "TP2", "TRAIL"}


@dataclass
class Metrics:
    trades: int = 0
    wins: int = 0
    sl: int = 0
    expired: int = 0
    net_r: float = 0.0
    gross_r: float = 0.0

    @property
    def wr(self) -> float:
        return self.wins / self.trades * 100.0 if self.trades else 0.0

    @property
    def rpt(self) -> float:
        return self.net_r / self.trades if self.trades else 0.0


def _float(value, default: float = 0.0) -> float:
    try:
        return float(value or default)
    except (TypeError, ValueError):
        return default


def _month(row: dict[str, str]) -> str:
    return datetime.fromtimestamp(_float(row.get("entry_time")), tz=timezone.utc).strftime("%Y-%m")


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    rows.sort(key=lambda r: (_float(r.get("entry_time")), r.get("symbol", "")))
    return rows


def summarize(rows: list[dict[str, str]]) -> Metrics:
    m = Metrics()
    for row in rows:
        outcome = row.get("outcome", "")
        m.trades += 1
        m.wins += int(outcome in WIN_OUTCOMES)
        m.sl += int(outcome == "SL")
        m.expired += int(outcome == "EXPIRED")
        m.net_r += _float(row.get("net_r"))
        m.gross_r += _float(row.get("gross_r"))
    return m


def fmt_metrics(label: str, m: Metrics, base: Metrics | None = None) -> str:
    delta = ""
    if base is not None:
        delta = (
            f"  dTr={m.trades - base.trades:+d}"
            f" dWR={m.wr - base.wr:+.1f}pp"
            f" dNet={m.net_r - base.net_r:+.2f}R"
            f" dRPT={m.rpt - base.rpt:+.3f}"
        )
    return (
        f"{label:<22} tr={m.trades:<5} wr={m.wr:5.1f}% "
        f"sl={m.sl / m.trades * 100.0 if m.trades else 0:5.1f}% "
        f"exp={m.expired / m.trades * 100.0 if m.trades else 0:5.1f}% "
        f"net={m.net_r:+9.2f}R rpt={m.rpt:+.3f}{delta}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare exported backtest CSV files.")
    parser.add_argument("csv_paths", nargs="+", type=Path)
    parser.add_argument("--labels", default="", help="Comma-separated labels. Defaults to file stems.")
    parser.add_argument("--monthly", action="store_true", help="Also print monthly comparison.")
    parser.add_argument("--outcomes", action="store_true", help="Also print outcome counts.")
    args = parser.parse_args()

    labels = [x.strip() for x in args.labels.split(",") if x.strip()]
    if len(labels) != len(args.csv_paths):
        labels = [p.stem for p in args.csv_paths]

    datasets = [(label, load_rows(path)) for label, path in zip(labels, args.csv_paths)]
    base_metrics = summarize(datasets[0][1]) if datasets else None

    print("TOTAL")
    for label, rows in datasets:
        print(fmt_metrics(label, summarize(rows), None if label == datasets[0][0] else base_metrics))

    if args.monthly:
        months = sorted({m for _, rows in datasets for m in [_month(r) for r in rows]})
        print("\nMONTHLY")
        for month in months:
            print(month)
            month_base = summarize([r for r in datasets[0][1] if _month(r) == month])
            for label, rows in datasets:
                m = summarize([r for r in rows if _month(r) == month])
                print("  " + fmt_metrics(label, m, None if label == datasets[0][0] else month_base))

    if args.outcomes:
        print("\nOUTCOMES")
        for label, rows in datasets:
            counts = Counter(row.get("outcome", "UNKNOWN") for row in rows)
            parts = " ".join(f"{k}={v}" for k, v in sorted(counts.items()))
            print(f"{label:<22} {parts}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
