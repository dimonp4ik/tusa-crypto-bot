"""
Monte Carlo stress test for exported backtest trades.

This bootstraps the trade stream and compounds equity by R results. It is not a
time-accurate portfolio simulator; use simulate_bankroll.py for that.
"""

from __future__ import annotations

import argparse
import csv
import random
from statistics import mean


def load_r_values(path: str, *, use_risk_mult: bool) -> list[float]:
    values: list[float] = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            r = float(row.get("net_r") or 0.0)
            if use_risk_mult:
                r *= float(row.get("risk_mult") or 1.0)
            values.append(r)
    return values


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    vals = sorted(values)
    idx = min(len(vals) - 1, max(0, round((pct / 100.0) * (len(vals) - 1))))
    return vals[idx]


def run_once(
    values: list[float],
    *,
    start: float,
    risk: float,
    rng: random.Random,
    mode: str,
) -> tuple[float, float]:
    equity = start
    peak = equity
    max_dd = 0.0
    if mode == "shuffle":
        path = list(values)
        rng.shuffle(path)
    else:
        path = [values[rng.randrange(len(values))] for _ in range(len(values))]
    for r in path:
        equity += equity * risk * r
        peak = max(peak, equity)
        if peak > 0:
            max_dd = min(max_dd, (equity - peak) / peak * 100.0)
        if equity <= 0:
            return equity, -100.0
    return equity, max_dd


def main() -> int:
    parser = argparse.ArgumentParser(description="Monte Carlo stress test for backtest trade CSV")
    parser.add_argument("csv_path")
    parser.add_argument("--start", type=float, default=1000.0)
    parser.add_argument("--risk", type=float, default=0.01)
    parser.add_argument("--runs", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mode", choices=["bootstrap", "shuffle"], default="bootstrap")
    parser.add_argument("--use-risk-mult", action="store_true")
    args = parser.parse_args()

    values = load_r_values(args.csv_path, use_risk_mult=args.use_risk_mult)
    rng = random.Random(args.seed)
    finals = []
    dds = []
    ruined = 0
    for _ in range(max(1, args.runs)):
        final, dd = run_once(values, start=args.start, risk=args.risk, rng=rng, mode=args.mode)
        finals.append(final)
        dds.append(dd)
        ruined += int(final <= 0)

    print(f"trades={len(values)} runs={args.runs} mode={args.mode} start={args.start:.2f} risk={args.risk * 100:.2f}%")
    print(
        "final "
        f"p05={percentile(finals, 5):.2f} "
        f"p25={percentile(finals, 25):.2f} "
        f"avg={mean(finals):.2f} "
        f"p50={percentile(finals, 50):.2f} "
        f"p75={percentile(finals, 75):.2f} "
        f"p95={percentile(finals, 95):.2f}"
    )
    print(
        "maxDD "
        f"p05={percentile(dds, 5):.1f}% "
        f"p25={percentile(dds, 25):.1f}% "
        f"avg={mean(dds):.1f}% "
        f"p50={percentile(dds, 50):.1f}% "
        f"p75={percentile(dds, 75):.1f}% "
        f"p95={percentile(dds, 95):.1f}%"
    )
    print(f"ruin={ruined}/{args.runs} ({ruined / max(1, args.runs) * 100:.1f}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
