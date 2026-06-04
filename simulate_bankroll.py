"""
Simulate account growth from an exported fast-backtest CSV.

Examples:
    python simulate_bankroll.py trades.csv --risk 0.01
    python simulate_bankroll.py trades.csv --risk 0.005,0.01 --same-symbol-lock --cooldown-hours 3 --max-open 5
"""

from __future__ import annotations

import argparse
import csv
import heapq
from dataclasses import dataclass
from itertools import count


@dataclass(frozen=True)
class Trade:
    symbol: str
    entry_time: int
    exit_time: int
    net_r: float
    risk_mult: float = 1.0


@dataclass
class SimResult:
    risk: float
    final_equity: float
    peak_equity: float
    min_equity: float
    max_dd_pct: float
    accepted: int
    skipped_symbol: int
    skipped_max_open: int
    ruined: bool

    @property
    def return_pct(self) -> float:
        return (self.final_equity / self.start_equity - 1.0) * 100.0  # type: ignore[attr-defined]


def parse_risks(value: str) -> list[float]:
    risks = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        risk = float(item)
        if risk <= 0:
            raise ValueError("Risk values must be positive")
        risks.append(risk)
    if not risks:
        raise ValueError("At least one risk value is required")
    return risks


def load_trades(path: str) -> list[Trade]:
    trades: list[Trade] = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            entry_time = int(float(row.get("entry_time") or 0))
            exit_time = int(float(row.get("exit_time") or entry_time))
            if exit_time <= entry_time:
                exit_time = entry_time + 1
            trades.append(
                Trade(
                    symbol=(row.get("symbol") or "").upper(),
                    entry_time=entry_time,
                    exit_time=exit_time,
                    net_r=float(row.get("net_r") or 0.0),
                    risk_mult=float(row.get("risk_mult") or 1.0),
                )
            )
    trades.sort(key=lambda t: (t.entry_time, t.symbol, t.exit_time))
    return trades


def simulate(
    trades: list[Trade],
    *,
    start_equity: float,
    risk: float,
    same_symbol_lock: bool,
    cooldown_hours: float,
    max_open: int,
    use_risk_mult: bool,
) -> SimResult:
    equity = float(start_equity)
    peak = equity
    min_equity = equity
    max_dd_pct = 0.0
    active: list[tuple[int, int, float]] = []
    seq = count()
    symbol_locked_until: dict[str, int] = {}
    cooldown_sec = int(cooldown_hours * 3600)
    accepted = 0
    skipped_symbol = 0
    skipped_max_open = 0
    ruined = False

    def settle_until(ts: int) -> None:
        nonlocal equity, peak, min_equity, max_dd_pct, ruined
        while active and active[0][0] <= ts:
            _, _, pnl = heapq.heappop(active)
            equity += pnl
            peak = max(peak, equity)
            min_equity = min(min_equity, equity)
            if peak > 0:
                max_dd_pct = min(max_dd_pct, (equity - peak) / peak * 100.0)
            if equity <= 0:
                ruined = True

    for trade in trades:
        settle_until(trade.entry_time)
        if ruined:
            break

        if same_symbol_lock and trade.entry_time < symbol_locked_until.get(trade.symbol, 0):
            skipped_symbol += 1
            continue

        if max_open > 0 and len(active) >= max_open:
            skipped_max_open += 1
            continue

        risk_amount = equity * risk * (trade.risk_mult if use_risk_mult else 1.0)
        pnl = risk_amount * trade.net_r
        heapq.heappush(active, (trade.exit_time, next(seq), pnl))
        if same_symbol_lock:
            symbol_locked_until[trade.symbol] = trade.exit_time + cooldown_sec
        accepted += 1

    settle_until(10**18)

    result = SimResult(
        risk=risk,
        final_equity=equity,
        peak_equity=peak,
        min_equity=min_equity,
        max_dd_pct=max_dd_pct,
        accepted=accepted,
        skipped_symbol=skipped_symbol,
        skipped_max_open=skipped_max_open,
        ruined=ruined,
    )
    result.start_equity = start_equity  # type: ignore[attr-defined]
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Bankroll simulator for exported backtest trades")
    parser.add_argument("csv_path")
    parser.add_argument("--start", type=float, default=1000.0, help="Starting equity")
    parser.add_argument("--risk", default="0.005,0.01,0.02", help="Comma-separated risk per accepted trade")
    parser.add_argument("--same-symbol-lock", action="store_true", help="Block a symbol while its previous trade is active")
    parser.add_argument("--cooldown-hours", type=float, default=0.0, help="Cooldown after a symbol trade exits")
    parser.add_argument("--max-open", type=int, default=0, help="Maximum concurrent trades. 0 = unlimited")
    parser.add_argument("--use-risk-mult", action="store_true", help="Apply adaptive risk_mult column when present")
    parser.add_argument("--label", default="", help="Optional label printed before results")
    args = parser.parse_args()

    trades = load_trades(args.csv_path)
    label = args.label or args.csv_path
    mode = "raw"
    if args.same_symbol_lock:
        mode += f", symbol-lock {args.cooldown_hours:g}h"
    if args.max_open > 0:
        mode += f", max-open {args.max_open}"
    if args.use_risk_mult:
        mode += ", adaptive-risk"

    print(f"{label} | {mode} | loaded={len(trades)} start={args.start:.2f}")
    for risk in parse_risks(args.risk):
        result = simulate(
            trades,
            start_equity=args.start,
            risk=risk,
            same_symbol_lock=args.same_symbol_lock,
            cooldown_hours=args.cooldown_hours,
            max_open=args.max_open,
            use_risk_mult=args.use_risk_mult,
        )
        print(
            f"risk={risk * 100:>4.1f}% "
            f"final={result.final_equity:>10.2f} "
            f"return={result.return_pct:>8.1f}% "
            f"maxDD={result.max_dd_pct:>7.1f}% "
            f"min={result.min_equity:>10.2f} "
            f"peak={result.peak_equity:>10.2f} "
            f"taken={result.accepted:<5} "
            f"skip_symbol={result.skipped_symbol:<5} "
            f"skip_max_open={result.skipped_max_open:<5} "
            f"ruined={result.ruined}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
