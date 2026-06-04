"""
Simulate account growth from an exported fast-backtest CSV.

Examples:
    python simulate_bankroll.py trades.csv --risk 0.01
    python simulate_bankroll.py trades.csv --risk 0.01 --portfolio-guard --use-risk-mult
"""

from __future__ import annotations

import argparse
import csv
import heapq
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import count


BAD_OUTCOMES = {"SL", "EXPIRED"}
LOSS_OUTCOMES = {"SL"}


@dataclass(frozen=True)
class Trade:
    symbol: str
    direction: str
    outcome: str
    entry_time: int
    exit_time: int
    net_r: float
    risk_mult: float = 1.0
    mtf_score: int = 0
    adaptive_pack: str = ""


@dataclass
class SimResult:
    risk: float
    start_equity: float
    final_equity: float
    peak_equity: float
    min_equity: float
    max_dd_pct: float
    accepted: int
    skipped_symbol: int = 0
    skipped_max_open: int = 0
    skipped_direction: int = 0
    skipped_cluster: int = 0
    skipped_daily_stop: int = 0
    skipped_memory: int = 0
    ruined: bool = False

    @property
    def return_pct(self) -> float:
        return (self.final_equity / self.start_equity - 1.0) * 100.0


def parse_risks(value: str) -> list[float]:
    risks = []
    for item in value.split(","):
        item = item.strip()
        if item:
            risk = float(item)
            if risk <= 0:
                raise ValueError("Risk values must be positive")
            risks.append(risk)
    if not risks:
        raise ValueError("At least one risk value is required")
    return risks


def parse_loss_streak_cut(value: str) -> list[tuple[int, float]]:
    if not value.strip():
        return []
    rules: list[tuple[int, float]] = []
    for item in value.split(","):
        if not item.strip():
            continue
        streak_s, mult_s = item.split(":", 1)
        streak = int(streak_s)
        mult = float(mult_s)
        if streak <= 0 or mult <= 0:
            raise ValueError("Loss-streak rules must be like 2:0.5,3:0.25")
        rules.append((streak, mult))
    return sorted(rules)


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
                    direction=(row.get("direction") or "").upper(),
                    outcome=(row.get("outcome") or "").upper(),
                    entry_time=entry_time,
                    exit_time=exit_time,
                    net_r=float(row.get("net_r") or 0.0),
                    risk_mult=float(row.get("risk_mult") or 1.0),
                    mtf_score=int(float(row.get("mtf_score") or 0)),
                    adaptive_pack=row.get("adaptive_pack") or "",
                )
            )
    trades.sort(key=lambda t: (t.entry_time, -t.mtf_score, t.symbol, t.exit_time))
    return trades


def _day_key(ts: int) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d")


def _loss_streak_multiplier(streak: int, rules: list[tuple[int, float]]) -> float:
    mult = 1.0
    for need_streak, rule_mult in rules:
        if streak >= need_streak:
            mult = rule_mult
    return mult


def simulate(
    trades: list[Trade],
    *,
    start_equity: float,
    risk: float,
    same_symbol_lock: bool,
    cooldown_hours: float,
    max_open: int,
    max_open_long: int,
    max_open_short: int,
    max_new_per_window: int,
    cluster_window_minutes: float,
    daily_loss_pct: float,
    use_risk_mult: bool,
    symbol_memory_lookback: int,
    symbol_memory_max_losses: int,
    symbol_memory_cooldown_hours: float,
    loss_streak_rules: list[tuple[int, float]],
) -> SimResult:
    equity = float(start_equity)
    peak = equity
    min_equity = equity
    max_dd_pct = 0.0
    active: list[tuple[int, int, float, Trade]] = []
    seq = count()
    symbol_locked_until: dict[str, int] = {}
    recent_symbol_outcomes: dict[str, deque[tuple[int, str]]] = defaultdict(deque)
    recent_entries: deque[int] = deque()
    cooldown_sec = int(cooldown_hours * 3600)
    memory_cooldown_sec = int(symbol_memory_cooldown_hours * 3600)
    cluster_window_sec = int(cluster_window_minutes * 60)

    current_day = _day_key(trades[0].entry_time) if trades else _day_key(0)
    day_start_equity = equity
    day_realized_pnl = 0.0
    global_loss_streak = 0

    result = SimResult(
        risk=risk,
        start_equity=start_equity,
        final_equity=equity,
        peak_equity=peak,
        min_equity=min_equity,
        max_dd_pct=max_dd_pct,
        accepted=0,
    )

    def roll_day(ts: int) -> None:
        nonlocal current_day, day_start_equity, day_realized_pnl
        key = _day_key(ts)
        if key != current_day:
            current_day = key
            day_start_equity = equity
            day_realized_pnl = 0.0

    def settle_until(ts: int) -> None:
        nonlocal equity, peak, min_equity, max_dd_pct, day_realized_pnl, global_loss_streak
        while active and active[0][0] <= ts:
            exit_time, _, pnl, trade = heapq.heappop(active)
            roll_day(exit_time)
            equity += pnl
            day_realized_pnl += pnl
            peak = max(peak, equity)
            min_equity = min(min_equity, equity)
            if peak > 0:
                max_dd_pct = min(max_dd_pct, (equity - peak) / peak * 100.0)
            if trade.outcome in LOSS_OUTCOMES:
                global_loss_streak += 1
            elif trade.net_r > 0:
                global_loss_streak = 0
            mem = recent_symbol_outcomes[trade.symbol]
            mem.appendleft((exit_time, trade.outcome))
            while len(mem) > max(1, symbol_memory_lookback):
                mem.pop()
            if equity <= 0:
                result.ruined = True

    def active_direction_count(direction: str) -> int:
        return sum(1 for _, _, _, trade in active if trade.direction == direction)

    def symbol_memory_blocks(trade: Trade, ts: int) -> bool:
        if symbol_memory_lookback <= 0 or symbol_memory_max_losses <= 0:
            return False
        rows = list(recent_symbol_outcomes.get(trade.symbol, ()))[:symbol_memory_lookback]
        if len(rows) < symbol_memory_max_losses:
            return False
        streak_rows = []
        for row in rows:
            if row[1] not in BAD_OUTCOMES:
                break
            streak_rows.append(row)
        if len(streak_rows) < symbol_memory_max_losses:
            return False
        last_exit = max(t for t, _ in streak_rows)
        return ts - last_exit < memory_cooldown_sec

    for trade in trades:
        settle_until(trade.entry_time)
        if result.ruined:
            break
        roll_day(trade.entry_time)

        while recent_entries and trade.entry_time - recent_entries[0] > cluster_window_sec:
            recent_entries.popleft()

        if same_symbol_lock and trade.entry_time < symbol_locked_until.get(trade.symbol, 0):
            result.skipped_symbol += 1
            continue
        if max_open > 0 and len(active) >= max_open:
            result.skipped_max_open += 1
            continue
        if max_open_long > 0 and trade.direction == "LONG" and active_direction_count("LONG") >= max_open_long:
            result.skipped_direction += 1
            continue
        if max_open_short > 0 and trade.direction == "SHORT" and active_direction_count("SHORT") >= max_open_short:
            result.skipped_direction += 1
            continue
        if max_new_per_window > 0 and len(recent_entries) >= max_new_per_window:
            result.skipped_cluster += 1
            continue
        if daily_loss_pct > 0 and day_realized_pnl <= -daily_loss_pct * day_start_equity:
            result.skipped_daily_stop += 1
            continue
        if symbol_memory_blocks(trade, trade.entry_time):
            result.skipped_memory += 1
            continue

        dynamic_mult = _loss_streak_multiplier(global_loss_streak, loss_streak_rules)
        adaptive_mult = trade.risk_mult if use_risk_mult else 1.0
        risk_amount = equity * risk * adaptive_mult * dynamic_mult
        pnl = risk_amount * trade.net_r
        heapq.heappush(active, (trade.exit_time, next(seq), pnl, trade))
        if same_symbol_lock:
            symbol_locked_until[trade.symbol] = trade.exit_time + cooldown_sec
        recent_entries.append(trade.entry_time)
        result.accepted += 1

    settle_until(10**18)
    result.final_equity = equity
    result.peak_equity = peak
    result.min_equity = min_equity
    result.max_dd_pct = max_dd_pct
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Bankroll simulator for exported backtest trades")
    parser.add_argument("csv_path")
    parser.add_argument("--start", type=float, default=1000.0, help="Starting equity")
    parser.add_argument("--risk", default="0.005,0.01,0.02", help="Comma-separated risk per accepted trade")
    parser.add_argument("--same-symbol-lock", action="store_true", help="Block a symbol while its previous trade is active")
    parser.add_argument("--cooldown-hours", type=float, default=0.0, help="Cooldown after a symbol trade exits")
    parser.add_argument("--max-open", type=int, default=0, help="Maximum concurrent trades. 0 = unlimited")
    parser.add_argument("--max-open-long", type=int, default=0, help="Maximum concurrent LONG trades. 0 = unlimited")
    parser.add_argument("--max-open-short", type=int, default=0, help="Maximum concurrent SHORT trades. 0 = unlimited")
    parser.add_argument("--max-new-per-window", type=int, default=0, help="Signal clustering cap. 0 = unlimited")
    parser.add_argument("--cluster-window-min", type=float, default=15.0, help="Clustering window in minutes")
    parser.add_argument("--daily-loss-pct", type=float, default=0.0, help="Stop new entries after daily realized loss reaches this percent of day-start equity")
    parser.add_argument("--symbol-memory-lookback", type=int, default=0, help="Recent outcomes checked per symbol. 0 = disabled")
    parser.add_argument("--symbol-memory-max-losses", type=int, default=3, help="Bad outcomes in a row that trigger symbol cooldown")
    parser.add_argument("--symbol-memory-cooldown-hours", type=float, default=24.0, help="Symbol cooldown after bad streak")
    parser.add_argument("--loss-streak-cut", default="", help="Global dynamic risk rules, e.g. 2:0.5,3:0.25")
    parser.add_argument("--use-risk-mult", action="store_true", help="Apply adaptive risk_mult column when present")
    parser.add_argument("--portfolio-guard", action="store_true", help="Shortcut: symbol lock, max 10 open, 7 per side, cluster 3/15m, daily -4%, symbol memory, adaptive risk")
    parser.add_argument("--label", default="", help="Optional label printed before results")
    args = parser.parse_args()

    if args.portfolio_guard:
        args.same_symbol_lock = True
        args.cooldown_hours = max(args.cooldown_hours, 3.0)
        args.max_open = args.max_open or 10
        args.max_open_long = args.max_open_long or 7
        args.max_open_short = args.max_open_short or 7
        args.max_new_per_window = args.max_new_per_window or 3
        args.cluster_window_min = args.cluster_window_min or 15.0
        args.daily_loss_pct = args.daily_loss_pct or 0.04
        args.symbol_memory_lookback = args.symbol_memory_lookback or 4
        args.symbol_memory_max_losses = args.symbol_memory_max_losses or 3
        args.symbol_memory_cooldown_hours = max(args.symbol_memory_cooldown_hours, 24.0)
        args.use_risk_mult = True

    trades = load_trades(args.csv_path)
    label = args.label or args.csv_path
    mode_parts = ["raw"]
    if args.same_symbol_lock:
        mode_parts.append(f"symbol-lock {args.cooldown_hours:g}h")
    if args.max_open > 0:
        mode_parts.append(f"max-open {args.max_open}")
    if args.max_open_long > 0 or args.max_open_short > 0:
        mode_parts.append(f"dir-cap L{args.max_open_long}/S{args.max_open_short}")
    if args.max_new_per_window > 0:
        mode_parts.append(f"cluster {args.max_new_per_window}/{args.cluster_window_min:g}m")
    if args.daily_loss_pct > 0:
        mode_parts.append(f"daily-stop {args.daily_loss_pct * 100:g}%")
    if args.symbol_memory_lookback > 0:
        mode_parts.append("symbol-memory")
    if args.use_risk_mult:
        mode_parts.append("adaptive-risk")
    if args.loss_streak_cut:
        mode_parts.append(f"loss-cut {args.loss_streak_cut}")

    print(f"{label} | {', '.join(mode_parts)} | loaded={len(trades)} start={args.start:.2f}")
    loss_rules = parse_loss_streak_cut(args.loss_streak_cut)
    for risk in parse_risks(args.risk):
        result = simulate(
            trades,
            start_equity=args.start,
            risk=risk,
            same_symbol_lock=args.same_symbol_lock,
            cooldown_hours=args.cooldown_hours,
            max_open=args.max_open,
            max_open_long=args.max_open_long,
            max_open_short=args.max_open_short,
            max_new_per_window=args.max_new_per_window,
            cluster_window_minutes=args.cluster_window_min,
            daily_loss_pct=args.daily_loss_pct,
            use_risk_mult=args.use_risk_mult,
            symbol_memory_lookback=args.symbol_memory_lookback,
            symbol_memory_max_losses=args.symbol_memory_max_losses,
            symbol_memory_cooldown_hours=args.symbol_memory_cooldown_hours,
            loss_streak_rules=loss_rules,
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
            f"skip_open={result.skipped_max_open:<5} "
            f"skip_dir={result.skipped_direction:<5} "
            f"skip_cluster={result.skipped_cluster:<5} "
            f"skip_day={result.skipped_daily_stop:<5} "
            f"skip_memory={result.skipped_memory:<5} "
            f"ruined={result.ruined}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
