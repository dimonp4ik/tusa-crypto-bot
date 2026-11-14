"""Research cash ledger using production contract rounding; no orders or keys."""
import heapq
import math

from src.okx_trader import calc_contracts, calc_partial_close


def simulate(rows, specs, *, capital=125.64, margin_fraction=.03, tp1_pct=5, slots=4):
    if not (math.isfinite(capital) and capital > 0 and 0 < margin_fraction <= .1
            and 0 <= tp1_pct < 100 and slots > 0):
        raise ValueError("invalid simulation settings")
    equity = peak = capital
    drawdown = 0.0
    pending = []
    active = set()
    trades = []
    skipped = dict(slots=0, minimum_size=0)
    sequence = 0

    def realize(until):
        nonlocal equity, peak, drawdown
        while pending and (until is None or pending[0][0] < until):
            _, _, pnl, release = heapq.heappop(pending)
            equity += pnl
            if release is not None:
                active.remove(release)
            peak = max(peak, equity)
            drawdown = max(drawdown, (peak-equity)/peak)

    for row in sorted(rows, key=lambda r: (r["entry_time"], r["symbol"])):
        now = float(row["entry_time"])
        required = [float(row[k]) for k in ("entry", "risk_pct", "sequential_runner_r", "exit_time")]
        if (not all(math.isfinite(v) for v in (now, *required))
                or required[0] <= 0 or required[1] <= 0 or required[3] < now):
            raise ValueError("invalid trade inputs")
        realize(now)
        symbol = row["symbol"]
        if len(active) >= slots or symbol in active:
            skipped["slots"] += 1
            continue
        spec = specs[symbol]
        size = calc_contracts(equity*margin_fraction, 10, row["entry"], spec)
        if size <= 0:
            skipped["minimum_size"] += 1
            continue
        partial = calc_partial_close(size, tp1_pct, spec)
        banked = bool(row["sequential_banked"])
        close_size = partial["size"] if banked else 0.0
        risk_dollars = float(row["entry"])*float(row["risk_pct"])*float(spec["ctVal"])
        runner_pnl = (size-close_size)*risk_dollars*float(row["sequential_runner_r"])
        bank_pnl = close_size*risk_dollars*.5
        end = float(row["exit_time"])
        if banked:
            bank_time = float(row["near_exit_time"])
            if not now <= bank_time <= end:
                raise ValueError("partial exit outside trade lifetime")
            sequence += 1
            heapq.heappush(pending, (bank_time, sequence, bank_pnl, None))
        sequence += 1
        heapq.heappush(pending, (end, sequence, runner_pnl, symbol))
        active.add(symbol)
        trades.append(dict(symbol=symbol, direction=row["direction"], entry_time=now,
            contracts=size, planned_partial=partial["size"],
            actual_partial_pct=partial["actual_pct"], tp1_hit=banked,
            bank_pnl=bank_pnl, runner_pnl=runner_pnl))
    realize(None)
    return dict(starting_capital=capital, ending_capital=round(equity, 6),
        return_pct=round(100*(equity/capital-1), 6), cash_drawdown_pct=round(100*drawdown, 6),
        trades=len(trades), skipped=skipped,
        unplaceable_partials=sum(t["planned_partial"] == 0 for t in trades),
        tp1_hits_without_bank=sum(t["tp1_hit"] and t["planned_partial"] == 0 for t in trades),
        ledger=trades)
