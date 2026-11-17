"""Research cash ledger using production contract rounding; no orders or keys."""
import heapq
import math

from src.okx_trader import calc_contracts, calc_partial_close


def simulate(rows, specs, *, capital=125.64, margin_fraction=.03, tp1_pct=5, slots=4,
             bot_slots=None, simultaneous_writers=1, release_delay=0, margin_cap=.9, tie_key=None,
             entry_guard=None, capture_events=False):
    if not (math.isfinite(capital) and capital > 0 and 0 < margin_fraction <= .1
            and 0 <= tp1_pct < 100 and slots > 0):
        raise ValueError("invalid simulation settings")
    if (simultaneous_writers<1 or release_delay<0 or not 0<margin_cap<=.9
            or (bot_slots is not None and any(v<1 for v in bot_slots.values()))):
        raise ValueError("invalid account settings")
    equity = peak = capital
    drawdown = 0.0
    pending = []
    active = {}
    peak_positions=0
    peak_margin=0.
    peak_by_bot={}
    trades = []
    skipped = dict(slots=0, minimum_size=0)
    sequence = 0
    cash_events = []
    guard_rejections = []

    def realize(until):
        nonlocal equity, peak, drawdown
        while pending and (until is None or pending[0][0] < until):
            stamp, _, pnl, release, bot, kind = heapq.heappop(pending)
            equity += pnl
            if capture_events and kind != 'slot_release':
                cash_events.append(dict(time=stamp,bot=bot,kind=kind,pnl=pnl,equity=equity))
            if release is not None:
                del active[release]
            peak = max(peak, equity)
            drawdown = max(drawdown, (peak-equity)/peak)

    for row in sorted(rows, key=lambda r: (r["entry_time"], tie_key(r) if tie_key else r["symbol"])):
        now = float(row["entry_time"])
        required = [float(row[k]) for k in ("entry", "risk_pct", "sequential_runner_r", "exit_time")]
        if (not all(math.isfinite(v) for v in (now, *required))
                or required[0] <= 0 or required[1] <= 0 or required[3] < now):
            raise ValueError("invalid trade inputs")
        realize(now)
        symbol = row["symbol"]
        bot=row.get('bot','crypto')
        if bot_slots is not None and bot not in bot_slots:
            raise ValueError('unknown bot allocation')
        if entry_guard is not None:
            reason = entry_guard(bot=bot,now=now,equity=equity)
            if reason:
                skipped['equity_guard'] = skipped.get('equity_guard',0)+1
                if capture_events:
                    guard_rejections.append(dict(time=now,bot=bot,reason=reason,equity=equity))
                continue
        local_full=bot_slots is not None and sum(v['bot']==bot for v in active.values())>=bot_slots[bot]
        if len(active)+simultaneous_writers > slots or symbol in active or local_full:
            skipped["slots"] += 1
            continue
        if equity<=0 or sum(v['margin'] for v in active.values())+equity*margin_fraction*simultaneous_writers>equity*margin_cap+1e-9:
            skipped['margin']=skipped.get('margin',0)+1
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
            heapq.heappush(pending, (bank_time, sequence, bank_pnl, None, bot, 'tp1'))
        sequence += 1
        heapq.heappush(pending, (end, sequence, runner_pnl, symbol if release_delay==0 else None, bot, 'runner'))
        if release_delay:
            sequence+=1
            heapq.heappush(pending,(end+release_delay,sequence,0.,symbol,bot,'slot_release'))
        actual_margin=size*float(spec['ctVal'])*float(row['entry'])/10
        active[symbol]=dict(bot=bot,margin=actual_margin)
        peak_positions=max(peak_positions,len(active))
        peak_margin=max(peak_margin,sum(v['margin'] for v in active.values())/equity)
        peak_by_bot[bot]=max(peak_by_bot.get(bot,0),sum(v['bot']==bot for v in active.values()))
        trades.append(dict(symbol=symbol, direction=row["direction"], entry_time=now,
            contracts=size, planned_partial=partial["size"],
            actual_partial_pct=partial["actual_pct"], tp1_hit=banked,
            bank_pnl=bank_pnl, runner_pnl=runner_pnl))
        if bot_slots is not None:
            trades[-1]['bot']=bot
        if capture_events:
            trades[-1].update(exit_time=end,near_exit_time=float(row['near_exit_time']) if banked else None)
    realize(None)
    result=dict(starting_capital=capital, ending_capital=round(equity, 6),
        return_pct=round(100*(equity/capital-1), 6), cash_drawdown_pct=round(100*drawdown, 6),
        trades=len(trades), skipped=skipped,
        unplaceable_partials=sum(t["planned_partial"] == 0 for t in trades),
        tp1_hits_without_bank=sum(t["tp1_hit"] and t["planned_partial"] == 0 for t in trades),
        ledger=trades)
    if bot_slots is not None:
        cash={bot:sum(t['bank_pnl']+t['runner_pnl'] for t in trades if t['bot']==bot) for bot in bot_slots}
        result.update(peak_positions=peak_positions,peak_by_bot=peak_by_bot,
            peak_reserved_margin_pct=100*peak_margin,pnl_by_bot=cash,
            return_contribution_pct={bot:100*pnl/capital for bot,pnl in cash.items()})
    if capture_events:
        result.update(cash_events=cash_events,guard_rejections=guard_rejections)
    return result
