"""Versioned research cash ledger for explicit client TP1 volume policies."""
import heapq
import math
from src.okx_trader import calc_contracts
from src.tp1_plan import plan_tp1_close


def simulate(rows,specs,*,capital=130.,margin_fraction=.03,tp1_pct=5.,minimum_close=False,
        slots=12,bot_slots=None,simultaneous_writers=2,release_delay=900,margin_cap=.9,
        tie_key=None,capture_events=True):
    if bot_slots is None:bot_slots=dict(crypto=4,stocks=8)
    if not (math.isfinite(capital) and capital>0 and 0<margin_fraction<=.1 and 0<=tp1_pct<=100
            and slots>0 and simultaneous_writers>=1 and release_delay>=0 and 0<margin_cap<=.9):
        raise ValueError('invalid_cash_policy_settings')
    equity=peak=capital;drawdown=0.;pending=[];active={};sequence=0
    ledger=[];events=[];peak_positions=0;peak_margin=0.;peak_by_bot={}
    skipped=dict(slots=0,minimum_size=0)

    def realize(until):
        nonlocal equity,peak,drawdown
        while pending and (until is None or pending[0][0]<until):
            stamp,_,pnl,release,bot,kind=heapq.heappop(pending)
            equity+=pnl
            if capture_events and kind!='slot_release':
                events.append(dict(time=stamp,bot=bot,kind=kind,pnl=pnl,equity=equity))
            if release is not None:del active[release]
            peak=max(peak,equity);drawdown=max(drawdown,(peak-equity)/peak)

    for row in sorted(rows,key=lambda r:(r['entry_time'],tie_key(r) if tie_key else r['symbol'])):
        now=float(row['entry_time']);entry=float(row['entry']);risk=float(row['risk_pct'])
        runner=float(row['sequential_runner_r']);model_end=float(row['exit_time'])
        if not all(math.isfinite(x) for x in (now,entry,risk,runner,model_end)) or min(entry,risk)<=0 or model_end<now:
            raise ValueError('invalid_cash_trade_inputs')
        realize(now);symbol=row['symbol'];bot=row.get('bot','crypto')
        if bot not in bot_slots:raise ValueError('unknown_bot')
        if (len(active)+simultaneous_writers>slots or symbol in active
                or sum(v['bot']==bot for v in active.values())>=bot_slots[bot]):
            skipped['slots']+=1;continue
        if equity<=0 or sum(v['margin'] for v in active.values())+equity*margin_fraction*simultaneous_writers>equity*margin_cap+1e-9:
            skipped['margin']=skipped.get('margin',0)+1;continue
        spec=specs[symbol];size=calc_contracts(equity*margin_fraction,10,entry,spec)
        if size<=0:skipped['minimum_size']+=1;continue
        partial=plan_tp1_close(size,tp1_pct,spec,ensure_minimum=minimum_close)
        banked=bool(row['sequential_banked']);close_size=partial['size'] if banked else 0.
        risk_dollars=entry*risk*float(spec['ctVal'])
        bank_pnl=close_size*risk_dollars*.5;runner_pnl=(size-close_size)*risk_dollars*runner
        end=model_end;full_tp1=False
        if banked:
            bank_time=float(row['near_exit_time'])
            if not math.isfinite(bank_time) or not now<=bank_time<=model_end:
                raise ValueError('partial_exit_outside_trade_lifetime')
            full_tp1=close_size>=size
            if full_tp1:end=bank_time
            sequence+=1;heapq.heappush(pending,(bank_time,sequence,bank_pnl,None,bot,'tp1'))
        sequence+=1
        heapq.heappush(pending,(end,sequence,runner_pnl,symbol if release_delay==0 else None,bot,'runner'))
        if release_delay:
            sequence+=1;heapq.heappush(pending,(end+release_delay,sequence,0.,symbol,bot,'slot_release'))
        actual_margin=size*float(spec['ctVal'])*entry/10
        active[symbol]=dict(bot=bot,margin=actual_margin)
        peak_positions=max(peak_positions,len(active))
        peak_margin=max(peak_margin,sum(v['margin'] for v in active.values())/equity)
        peak_by_bot[bot]=max(peak_by_bot.get(bot,0),sum(v['bot']==bot for v in active.values()))
        ledger.append(dict(symbol=symbol,direction=row['direction'],bot=bot,entry_time=now,
            contracts=size,planned_partial=partial['size'],actual_partial_pct=partial['actual_pct'],
            tp1_hit=banked,bank_pnl=bank_pnl,runner_pnl=runner_pnl,exit_time=end,
            model_exit_time=model_end,full_tp1_close=full_tp1,
            near_exit_time=float(row['near_exit_time']) if banked else None))
    realize(None)
    pnl_by_bot={bot:sum(r['bank_pnl']+r['runner_pnl'] for r in ledger if r['bot']==bot) for bot in bot_slots}
    result=dict(starting_capital=capital,ending_capital=round(equity,6),return_pct=round(100*(equity/capital-1),6),
        cash_drawdown_pct=round(100*drawdown,6),trades=len(ledger),skipped=skipped,
        unplaceable_partials=sum(r['planned_partial']==0 for r in ledger),
        tp1_hits_without_bank=sum(r['tp1_hit'] and r['planned_partial']==0 for r in ledger),
        full_tp1_closes=sum(r['full_tp1_close'] for r in ledger),ledger=ledger,
        peak_positions=peak_positions,peak_by_bot=peak_by_bot,peak_reserved_margin_pct=100*peak_margin,
        pnl_by_bot=pnl_by_bot,return_contribution_pct={b:100*p/capital for b,p in pnl_by_bot.items()})
    if capture_events:result.update(cash_events=events,guard_rejections=[])
    return result
