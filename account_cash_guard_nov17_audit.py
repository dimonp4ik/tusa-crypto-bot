"""Continuous shared cash and observed-entry risk gate, without private fills."""
import hashlib
import json
import math
import pickle
import ast
import inspect
import subprocess
import collections
import heapq
from pathlib import Path

import pandas as pd

from account_cash_guard_nov17 import RealizedCashGuard,calendar_months,stamp
from lot_aware_tp1_audit import simulate
from crypto_runner_exit_path_nov16 import replay
from shared_factor_account_nov14_audit import stock_ready_rows
from src.risk_limits import equity_guard

ROOT=Path('reports/audit_2026_09_08')
STOCK=Path('../tusa stocks/reports/audit_2026_09_08')


def runtime_gate(rows: list[dict], *, loss_pause: int = 3,
                 hourly_cap: int = 2, latch_loss_pause: bool = True) -> list[dict]:
    """Replay current persistent exposure, cooldown and ordering rails."""
    pending: list[tuple[float, int, dict]] = []
    opened: dict[str, dict] = {}
    last: dict[tuple[str, str], float] = {}
    accepted_times: collections.deque[float] = collections.deque()
    output: list[dict] = []
    day = None
    streak = 0
    paused = False
    ordered = sorted(rows, key=lambda row: (
        float(row["entry_time"]), int(row["source_priority"]),
        str(row["symbol"]), int(row.get("module_priority", 999)),
    ))
    for row in ordered:
        now = float(row["entry_time"])
        new_day = int(now // 86400)
        if new_day != day:
            day, streak, paused = new_day, 0, False
        while pending and pending[0][0] < now:
            _, _, done = heapq.heappop(pending)
            opened.pop(str(done["symbol"]), None)
            if int(float(done["exit_time"]) // 86400) == day:
                streak = streak + 1 if done["outcome"] == "SL" else 0
                paused = loss_pause > 0 and (
                    streak >= loss_pause or (latch_loss_pause and paused))
        while accepted_times and accepted_times[0] <= now - 3600:
            accepted_times.popleft()
        symbol = str(row["symbol"])
        direction = str(row["direction"])
        if paused or symbol in opened or len(accepted_times) >= hourly_cap:
            continue
        if now - last.get((symbol, direction), -1e20) < 10800:
            continue
        if sum(str(item["direction"]) == direction
               for item in opened.values()) >= 3:
            continue
        opened[symbol] = row
        last[(symbol, direction)] = now
        accepted_times.append(now)
        output.append(row)
        heapq.heappush(pending, (float(row["exit_time"]), len(output), row))
    return output


def signal_portfolio(records):
    rows=[dict(r,module='risk_guard_research',source_priority=3,module_priority=999,
        signed_price_return=r['risk_pct']*r['net_r'],
        outcome='SL' if r['sequential_outcome']=='SL' else 'TP') for r in records]
    return runtime_gate(rows,latch_loss_pause=False)


def main():
    tracked=subprocess.check_output(['git','show','HEAD:src/risk_limits.py']).decode('utf-8')
    node=next(n for n in ast.parse(tracked).body if isinstance(n,ast.FunctionDef) and n.name=='equity_guard')
    actual=inspect.getsource(equity_guard).strip().replace('\r\n','\n')
    if actual!=ast.get_source_segment(tracked,node).strip().replace('\r\n','\n'):
        raise ValueError('equity guard differs from tracked production function')
    source=ROOT/'crypto_peer_breadth_nov17.json'
    frozen=json.loads(source.read_text(encoding='utf-8'))
    manifest=ROOT/'xperp_15m_nov14_merged/manifest.json'
    meta=json.loads(manifest.read_text(encoding='utf-8'))
    candles={};hashes={}
    for symbol,item in meta['symbols'].items():
        if not item.get('path'):
            continue
        p=Path(item['path']);raw=p.read_bytes();digest=hashlib.sha256(raw).hexdigest()
        if digest!=item['sha256']:raise ValueError('native candle hash mismatch')
        candles[symbol]=pickle.loads(raw);hashes[str(p)]=digest
    indices={s:{int(t):i for i,t in enumerate(c['time'])} for s,c in candles.items()}
    records=[]
    for item in frozen['candidate_diagnostics']:
        c=candles[item['symbol']];i=indices[item['symbol']][int(item['entry_time'])]
        entry=float(c['close'][i-1]);sign=1 if item['direction']=='LONG' else -1
        stop=entry-sign*entry*item['risk_pct']
        path=replay(c,i,direction=item['direction'],entry=entry,stop=stop)
        net=.025+.95*path['runner_r'] if path['banked'] else path['runner_r']
        if abs(net-item['model_r'])>1e-9 or path['exit_time']!=item['exit_time']:
            raise ValueError('candidate public path parity failed')
        records.append(dict(item,entry=entry,sl=stop,net_r=net,
            sequential_banked=path['banked'],sequential_runner_r=path['runner_r'],
            sequential_outcome=path['outcome'],
            near_exit_time=path['near_exit_time'] if path['near_exit_time'] is not None else math.nan))
    print('literal_candidate_parity',len(records),flush=True)
    stock_path=STOCK/'stock_close_anchor_nov16_causal_candidates_known_close.pkl'
    stock=stock_ready_rows(pd.read_pickle(stock_path)).assign(bot='stocks',
        sequential_banked=False,near_exit_time=math.nan)
    stock['sequential_runner_r']=stock.net_r
    spec_path=ROOT/'shared_xperp_specs_nov14.json'
    specs=json.loads(spec_path.read_text(encoding='utf-8'))
    decisions={(r['day_start'],r['btc_regime']):r['policy'] for r in frozen['daily_rotation_decisions']}
    report=dict(status='CONTINUOUS_REALIZED_CASH_GUARD_DIAGNOSTIC',objective_verified=False,
        live_activation_supported=False,capital=130.,daily_loss_limit=.01,max_drawdown=.03,
        starting_utc='2026-07-01',candidate_path_parity=len(records),results={},
        guard_matches_tracked_production=True,
        equity_guard_function_sha256=hashlib.sha256(actual.encode('utf-8')).hexdigest(),
        limitations=[
            'Flat hypothetical account starts July 1; no actual customer ledger or inherited positions.',
            'Guard sees realized cash only at hypothetical entry attempts; real OKX totalEq includes floating PnL.',
            'Same production equity_guard; independent state per bot, common cash; known initial peak floor $130.',
            'Continuous guards never reset monthly; an additional monthly-reset comparison is explicitly counterfactual.',
            'Market prices, signal feed and private execution are not reconstructed; publication/quote failures are absent.',
            'Previously inspected native data; no independent holdout and no monthly profit forecast.',
            '15m stop-first paths, delayed recognition and two-writer headroom are proxies for actual execution.',
            'No deposits/withdrawals,funding,fees,MTM equity or liquidation reconstruction.',
            'Model signal/stock-paper eligibility continues independently of account entry refusals.',
            'Daily policy selection is imported from a frozen exploratory report; never selected on later account profits.'])
    for policy in ('baseline','opposition_both','daily_causal_rotation'):
        selected=[]
        for r in records:
            p=policy
            if p=='daily_causal_rotation':p=decisions[(int(r['entry_time']//86400)*86400,r['btc_regime'])]
            if p!='cash' and r['allowed'][p]:selected.append(r)
        crypto=pd.DataFrame(signal_portfolio(selected)).assign(bot='crypto')
        feed=pd.concat([crypto,stock],ignore_index=True)
        reset={};continuous={}
        for priority in ('crypto','stocks'):
            options=dict(capital=130.,slots=12,bot_slots=dict(crypto=4,stocks=8),
                simultaneous_writers=2,release_delay=900,
                tie_key=lambda r,p=priority:(r['bot']!=p,r['symbol']))
            # Exactly reproduce the earlier flat monthly-cohort result first.
            reset[priority]={}
            for month in (7,8,9,10,11):
                label=f'2026-{month:02}';start,end=stamp(label+'-01'),stamp(f'2026-{month+1:02}-01')
                cohort=feed[feed.entry_time.ge(start)&feed.exit_time.lt(end)].to_dict('records')
                base=simulate(cohort,specs,**options)
                expected=frozen['results'][policy]['joint_cash'][label][priority+'_first']
                for key in ('ending_capital','return_pct','cash_drawdown_pct','trades'):
                    if abs(base[key]-expected[key])>1e-6:raise ValueError('monthly cash baseline parity failed: '+key)
                guard=RealizedCashGuard(capital=130.)
                tested=simulate(cohort,specs,**options,entry_guard=guard)
                tested.pop('ledger')
                reset[priority][label]=dict(summary=tested,guard_states=guard.states,first_blocks=guard.first_blocks,
                    first_drawdown_blocks=guard.first_drawdown_blocks)
            active=feed[feed.entry_time.ge(stamp('2026-07-01'))].to_dict('records')
            continuous[priority]={}
            for mode in ('no_account_guard','realized_cash_guard'):
                guard=RealizedCashGuard(capital=130.) if mode=='realized_cash_guard' else None
                value=simulate(active,specs,**options,entry_guard=guard,capture_events=True)
                assert value['peak_positions']<=12 and value['peak_reserved_margin_pct']<=90+1e-9
                monthly=calendar_months(value)
                value.pop('ledger');value.pop('cash_events');value.pop('guard_rejections')
                continuous[priority][mode]=dict(summary=value,calendar_months=monthly,
                    guard_states=guard.states if guard else {},first_blocks=guard.first_blocks if guard else {},
                    first_drawdown_blocks=guard.first_drawdown_blocks if guard else {})
                print(policy,priority,mode,'months',[(m,round(v['return_pct'],3),v['entries']) for m,v in monthly.items()],flush=True)
        report['results'][policy]=dict(monthly_cash_baseline_parity=True,monthly_reset_guarded=reset,
                                      continuous=continuous)
    for p in (source,manifest,stock_path,spec_path,Path(__file__),Path('account_cash_guard_nov17.py'),
              Path('lot_aware_tp1_audit.py'),Path('crypto_runner_exit_path_nov16.py'),
              Path('shared_factor_account_nov14_audit.py'),
              Path('src/risk_limits.py'),Path('src/okx_trader.py')):
        hashes[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
    report['sources']=hashes
    (ROOT/'account_cash_guard_nov17.json').write_text(json.dumps(report,indent=2),encoding='utf-8')


if __name__=='__main__':main()
