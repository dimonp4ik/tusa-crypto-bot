"""Attribute common-account capacity rejects on frozen public price scenarios."""
from collections import Counter
from pathlib import Path
import hashlib
import json
import math
from datetime import datetime,timezone

from account_cash_guard_nov17 import calendar_months, stamp
from account_cash_guard_nov17_audit import signal_portfolio
from crypto_active_entry_runner_nov19_audit import activity_gate
from crypto_minimum_bank_runner_nov19_audit import checked_row, CLIENTS
from crypto_native_price_pool_nov19 import load_pool, ROOT, MANIFEST, META
from liquidity_trap_joint_nov19_audit import load_frame
from phase_scenarios_nov19_audit import stock_ready
from cash_tp1_policy_nov19 import simulate
from src.okx_trader import calc_contracts

STOCK = Path('../tusa stocks/reports/audit_2026_09_08')
STOCK_PATH = STOCK/'stock_close_anchor_nov16_causal_candidates_known_close.pkl'
STOCK_MANIFEST = STOCK/'stock_xperp_forward_nov14_merged/manifest.json'
START = stamp('2026-08-01')
SLOTS = {'crypto':4, 'stocks':8}
RELEASE_DELAY = 900


def trace_skips(rows, result, specs, *, order, writers, available_slots, margin_cap):
    """Cross-check the frozen ledger and unbundle its combined slot bucket."""
    ledger = {(r['bot'],r['symbol'],r['entry_time']):r for r in result['ledger']}
    if len(ledger) != len(result['ledger']):
        raise ValueError('duplicate filled entry in ledger')
    events = iter(result['cash_events'])
    event = next(events, None)
    equity = result['starting_capital']
    active = {}
    rejected = {bot:Counter() for bot in SLOTS}
    accepted = Counter()
    verified_margin = 0.
    maximum_overlap = 0
    for row in sorted(rows,key=lambda r:(r['entry_time'],*order(r))):
        now, symbol, bot = row['entry_time'], row['symbol'], row['bot']
        while event is not None and event['time'] < now:
            equity += event['pnl']
            event = next(events,None)
        active = {s:x for s,x in active.items() if x['release'] >= now}
        identity = (bot,symbol,now)
        filled = ledger.get(identity)
        margin = sum(r['margin'] for r in active.values())
        size = None
        if len(active)+writers > available_slots:
            reason = 'shared_slots'
        elif symbol in active:
            reason = 'instrument_occupied'
        elif sum(x['bot']==bot for x in active.values()) >= SLOTS[bot]:
            reason = 'local_slots'
        elif equity<=0 or margin + equity*.03*writers > equity*margin_cap+1e-9:
            reason = 'shared_margin'
        else:
            size = calc_contracts(equity*.03,10,row['entry'],specs[symbol])
            reason = None if size>0 else 'minimum_lot'
        if (filled is None) != (reason is not None):
            raise ValueError(f'ledger admission mismatch: {identity} {reason}')
        if reason:
            rejected[bot][reason] += 1
            continue
        if size != filled['contracts']:
            raise ValueError('lot mismatch in replay')
        size_margin = size*float(specs[symbol]['ctVal'])*row['entry']/10
        active[symbol] = dict(bot=bot,release=filled['exit_time']+RELEASE_DELAY,margin=size_margin)
        accepted[bot] += 1
        maximum_overlap = max(maximum_overlap, len(active))
        verified_margin = max(verified_margin, usage(active)/equity*100)
    if maximum_overlap != result['peak_positions'] or abs(verified_margin-result['peak_reserved_margin_pct'])>1e-9:
        raise ValueError('ledger occupancy mismatch')
    if sum(accepted.values())!=result['trades']:
        raise ValueError('ledger fills missing from input')
    return dict(accepted={bot:accepted[bot] for bot in SLOTS},
                rejected={bot:dict(v) for bot,v in rejected.items()}, peak_cross_bot_overlap=maximum_overlap)


def usage(active):
    return sum(r['margin'] for r in active.values())


def case(rows,specs,*,priority,policy,writers=1,external=0,external_margin=0.):
    order = lambda r:(r['bot']!=priority,r['symbol'])
    slots = 12-external
    margin_cap = .9-external_margin
    sim = simulate(rows,specs,slots=slots,bot_slots=SLOTS,simultaneous_writers=writers,
        release_delay=RELEASE_DELAY,tie_key=order,margin_cap=margin_cap,**CLIENTS[policy])
    admission = trace_skips(rows,sim,specs,order=order,writers=writers,available_slots=slots,margin_cap=margin_cap)
    if sim['peak_positions']+external>12 or sim['peak_reserved_margin_pct']+external_margin*100>90+1e-9:
        raise ValueError('shared limits violated in stress replay')
    if abs(sum(sim['return_contribution_pct'].values())-sim['return_pct'])>1e-6:
        raise ValueError('per-bot contribution does not sum to account result')
    for bot,n in sim['peak_by_bot'].items():
        if n>SLOTS[bot]:
            raise ValueError('local cap violated')
    monthly = {m:v for m,v in calendar_months(sim).items() if m!='2026-07'}
    for month,item in monthly.items():
        start = stamp(month+'-01')
        end = stamp(f'2026-{int(month[-2:])+1:02}-01')
        for bot in SLOTS:
            closed = [r for r in sim['ledger'] if r['bot']==bot and start<=r['exit_time']<end]
            item[bot+'_closed'] = len(closed)
            item[bot+'_winrate_pct'] = (100*sum(r['bank_pnl']+r['runner_pnl']>1e-12 for r in closed)/len(closed)
                                        if closed else None)
    fields = ('return_pct','cash_drawdown_pct','peak_positions','peak_by_bot','peak_reserved_margin_pct')
    return dict(admission=admission, monthly=monthly, **{k:sim[k] for k in fields})


def records():
    frame,_ = load_pool()
    stock_manifest = json.loads(STOCK_MANIFEST.read_bytes())['symbols']
    crypto_manifest = json.loads(MANIFEST.read_bytes())['symbols']
    specs = json.loads((ROOT/'shared_xperp_specs_nov14.json').read_bytes())
    for manifest in (stock_manifest,crypto_manifest):
        for symbol,item in manifest.items():
            if item.get('path') and specs[symbol]['instId'] != item['instrument']:
                raise ValueError('contract reference changed')
    end = min(item['end']+900 for m in (crypto_manifest,stock_manifest) for item in m.values() if item.get('path'))
    crypto = []
    for symbol,f in frame[frame.is_frozen_candidate & frame.entry_time.ge(START)].groupby('symbol'):
        item = crypto_manifest[symbol]
        bars = load_frame(Path(item['path']),item['sha256'])
        lookup = {int(t):i for i,t in enumerate(bars['time'])}
        for row in f.to_dict('records'):
            if row['entry_time']+48*900>end:
                continue
            index = lookup[int(row['entry_time'])]
            if activity_gate(bars,index):
                crypto.append(checked_row(row,bars,index,'be_tp1',.0005))
    c = signal_portfolio(crypto)
    if any(not r['known'] for r in c):
        raise ValueError('admitted crypto path unknown in shared replay')
    for row in c:
        unit = 2*row['bank_price_return']
        if not unit>0:
            raise ValueError('nonpositive bank unit')
        row.update(bot='crypto',risk_pct=unit,sequential_runner_r=row['runner_price_return']/unit)
    import pandas as pd
    stock = pd.read_pickle(STOCK_PATH)
    if stock.duplicated(['symbol','entry_time']).any() or not all(
        math.isfinite(r.net_r) and r.risk_pct>0 and abs(r.net_r*r.risk_pct-r.signed_return)<1e-10
        and r.entry_time<=r.exit_time<r.entry_time+65*900 for r in stock.itertuples()):
        raise ValueError('stock paths do not reconcile')

    def cash_stock(f):
        eligible = f[f.entry_time.ge(START) & (f.entry_time+64*900<=end)]
        return [dict(r,bot='stocks',sequential_runner_r=r['net_r'],sequential_banked=False)
                for r in eligible.to_dict('records')]
    return c,cash_stock(stock_ready(stock)),cash_stock(stock),specs,end


def main():
    base = ROOT/'shared_admission_replay_nov19'
    paths = [Path(__file__).resolve(), STOCK_PATH, STOCK_MANIFEST, MANIFEST, META,
             ROOT/'shared_xperp_specs_nov14.json', Path('cash_tp1_policy_nov19.py'),
             Path('phase_scenarios_nov19_audit.py'),Path('crypto_minimum_bank_runner_nov19_audit.py'),
             Path('crypto_active_entry_runner_nov19_audit.py'),Path('account_cash_guard_nov17_audit.py'),
             Path('banked_runner_path_nov19.py'),Path('src/tp1_plan.py'),Path('src/okx_trader.py')]
    hashes = {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    protocol = base.with_name(base.name+'_protocol.json')
    scenarios = {
        'current_stock_paper': ('paper',1,0,0.),
        'ready_one_writer': ('ready',1,0,0.),
        'ready_legacy_headroom': ('ready',2,0,0.),
        'all_stock_capacity_diagnostic': ('all',1,0,0.),
        'ready_eight_external_slots': ('ready',1,8,.60),
    }
    if protocol.exists():
        if json.loads(protocol.read_bytes())['sources'] != hashes:
            raise ValueError('shared capacity study input changed')
    else:
        protocol.write_text(json.dumps(dict(sources=hashes,scenarios=scenarios,
            declared_at=datetime.now(timezone.utc).isoformat(),
            account='one continuous $130, 3% margin, 10x, service 4/8, common 12/90%, 15min release delay',
            readiness='prior 20 stock paper exits strictly earlier with positive net R and price return',
            notes='External 8 slots and 60% margin are a synthetic pressure scenario. All-stock is a diagnostic readiness bypass only in research. Neither is a live toggle.'),indent=2),encoding='utf-8')
    c,ready,all_stocks,specs,end = records()
    stocks = {'paper':[], 'ready':ready, 'all':all_stocks}
    results = {}
    for scenario,(mode,writers,external,extra_margin) in scenarios.items():
        feed = c + stocks[mode]
        duplicates = Counter((r['bot'],r['symbol'],r['entry_time']) for r in feed)
        if any(n>1 for n in duplicates.values()):
            raise ValueError('duplicate proposed entry')
        results[scenario] = {policy:{priority:case(feed,specs,priority=priority,policy=policy,
            writers=writers,external=external,external_margin=extra_margin) for priority in SLOTS} for policy in CLIENTS}
        view = results[scenario]['half_minimum']['crypto']
        print(scenario,view['admission'],view['peak_positions'],flush=True)
    output = dict(status='SHARED_CAPACITY_ATTRIBUTION_NOT_LIVE_PNL', data_end_utc=end,
        candidate_counts=dict(crypto=len(c), stocks_paper=len(all_stocks), stocks_ready=len(ready)),
        simultaneous_entry_buckets=len({r['entry_time'] for r in c}&{r['entry_time'] for r in ready}),
        results=results, objective_verified=False, live_activation_supported=False,
        limitations=['Public reference paths, no book history/private fills, no full-equity or account drawdown pause replay.',
                     'One-writer scenario assumes atomic admission under a healthy coordinator. Actual remote sync delays and ambiguous orders are not recreated.',
                     'Stock historical frozen single-target reference may include carried expiry prices. Readiness on this history is hypothetical.',
                     'Coverage ends in partial November. Results are not independent validation.'])
    base.with_suffix('.json').write_text(json.dumps(output,indent=2,allow_nan=False),encoding='utf-8')


if __name__=='__main__':
    main()
