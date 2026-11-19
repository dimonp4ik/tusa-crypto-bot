"""Paired exits and minimum-lot banking on the frozen crypto entry proposals."""
from datetime import datetime,timezone
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import pickle
import argparse

from banked_runner_path_nov19 import resolve,POLICIES
from crypto_native_price_pool_nov19 import load_pool,ROOT,MANIFEST,META
from account_cash_guard_nov17_audit import signal_portfolio
from account_cash_guard_nov17 import calendar_months,stamp
from cash_tp1_policy_nov19 import simulate

PREFIX='crypto_minimum_bank_runner_nov19'
START=stamp('2026-07-01')
CLIENTS=dict(requested5=dict(tp1_pct=5,minimum_close=False),half_minimum=dict(tp1_pct=50,minimum_close=True))


def checked_row(row,c,index,policy,stress):
    result=resolve(c,index,direction=row['direction'],entry=row['entry'],stop=row['sl'],policy=policy,exit_stress=stress)
    if not result['known']:
        return dict(row,known=False,unknown_reason=result['reason'])
    price=(.05*result['bank_price_return']+.95*result['runner_price_return'] if result['banked']
        else result['runner_price_return'])
    return dict(row,known=True,exit_time=result['exit_time'],
        sequential_outcome=result['outcome'],sequential_banked=result['banked'],
        near_exit_time=result['near_exit_time'] if result['near_exit_time'] is not None else math.nan,
        runner_price_return=result['runner_price_return'],bank_price_return=result['bank_price_return'],
        net_r=price/row['risk_pct'])


def cash_result(records,specs,client):
    rows=[]
    for row in signal_portfolio(records):
        if row['entry_time']<START:continue
        unit=2*row['bank_price_return']
        if not unit>0:raise ValueError('invalid_runner_bank_unit')
        rows.append(dict(row,bot='crypto',actual_risk_fraction=row['risk_pct'],risk_pct=unit,
            sequential_runner_r=row['runner_price_return']/unit))
    v=simulate(rows,specs,capital=130.,slots=12,bot_slots=dict(crypto=4,stocks=8),
        simultaneous_writers=2,release_delay=900,**CLIENTS[client])
    if (v['peak_positions']>12 or v['peak_by_bot'].get('crypto',0)>4
            or v['peak_reserved_margin_pct']>90+1e-9):raise ValueError('runner_shared_capacity_violation')
    months=calendar_months(v)
    for month,item in months.items():
        start=stamp(month+'-01');end=stamp(f'2026-{int(month[-2:])+1:02}-01')
        closed=[r for r in v['ledger'] if start<=r['exit_time']<end]
        pnl=[r['bank_pnl']+r['runner_pnl'] for r in closed]
        item.update(closed=len(closed),winrate_pct=100*sum(p>1e-12 for p in pnl)/len(pnl) if pnl else None,
            bank_pnl=sum(r['bank_pnl'] for r in closed),runner_pnl=sum(r['runner_pnl'] for r in closed),
            tp1_unbanked=sum(r['tp1_hit'] and r['planned_partial']==0 for r in closed))
    return dict(summary={k:x for k,x in v.items() if k not in ('ledger','cash_events','guard_rejections')},
        calendar_months=months)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--declare-only',action='store_true')
    args=parser.parse_args()
    paths=[Path(__file__),Path('banked_runner_path_nov19.py'),Path('crypto_native_price_pool_nov19.py'),
        Path('crypto_native_price_features_nov19.py'),Path('crypto_runner_exit_path_nov16.py'),
        Path('account_cash_guard_nov17_audit.py'),Path('account_cash_guard_nov17.py'),
        Path('cash_tp1_policy_nov19.py'),Path('src/tp1_plan.py'),Path('src/okx_trader.py'),
        ROOT/'shared_xperp_specs_nov14.json',ROOT/'crypto_tp1_bank_size_nov19.json',MANIFEST,META]
    sources={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    protocol=dict(declared_at=datetime.now(timezone.utc).isoformat(),sources=sources,
        policies=list(POLICIES),client_policies=CLIENTS,entry_cohort='all frozen native candidate proposals',
        scenario='stocks remain PAPER: no real stock entries; common-account crypto cap4/shared cap12 and90% margin',
        capital=130.,client_margin_fraction=.03,leverage=10,horizon_bars=48,
        entry_levels='original known-close references and original fixed0.5R/2R targets',
        checked_exit_stress=.0005,controls=['legacy_unchecked_zero_stress','be_tp1_zero_stress'],
        policy_comparison='paired exploratory sensitivity on previously inspected history; no fitted selection or holdout claim',
        unknown_paths='case unscorable if any full-horizon candidate path is unknown; no silent outcome deletion',
        market_fills_and_price_grid_verified=False,objective_verified=False,live_activation_supported=False)
    declaration=ROOT/f'{PREFIX}_protocol.json'
    if declaration.exists():
        previous=json.loads(declaration.read_text(encoding='utf-8'))
        if previous['sources']!=sources:raise ValueError('banked_runner_declared_sources_changed')
        protocol=previous
    else:declaration.write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    if args.declare_only:
        print('banked_runner_declared',protocol['declared_at'],flush=True);return
    pool,meta=load_pool();frame=pool[pool.is_frozen_candidate]
    manifest=json.loads(MANIFEST.read_text(encoding='utf-8'));candles={};indices={}
    for symbol,item in manifest['symbols'].items():
        path=Path(item['path']);raw=path.read_bytes()
        if hashlib.sha256(raw).hexdigest()!=item['sha256']:raise ValueError('runner_candle_hash_changed')
        candles[symbol]=pickle.loads(raw);indices[symbol]={int(t):i for i,t in enumerate(candles[symbol]['time'])}
        sources[str(path)]=item['sha256']
    specs=json.loads((ROOT/'shared_xperp_specs_nov14.json').read_text(encoding='utf-8'))
    proposals=frame.to_dict('records');results={};checked_baseline=None
    cases=[('legacy_unchecked_zero_stress',None,0.),('be_tp1_zero_stress','be_tp1',0.)]
    cases.extend((p,p,.0005) for p in POLICIES)
    for name,policy,stress in cases:
        if policy is None:
            records=[dict(r,known=True,bank_price_return=.5*r['risk_pct']) for r in proposals]
        else:
            records=[checked_row(r,candles[r['symbol']],indices[r['symbol']][int(r['entry_time'])],policy,stress)
                for r in proposals]
        unknown=[r for r in records if not r['known']]
        diagnostic=dict(candidates=len(records),unknown=len(unknown),unknown_reasons=dict(Counter(
            r['unknown_reason'] for r in unknown)))
        if unknown:
            results[name]=dict(status='UNSCORABLE_UNKNOWN_PATHS',diagnostics=diagnostic)
            print(name,'unknown',diagnostic,flush=True);continue
        if name=='be_tp1_zero_stress':checked_baseline=records
        if checked_baseline is not None:
            diagnostic['changed_from_checked_baseline']=sum(
                a['exit_time']!=b['exit_time'] or abs(a['runner_price_return']-b['runner_price_return'])>1e-12
                or a['sequential_banked']!=b['sequential_banked'] for a,b in zip(records,checked_baseline))
        value={client:cash_result(records,specs,client) for client in CLIENTS}
        if policy is None:
            prior=json.loads((ROOT/'crypto_tp1_bank_size_nov19.json').read_text(encoding='utf-8'))
            for client in CLIENTS:
                expected=prior['results']['crypto_only'][client]['summary']
                for key in ('return_pct','cash_drawdown_pct','trades','tp1_hits_without_bank'):
                    if abs(value[client]['summary'][key]-expected[key])>1e-6:
                        raise ValueError('legacy_runner_cash_parity_failed:'+key)
        results[name]=dict(status='RESOLVED_CASH_SENSITIVITY',diagnostics=diagnostic,cash=value)
        print(name,{p:dict(total=v['summary']['return_pct'],dd=v['summary']['cash_drawdown_pct'],
            months={m:round(x['return_pct'],3) for m,x in v['calendar_months'].items()}) for p,v in value.items()},flush=True)
    report=dict(status='CHECKED_MINIMUM_BANK_RUNNER_SENSITIVITY',protocol=protocol,sources=sources,
        results=results,objective_verified=False,live_activation_supported=False,selected_policy=None,
        limitations=['Already inspected native history; comparison is exploratory, not independent validation.',
            'Current stocks remain PAPER; crypto-only cash cannot prove shared two-bot monthly target.',
            'Original unrounded published price references are not private fills or verified current price-grid orders.',
            'Known candle close amendments and5bp exit stress are scenario proxies; real latency/bid-ask unknown.',
            'New stops apply only after the observing candle; a crossed amendment exits at its close, not a past stop.',
            'Trade-volume-zero bars do not bank or trigger; stale terminal references remain unknown.',
            'Full-equity guard,floating PnL,funding,fees,liquidation and publication/account rejections omitted.',
            'Client margin3%,10x,contracts rounded; November partial; no guarantee of30% every month.'])
    (ROOT/f'{PREFIX}.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')


if __name__=='__main__':main()
