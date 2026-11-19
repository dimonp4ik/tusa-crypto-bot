"""Pre-entry activity gate plus paired runner banking; research only."""
from collections import Counter
from datetime import datetime,timezone
import hashlib
import json
import math
from pathlib import Path
import pickle
import argparse

from banked_runner_path_nov19 import POLICIES
from crypto_minimum_bank_runner_nov19_audit import checked_row,admitted_records,cash_result,CLIENTS
from crypto_native_price_pool_nov19 import load_pool,ROOT,MANIFEST,META

PREFIX='crypto_active_entry_runner_nov19'


def activity_gate(c,index):
    if index<16 or index>len(c['time']):return False
    volumes=[float(v) for v in c['volume'][index-16:index]]
    if len(volumes)!=16 or any(not math.isfinite(v) or v<0 for v in volumes):return False
    # No future print, pending exit, label or classifier outcome is read.
    return volumes[-1]>0 and sum(v>0 for v in volumes)>=12


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--declare-only',action='store_true')
    args=parser.parse_args()
    paths=[Path(__file__),Path('banked_runner_path_nov19.py'),Path('crypto_minimum_bank_runner_nov19_audit.py'),
        Path('crypto_native_price_pool_nov19.py'),Path('crypto_native_price_features_nov19.py'),
        Path('account_cash_guard_nov17_audit.py'),Path('account_cash_guard_nov17.py'),
        Path('cash_tp1_policy_nov19.py'),Path('src/tp1_plan.py'),Path('src/okx_trader.py'),
        ROOT/'shared_xperp_specs_nov14.json',ROOT/'crypto_minimum_bank_runner_v2_nov19.json',MANIFEST,META]
    sources={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    protocol=dict(declared_at=datetime.now(timezone.utc).isoformat(),sources=dict(sources),
        hypothesis='12 of16 positive closed trade-volume bars, last closed bar positive; no exit-label filtering',
        motivation='Known stale expiry paths in old comparison; same12/16 rule was already declared for new entry scenarios.',
        comparisons=list(POLICIES),clients=CLIENTS,capital=130.,margin_fraction=.03,leverage=10,
        cash='continuous from July1; current stocks PAPER; cap4crypto/12shared,90% margin, delayed shared release',
        unknown='Admitted unknown path makes the whole scenario unscorable, never removed or marked flat.',
        selection=None,previously_inspected_history=True,objective_verified=False,live_activation_supported=False)
    declaration=ROOT/f'{PREFIX}_protocol.json'
    if declaration.exists():
        previous=json.loads(declaration.read_text(encoding='utf-8'))
        if previous['sources']!=sources:raise ValueError('active_entry_runner_declared_sources_changed')
        protocol=previous
    else:declaration.write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    if args.declare_only:
        print('active_entry_runner_declared',protocol['declared_at'],flush=True);return
    frame,_=load_pool();proposals=frame[frame.is_frozen_candidate].to_dict('records');candles={}
    for symbol,item in json.loads(MANIFEST.read_text(encoding='utf-8'))['symbols'].items():
        raw=Path(item['path']).read_bytes()
        if hashlib.sha256(raw).hexdigest()!=item['sha256']:raise ValueError('active_entry_candle_hash_changed')
        c=pickle.loads(raw);candles[symbol]=(c,{int(t):i for i,t in enumerate(c['time'])})
        sources[str(item['path'])]=item['sha256']
    accepted=[];rejected=Counter()
    for row in proposals:
        c,lookup=candles[row['symbol']];i=lookup[int(row['entry_time'])]
        zero_share=sum(float(v)==0 for v in c['volume'][i-16:i])/16
        if abs(zero_share-row['feature__zero_volume_share16'])>1e-12:
            raise ValueError('cached_entry_activity_feature_mismatch')
        if activity_gate(c,i):accepted.append(row)
        else:rejected[row['symbol']]+=1
    specs=json.loads((ROOT/'shared_xperp_specs_nov14.json').read_text(encoding='utf-8'));results={}
    for policy in POLICIES:
        rows=[]
        for row in accepted:
            c,lookup=candles[row['symbol']]
            rows.append(checked_row(row,c,lookup[int(row['entry_time'])],policy,.0005))
        admitted,unknown=admitted_records(rows)
        diagnostic=dict(activity_eligible=len(rows),unknown_proposals=sum(not r['known'] for r in rows),
            admitted_unknown=unknown)
        if unknown:
            results[policy]=dict(status='UNSCORABLE_UNKNOWN_PATHS',diagnostics=diagnostic)
            print(policy,'unknown',diagnostic,flush=True);continue
        diagnostic['publication_admitted']=len(admitted)
        value={client:cash_result(admitted,specs,client) for client in CLIENTS}
        results[policy]=dict(status='RESOLVED_RESEARCH_CASH',diagnostics=diagnostic,cash=value)
        print(policy,{p:dict(total=v['summary']['return_pct'],dd=v['summary']['cash_drawdown_pct'],
            months={m:round(x['return_pct'],3) for m,x in v['calendar_months'].items()}) for p,v in value.items()},flush=True)
    report=dict(status='PRE_ENTRY_ACTIVITY_AND_RUNNER_SENSITIVITY',protocol=protocol,sources=sources,
        candidates=len(proposals),activity_eligible=len(accepted),activity_rejected=dict(rejected),
        results=results,selected_policy=None,objective_verified=False,live_activation_supported=False,
        limitations=['Rule motivated by already observed stale paths; all historical outcomes previously inspected.',
            'Only entry-time closed volume filters proposals; no unknown or losing outcome is silently removed.',
            'Historical native contract candles/specs; unrounded original known-close brackets and5bp exit stress are proxies.',
            'Cash accounts for client lot-rounded TP1 and shared caps, not floating equity, current guard, fees/funding or private fills.',
            'Current stock bot remains PAPER; crypto-only study cannot prove each bot30% per month.',
            'November partial; no independent holdout or live activation supported.'])
    (ROOT/f'{PREFIX}.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    print('activity_counts',len(proposals),len(accepted),dict(rejected),flush=True)


if __name__=='__main__':main()
