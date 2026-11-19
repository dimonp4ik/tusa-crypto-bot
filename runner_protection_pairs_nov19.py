"""Path transition counts; unknown portfolio PnL is never scored or omitted."""
from collections import Counter
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import pickle

from banked_runner_path_nov19 import POLICIES
from crypto_minimum_bank_runner_nov19_audit import checked_row
from crypto_native_price_pool_nov19 import load_pool,ROOT,MANIFEST
from src.okx_trader import calc_contracts
from src.tp1_plan import plan_tp1_close


def isolated_pnl(row,spec):
    # Fixed independent notional for path comparison, no reinvestment or live
    # admission claim. Client3% of130 margin,10x and actual contract rounding.
    size=calc_contracts(130*.03,10,row['entry'],spec)
    plan=plan_tp1_close(size,50,spec,ensure_minimum=True)
    bank=plan['size'] if row['sequential_banked'] else 0.
    return float(spec['ctVal'])*row['entry']*(bank*row['bank_price_return']
        +(size-bank)*row['runner_price_return'])


def main():
    paths=[Path(__file__),Path('banked_runner_path_nov19.py'),Path('crypto_minimum_bank_runner_nov19_audit.py'),
        Path('src/tp1_plan.py'),Path('src/okx_trader.py'),MANIFEST,
        ROOT/'crypto_minimum_bank_runner_v2_nov19.json',ROOT/'shared_xperp_specs_nov14.json']
    sources={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    prefix=ROOT/'runner_protection_pairs_nov19';declaration=prefix.with_name(prefix.name+'_protocol.json')
    protocol=dict(declared_at=datetime.now(timezone.utc).isoformat(),sources=sources,
        analysis='paired path transitions only; common-known intersections are diagnostics, never portfolio returns',
        fixed_independent_margin=3.9,client_tp1='50%/minimum lot',exit_stress=.0005,
        objective_verified=False,live_activation_supported=False)
    if declaration.exists():
        previous=json.loads(declaration.read_text(encoding='utf-8'))
        if previous['sources']!=sources:raise ValueError('runner_pairs_declared_sources_changed')
        protocol=previous
    else:declaration.write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    frame,_=load_pool();proposals=frame[frame.is_frozen_candidate].to_dict('records')
    specs=json.loads((ROOT/'shared_xperp_specs_nov14.json').read_text(encoding='utf-8'))
    candles={}
    for symbol,item in json.loads(MANIFEST.read_text(encoding='utf-8'))['symbols'].items():
        raw=Path(item['path']).read_bytes()
        if hashlib.sha256(raw).hexdigest()!=item['sha256']:raise ValueError('paired_candle_hash_changed')
        c=pickle.loads(raw);candles[symbol]=(c,{int(t):i for i,t in enumerate(c['time'])})
    rows={p:[] for p in POLICIES}
    for row in proposals:
        c,lookup=candles[row['symbol']]
        for policy in POLICIES:
            rows[policy].append(checked_row(row,c,lookup[int(row['entry_time'])],policy,.0005))
    results={}
    for policy in POLICIES:
        transitions=Counter();better=worse=equal=0;delta=0.;unknown=[]
        targets_recovered=new_runner_losses=banked_both=0
        for a,b in zip(rows['be_tp1'],rows[policy]):
            if not a['known'] or not b['known']:
                unknown.append(dict(symbol=a['symbol'],entry_time=a['entry_time'],
                    baseline_reason=a.get('unknown_reason'),alternative_reason=b.get('unknown_reason')))
                continue
            transitions[(a['sequential_outcome'],b['sequential_outcome'])]+=1
            difference=isolated_pnl(b,specs[b['symbol']])-isolated_pnl(a,specs[a['symbol']])
            better+=difference>1e-10;worse+=difference<-1e-10;equal+=abs(difference)<=1e-10;delta+=difference
            targets_recovered+=a['sequential_outcome']!='TARGET' and b['sequential_outcome']=='TARGET'
            new_runner_losses+=a['runner_price_return']>=-.00051 and b['runner_price_return']<-.00051
            banked_both+=a['sequential_banked'] and b['sequential_banked']
        results[policy]=dict(total_proposals=len(proposals),unknown_pairs=unknown,
            known_pairs=better+worse+equal,better=better,worse=worse,equal=equal,
            recovered_final_targets=targets_recovered,new_runner_losses=new_runner_losses,
            first_target_banked_in_both=banked_both,
            fixed_size_pnl_delta_sum_known_pairs=delta,
            transitions=[dict(baseline=a,alternative=b,count=n) for (a,b),n in sorted(transitions.items())])
        print(policy,'known',better+worse+equal,'better/worse',better,worse,
            'targets_recovered',targets_recovered,'new_runner_losses',new_runner_losses,
            'fixed_size_delta',round(delta,4),flush=True)
    report=dict(status='PAIRED_RUNNER_TRANSITIONS_NOT_PORTFOLIO_PERFORMANCE',protocol=protocol,
        results=results,objective_verified=False,live_activation_supported=False,selected_policy=None,
        limitations=['All hypothetical proposals, not a sequence of executable trades; positions overlap.',
            'Unknown intersections reported explicitly; fixed-size known-pair delta is not account return or expectancy.',
            'Common account caps, cash compounding, full equity, private fills, bid/ask,funding/fees not reconstructed.',
            'Previously inspected native data; no independent validation or selection of a live policy.'])
    prefix.with_suffix('.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')


if __name__=='__main__':main()
