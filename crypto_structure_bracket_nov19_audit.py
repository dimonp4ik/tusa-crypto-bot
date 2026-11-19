"""Paired structural-stop study with fixed entry/TP prices and joint cash."""
from datetime import datetime,timezone
import hashlib
import json
import math
from pathlib import Path
import pickle
import pandas as pd

from crypto_native_price_pool_nov19 import load_pool,ROOT,META,MANIFEST
from crypto_native_price_features_nov19 import mature_before
from crypto_native_price_nov19_audit import STOCK_PATH,SPEC_PATH,SELECT_END,stamp,crypto_metrics
from crypto_native_weekly_nov19_audit import TP1_POLICIES,shared_cash,later
from crypto_structure_bracket_nov19 import POLICIES,stop_geometry,replay_fixed_targets
from account_cash_guard_nov17_audit import signal_portfolio
from shared_factor_account_nov14_audit import stock_ready_rows


def admitted(frame):
    return pd.DataFrame(signal_portfolio(frame.to_dict('records')),columns=frame.columns)


def main():
    source_paths=(Path(__file__),Path('crypto_structure_bracket_nov19.py'),
        Path('crypto_native_weekly_nov19_audit.py'),Path('cash_tp1_policy_nov19.py'),
        Path('account_cash_guard_nov17_audit.py'),META,MANIFEST,STOCK_PATH,SPEC_PATH,
        ROOT/'crypto_structure_bracket_nov19_protocol.json')
    sources={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths}
    protocol=dict(declared_at=datetime.now(timezone.utc).isoformat(),sources=dict(sources),policies=POLICIES,
        implementation_revision='v2: exact original stop-subtraction risk for target arithmetic; first run failed parity before cash selection',
        fixed_original_entry_and_tp_prices=True,stop_only_widens=True,stop_maximum_fraction=.02,
        buffer_atr=.25,pivot_left_right=[2,2],pivot_history_bars=96,
        selection_end=SELECT_END,selection='worst July/August crypto contribution minus additive DD across both TP1 policies and priorities',
        minimum_cash_trades_per_selection_case=30,no_later_retuning=True,
        compare_matched_original_stops=True,objective_verified=False,live_activation_supported=False)
    declaration=ROOT/'crypto_structure_bracket_nov19_v2_protocol.json'
    if declaration.exists():
        old=json.loads(declaration.read_text(encoding='utf-8'))
        if old['sources']!=sources:raise ValueError('structure_declared_source_changed')
        protocol=old
    else:declaration.write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    pool,meta=load_pool();frame=pool[pool.is_frozen_candidate].copy();candles={};lookups={}
    for symbol,item in json.loads(MANIFEST.read_text(encoding='utf-8'))['symbols'].items():
        p=Path(item['path']);raw=p.read_bytes()
        if hashlib.sha256(raw).hexdigest()!=item['sha256']:raise ValueError('structure_source_changed')
        candles[symbol]=pickle.loads(raw);lookups[symbol]={int(t):i for i,t in enumerate(candles[symbol]['time'])}
    variants={};matched={};geometry={};pairs={}
    for policy in POLICIES:
        rows=[];same=[];changed=0;cross={}
        for row in frame.to_dict('records'):
            c=candles[row['symbol']];i=lookups[row['symbol']][int(row['entry_time'])]
            g=stop_geometry(c,i,row,policy)
            if g is None:continue
            old_risk=abs(row['entry']-row['sl'])
            value=replay_fixed_targets(c,i,direction=row['direction'],entry=row['entry'],stop=g['stop'],original_risk=old_risk)
            if policy=='control':
                if (abs(value['runner_r']-row['sequential_runner_r'])>1e-9
                        or value['exit_time']!=row['exit_time'] or value['banked']!=row['sequential_banked']
                        or value['outcome']!=row['sequential_outcome']):
                    raise ValueError('fixed_target_control_path_parity_failed')
            changed+=g['changed'];same.append(row)
            code=row['sequential_outcome']+' -> '+value['outcome'];cross[code]=cross.get(code,0)+1
            # risk_pct remains the ORIGINAL price unit solely for cash/R
            # normalization. actual_stop_fraction records the changed stop.
            # This keeps TP1's original .5R PRICE and original2R final price.
            out=dict(row,sl=g['stop'],actual_stop_fraction=g['actual_risk_fraction'],
                exit_time=value['exit_time'],sequential_outcome=value['outcome'],
                sequential_runner_r=value['runner_r'],sequential_banked=value['banked'],
                near_exit_time=value['near_exit_time'] if value['near_exit_time'] is not None else math.nan,
                net_r=.025+.95*value['runner_r'] if value['banked'] else value['runner_r'],
                runner_price_return=row['risk_pct']*value['runner_r'])
            rows.append(out)
        variants[policy]=pd.DataFrame(rows,columns=list(frame.columns)+['actual_stop_fraction'])
        matched[policy]=pd.DataFrame(same,columns=frame.columns)
        geometry[policy]=dict(retained=len(rows),rejected=len(frame)-len(rows),widened=changed)
        pairs[policy]=cross
    if len(variants['control'])!=meta['frozen_candidate_path_parity']:raise ValueError('incomplete_structure_control')
    stock=stock_ready_rows(pd.read_pickle(STOCK_PATH)).assign(bot='stocks',sequential_banked=False,near_exit_time=math.nan)
    stock['sequential_runner_r']=stock.net_r;specs=json.loads(SPEC_PATH.read_text(encoding='utf-8'))
    selection={};eligible=[]
    for policy in POLICIES:
        parts=[];matched_parts=[]
        for start,end in ((stamp('2026-07-01'),stamp('2026-08-01')),(stamp('2026-08-01'),SELECT_END)):
            f=variants[policy];m=matched[policy]
            rows=admitted(f[f.entry_time.ge(start)&mature_before(f,end)])
            original=admitted(m[m.entry_time.ge(start)&mature_before(m,end)])
            for tp1 in TP1_POLICIES:
                for priority in ('crypto','stocks'):
                    options=dict(start=start,end=end,policy=tp1,priority=priority)
                    parts.append(dict(start=start,tp1=tp1,priority=priority,**crypto_metrics(shared_cash(rows,stock,specs,**options))))
                    matched_parts.append(dict(start=start,tp1=tp1,priority=priority,**crypto_metrics(shared_cash(original,stock,specs,**options))))
        valid=all(p['closed']>=30 and p['contribution_pct']>0 for p in parts)
        score=min(p['contribution_pct']-p['additive_crypto_cash_drawdown_pct'] for p in parts)
        selection[policy]=dict(parts=parts,matched_original=matched_parts,qualified=valid,score=score)
        if valid:eligible.append((score,sum(p['contribution_pct'] for p in parts),policy))
        print('structure_selection',policy,geometry[policy],'score',score,'qualified',valid,flush=True)
    best=max(eligible) if eligible else None;chosen=best[2] if best else None
    results={'control':later(admitted(variants['control'][variants['control'].entry_time.ge(SELECT_END)]),stock,specs)}
    if chosen is not None and chosen!='control':
        results['selected']=later(admitted(variants[chosen][variants[chosen].entry_time.ge(SELECT_END)]),stock,specs)
        results['matched_original']=later(admitted(matched[chosen][matched[chosen].entry_time.ge(SELECT_END)]),stock,specs)
    report=dict(status='STRUCTURAL_STOP_PAIRED_REFERENCE_RESEARCH',objective_verified=False,live_activation_supported=False,
        protocol=protocol,control_path_parity=len(variants['control']),geometry=geometry,
        paired_outcomes_descriptive=pairs,selection=selection,chosen_policy=chosen,later=results,sources=sources,
        limitations=[
            'All market eras inspected previously; structural hypotheses are exploratory, no new untouched holdout.',
            'Entries and TP prices frozen; only initial stop changes or entry-known geometry rejection.',
            'risk_pct/R retain the original normalization unit; actual_stop_fraction records actual changed stop distance.',
            'Matched original-stop controls distinguish entry exclusion from a stop-placement benefit.',
            'Later openings precede their ranges; stop-first ambiguous bars, BE active from following bar.',
            'Fifteen-minute public last-price references, not executable quotes or private fills; zero-volume bars remain public proxies.',
            'Both client TP1 modes and priorities; continuous later shared account with current lot specs assumed historically.',
            'No full-equity guard or minute recovery,floating PnL,funding,fees,latency or liquidation reconstruction.',
            'Stock is an old frozen reference feed, not current PAPER/live-account behavior; November partial.',
            'No margin change, live activation or promise of30% each month.'])
    (ROOT/'crypto_structure_bracket_nov19.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    print('structure_chosen',chosen,flush=True)


if __name__=='__main__':main()
