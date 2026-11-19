"""Frozen trap/pullback hypotheses with tick prices and expanded native assets."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
import math
from pathlib import Path
import pickle
import bisect
import pandas as pd

from liquidity_trap_entries_nov19 import candidates as new_candidates
from phase_entry_scenarios_nov19 import candidates as phase_candidates,HORIZON
from price_grid_bracket_nov19 import rounded_bracket
from price_bracket_path_nov19 import resolve
from phase_scenarios_nov19_audit import (frame_of,mature_before,portfolio,stock_ready,cash,
    ROOT,STOCK_ROOT,CRYPTO_MANIFEST,STOCK_MANIFEST,SPEC_PATH,TP1_POLICIES)
from account_cash_guard_nov17 import stamp

EXPANSION=ROOT/'native_expansion_nov19'
ROUTES={'phase_control':{'range_sweep_confirm','trend_retest_confirm'},
    'failed_only':{'failed_break_accept'},'pullback_only':{'trend_pullback_hold'},
    'new_union':{'failed_break_accept','trend_pullback_hold'},
    'all_scenarios':{'range_sweep_confirm','trend_retest_confirm','failed_break_accept','trend_pullback_hold'}}


def load_frame(path,digest):
    raw=path.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=digest:raise ValueError('trap_input_frame_changed')
    return pickle.loads(raw)


def attach_outcome(row,path,trade_volume):
    result=resolve(path,row['entry_index'],direction=row['direction'],entry=row['entry'],stop=row['sl'],
        tp1=row['tp1'],tp2=row['tp2'],expire_at=row['expire_at'],require_trade_volume=trade_volume)
    out=dict(row,known=result['known'])
    if not result['known']:return dict(out,unknown_reason=result['reason'],exit_time=row['entry_time']+HORIZON*900)
    price=.5*(result['bank_price_return']+result['runner_price_return']) if result['banked'] else result['runner_price_return']
    return dict(out,exit_time=result['exit_time'],runner_price_return=result['runner_price_return'],
        bank_price_return=result['bank_price_return'],signed_return=price,net_r=price/row['risk_pct'],
        sequential_outcome={'TARGET':'TP_RUNNER','TIME':'TP_EXPIRED' if result['banked'] else 'EXPIRED'}.get(result['outcome'],result['outcome']),
        sequential_banked=result['banked'],near_exit_time=result['near_exit_time'] if result['near_exit_time'] is not None else math.nan)


def select_rows(frame,route,universe,start,end=None):
    f=frame[frame.family.isin(ROUTES[route])&frame.entry_time.ge(start)]
    if universe=='base':f=f[f.origin.eq('base')]
    if end is not None:f=f[mature_before(f,end)]
    return f.copy()


def verified_cash(c,s,specs,*,policy,priority,start,end_month=None):
    result=cash(c,s,specs,policy=policy,priority=priority,start=start)
    summary=result['summary']
    if any(summary['peak_by_bot'].get(bot,0)>cap for bot,cap in (('crypto',4),('stocks',8))):
        raise ValueError('trap_local_slot_violation')
    if abs(sum(summary['return_contribution_pct'].values())-summary['return_pct'])>1e-4:
        raise ValueError('trap_cash_accounting_mismatch')
    if end_month is not None:
        result['calendar_months']={m:v for m,v in result['calendar_months'].items() if m<=end_month}
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--declare-only',action='store_true')
    args=parser.parse_args()
    paths=(Path(__file__),Path('liquidity_trap_entries_nov19.py'),Path('price_grid_bracket_nov19.py'),
        Path('phase_entry_scenarios_nov19.py'),Path('phase_scenarios_nov19_audit.py'),Path('price_bracket_path_nov19.py'),
        Path('cash_tp1_policy_nov19.py'),Path('src/okx_trader.py'),Path('src/tp1_plan.py'),
        CRYPTO_MANIFEST,STOCK_MANIFEST,SPEC_PATH,EXPANSION/'selection.json')
    sources={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    protocol=dict(declared_at=datetime.now(timezone.utc).isoformat(),sources=dict(sources),routes={k:sorted(v) for k,v in ROUTES.items()},
        universes=['base','expanded_current_snapshot'],stock_triggers=['trade','mark'],
        breakout_lookback=20,breakout_volume_minimum=1.2,breakout_body_atr=.5,
        failure_max_bars=4,failure_return_atr=.25,failure_body_atr=.4,failure_volume_minimum=1.,
        acceptance_wait_bars=2,acceptance_return_atr=.1,acceptance_volume_minimum=.75,
        pullback_zone='last closed1h EMA20',pullback_zone_tolerance_atr=.25,pullback_max_extension_atr=1.5,
        pullback_requires_closed_counterbody_then_closed_reclaim=True,macro='own4h and market4h not strongly opposite final direction',
        minimum_active_bars_last16=12,minimum_stop=.003,maximum_stop='clip(1.5ATR,0.008,0.015)',
        target1='clip(1ATR,0.005,0.01)',target2='clip(2.5ATR,0.01,0.02),cap at unbroken4h/opposite16h level',
        minimum_final_move=.008,minimum_final_reward_risk=1.5,price_grid='same nearest rounding as production; one outward tick if hard stop/TP1 minimum crossed',
        entry_exit_stress=.0005,horizon_bars=HORIZON,client_tp1_policies=TP1_POLICIES,
        selection_months=['2026-08','2026-09'],later_start='2026-10-01',minimum_each_bot_month=20,
        selection_score='minimum each-bot monthly contribution minus shared monthly cash DD across policies and priorities',
        require_each_bot_month_positive=True,unknown_paths_unscorable=True,
        expanded_rule_parameters_frozen_before_expanded_outcomes=True,
        objective_verified=False,live_activation_supported=False)
    declaration=ROOT/'liquidity_trap_joint_nov19_protocol.json'
    if declaration.exists():
        previous=json.loads(declaration.read_text(encoding='utf-8'))
        if previous['sources']!=sources:raise ValueError('trap_declared_sources_changed')
        protocol=previous
    else:declaration.write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    if args.declare_only:
        print('trap_protocol_frozen',protocol['declared_at'],flush=True);return
    manifest_path=EXPANSION/'manifest.json';expanded=json.loads(manifest_path.read_text(encoding='utf-8'))
    if not expanded['complete']:raise ValueError('expanded_download_not_complete')
    if expanded['selection_sha256']!=hashlib.sha256((EXPANSION/'selection.json').read_bytes()).hexdigest():
        raise ValueError('expanded_selection_mismatch')
    sources[str(manifest_path)]=hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    specs=json.loads(SPEC_PATH.read_text(encoding='utf-8'));datasets={}
    for symbol,item in json.loads(CRYPTO_MANIFEST.read_text(encoding='utf-8'))['symbols'].items():
        p=Path(item['path']);c=load_frame(p,item['sha256']);sources[str(p)]=item['sha256']
        if specs[symbol]['instId']!=item['instrument']:raise ValueError('trap_base_crypto_contract_changed')
        datasets[symbol]=dict(signal=c,trade=c,origin='base',bot='crypto')
    for symbol,item in json.loads(STOCK_MANIFEST.read_text(encoding='utf-8'))['symbols'].items():
        pair=[]
        for key,digest in (('path','sha256'),('trade_path','trade_sha256')):
            p=STOCK_ROOT/item[key];pair.append(load_frame(p,item[digest]));sources[str(p)]=item[digest]
        specs[symbol]=item['spec'];datasets[symbol]=dict(signal=pair[0],trade=pair[1],origin='base',bot='stocks')
    for symbol,item in expanded['symbols'].items():
        if symbol in datasets:raise ValueError('duplicate_expanded_asset')
        trade=load_frame(Path(item['trade']['path']),item['trade']['sha256'])
        signal=trade if item['category']=='1' else load_frame(Path(item['mark']['path']),item['mark']['sha256'])
        specs[symbol]=item['spec'];datasets[symbol]=dict(signal=signal,trade=trade,origin='expansion',
            bot='crypto' if item['category']=='1' else 'stocks')
    end=min(min(d['signal']['time'][-1],d['trade']['time'][-1])+900 for d in datasets.values())
    cutoff=end-HORIZON*900;records=dict(crypto=[],stock_trade=[],stock_mark=[]);diagnostics={};coverage={}
    for symbol,d in sorted(datasets.items()):
        signal,trade=d['signal'],d['trade'];is_stock=d['bot']=='stocks'
        references=[datasets['QQQUSDT']['signal'],datasets['SPYUSDT']['signal']] if is_stock else [datasets['BTCUSDT']['signal']]
        diag={};events=new_candidates(signal,trade,symbol,specs[symbol],references=references,stocks=is_stock,diagnostics=diag)
        old_diag={};control=phase_candidates(signal,trade,symbol,references=references,stocks=is_stock,
            profile='intraday_context',diagnostics=old_diag)
        aligned=[];grid_rejected=0
        for r in control:
            j=bisect.bisect_left(signal['time'],r['entry_time'])-1
            atr=sum(max(signal['high'][k]-signal['low'][k],abs(signal['high'][k]-signal['close'][k-1]),
                abs(signal['low'][k]-signal['close'][k-1])) for k in range(j-13,j+1))/14
            row=rounded_bracket(dict(r,risk_cap=min(.015,max(.008,1.5*atr/r['entry']))),specs[symbol])
            if row is None:grid_rejected+=1
            else:aligned.append(row)
        events=[dict(r,origin=d['origin']) for r in (*events,*aligned) if r['entry_time']<=cutoff]
        diagnostics[symbol]=dict(new=diag,phase_control=old_diag,control_grid_rejected=grid_rejected)
        coverage[symbol]=dict(origin=d['origin'],bot=d['bot'],instrument=specs[symbol]['instId'],
            signal_start=signal['time'][0],trade_start=trade['time'][0],end=min(signal['time'][-1],trade['time'][-1]),
            earliest_history_bound=max(signal['time'][0],trade['time'][0])+1040*900)
        if is_stock:
            records['stock_trade'].extend(attach_outcome(r,trade,True) for r in events)
            records['stock_mark'].extend(attach_outcome(dict(r,entry_index=bisect.bisect_left(signal['time'],r['entry_time'])),signal,False) for r in events)
        else:records['crypto'].extend(attach_outcome(r,trade,True) for r in events)
        print('trap_candidates',symbol,d['origin'],'new',sum(r['family'] in ROUTES['new_union'] for r in events),
            'control',sum(r['family'] in ROUTES['phase_control'] for r in events),flush=True)
    frames={k:frame_of(v) for k,v in records.items()}
    for k,f in frames.items():
        if f.duplicated(['symbol','entry_time','family']).any():raise ValueError('duplicate_trap_candidate')
        path=ROOT/f'liquidity_trap_joint_nov19_{k}.pkl';f.to_pickle(path);sources[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
    selection={};eligible=[];start=stamp('2026-08-01');select_end=stamp('2026-10-01')
    for universe in ('base','expanded_current_snapshot'):
        for cr in ROUTES:
            for sk in ('stock_trade','stock_mark'):
                for sr in ROUTES:
                    key='|'.join((universe,cr,sk,sr));cf=select_rows(frames['crypto'],cr,universe,start,select_end)
                    sf=select_rows(frames[sk],sr,universe,start,select_end)
                    unknown=int((~cf.known.astype(bool)).sum()+(~sf.known.astype(bool)).sum())
                    if unknown:selection[key]=dict(status='UNSCORABLE_UNKNOWN_PATHS',unknown=unknown);continue
                    c=portfolio(cf,'crypto');s=stock_ready(portfolio(sf,'stocks'));results={};cases=[]
                    for policy in TP1_POLICIES:
                        results[policy]={}
                        for priority in ('crypto','stocks'):
                            value=verified_cash(c,s,specs,policy=policy,priority=priority,start=start,end_month='2026-09')
                            results[policy][priority]=value
                            for month,item in value['calendar_months'].items():
                                for bot in ('crypto','stocks'):
                                    cases.append(dict(month=month,bot=bot,contribution=item['return_contribution_pct'][bot],
                                        count=item[bot+'_closed'],score=item['return_contribution_pct'][bot]-item['cash_drawdown_pct']))
                    valid=all(case['contribution']>0 and case['count']>=20 for case in cases)
                    score=min(case['score'] for case in cases)
                    selection[key]=dict(status='RESOLVED_CASH',qualified=valid,score=score,cash=results)
                    if valid:eligible.append((score,key))
    chosen=max(eligible)[1] if eligible else None;later=None
    if chosen:
        universe,cr,sk,sr=chosen.split('|');cf=select_rows(frames['crypto'],cr,universe,select_end)
        sf=select_rows(frames[sk],sr,universe,select_end)
        unknown=int((~cf.known.astype(bool)).sum()+(~sf.known.astype(bool)).sum())
        if unknown:later=dict(status='UNSCORABLE_UNKNOWN_PATHS',unknown=unknown)
        else:
            c=portfolio(cf,'crypto');s=stock_ready(portfolio(sf,'stocks'))
            later=dict(status='RESOLVED_CASH',cash={p:{q:verified_cash(c,s,specs,policy=p,priority=q,start=select_end)
                for q in ('crypto','stocks')} for p in TP1_POLICIES})
    summary={k:dict(total=len(f),known=int(f.known.astype(bool).sum()),origins=f.origin.value_counts().to_dict(),
        families=f.family.value_counts().to_dict(),unknown_reasons=f.loc[~f.known.astype(bool),'unknown_reason'].value_counts().to_dict()
        if 'unknown_reason' in f else {}) for k,f in frames.items()}
    report=dict(status='TICK_ALIGNED_TRAP_PULLBACK_JOINT_RESEARCH',objective_verified=False,live_activation_supported=False,
        protocol=protocol,signals=summary,coverage=coverage,diagnostics=diagnostics,selection=selection,
        chosen=chosen,later=later,sources=sources,limitations=[
            'Current snapshot/turnover universe is hindsight for older months; not a historical universe or independent temporal holdout.',
            'New rule parameters frozen before expanded outcomes; all old market eras were already inspected.',
            'Actual current matched contracts only; late listings truncate history/warmup, no surrogate prior contract data.',
            'Production tick rounding and current lot specs assumed historically; entry/exit price stresses are scenarios, not actual quotes.',
            'Stock mark-trigger differs from current last-trigger orders; no live order contract or client setting changes.',
            'Fresh stock20 PAPER gate uses nominal50% price/R; private cash separately rounds requested5 or half/minimum.',
            'No full floating-equity guard/minute recovery,funding,fees,publication latency,liquidation or private fills reconstructed.',
            'NY weekday clock is not holiday/early-close calendar; November and new-listing history partial.',
            'Multiple predeclared route comparisons are exploratory; no promise of30% each month.'])
    (ROOT/'liquidity_trap_joint_nov19.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    print('trap_signals',summary,'qualified',len(eligible),'chosen',chosen,flush=True)


if __name__=='__main__':main()
