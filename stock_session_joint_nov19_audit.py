"""Declared stock-session scenarios with a fixed crypto common-account reference."""
from datetime import datetime,timezone
from collections import Counter
from pathlib import Path
import hashlib
import json
import math
import pickle
import bisect
import argparse
import pandas as pd

from stock_session_entries_nov19 import SessionContext,candidates,FAMILIES,HORIZON
from us_cash_session_nov19 import SOURCES as CALENDAR_SOURCES
from liquidity_trap_joint_nov19_audit import with_exact_tick,attach_outcome,load_frame
from phase_scenarios_nov19_audit import (ROOT,STOCK_ROOT,STOCK_MANIFEST,CRYPTO_MANIFEST,SPEC_PATH,
    frame_of,mature_before,stock_portfolio_function,stock_ready,cash,TP1_POLICIES)
from crypto_native_price_pool_nov19 import load_pool,META
from crypto_minimum_bank_runner_nov19_audit import checked_row,admitted_records
from crypto_active_entry_runner_nov19_audit import activity_gate
from account_cash_guard_nov17 import stamp

PREFIX='stock_session_joint_nov19'
EXPANSION=ROOT/'native_expansion_nov19'
SNAPSHOT=ROOT/'native_universe_nov19_snapshot.json'
ROUTES={family:{family} for family in FAMILIES}|{'session_union':set(FAMILIES)}


def route(frame,name,universe,start,end=None):
    f=frame[frame.family.isin(ROUTES[name])&frame.entry_time.ge(start)]
    if universe=='base':f=f[f.origin.eq('base')]
    if end is not None:f=f[mature_before(f,end)]
    return f.copy()


def paper_portfolio(frame):
    rows=[]
    for row in frame.to_dict('records'):
        sign=1 if row['direction']=='LONG' else -1
        exposure=sign*(-1 if row['symbol']=='SOXSUSDT' else 1)
        rows.append(dict(row,module=row['family'],source_priority=3,module_priority=0,
            signed_price_return=row['signed_return'],outcome='SL' if row['sequential_outcome']=='SL' else 'TP',
            market_exposure=exposure))
    accepted=stock_portfolio_function()(rows,None,rolling_hour_cap=4,rolling_direction_hour_cap=2,
        rank_key=lambda r:-r['signal_strength'],loss_streak_cap=2,direction_cap=4,daily_loss_r_cap=2.,
        exposure_key=lambda r:r['market_exposure'],exposure_cap=4)
    return frame_of(accepted)


def cash_cases(crypto,stock,specs,start):
    output={}
    for policy in TP1_POLICIES:
        output[policy]={}
        for priority in ('crypto','stocks'):
            v=cash(crypto,stock,specs,policy=policy,priority=priority,start=start)
            if any(v['summary']['peak_by_bot'].get(bot,0)>cap for bot,cap in (('crypto',4),('stocks',8))):
                raise ValueError('session_local_slot_violation')
            if abs(sum(v['summary']['return_contribution_pct'].values())-v['summary']['return_pct'])>1e-4:
                raise ValueError('session_cash_contribution_mismatch')
            output[policy][priority]=v
    return output


def qualifying(cases,bot):
    rows=[]
    for p in cases.values():
        for v in p.values():
            for month in ('2026-08','2026-09'):
                item=v['calendar_months'].get(month,{})
                rows.append((item.get('return_contribution_pct',{}).get(bot,0),
                    item.get(bot+'_closed',0),item.get('cash_drawdown_pct',0)))
    return all(p>0 and n>=20 for p,n,_ in rows),min(p-dd for p,_,dd in rows)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--declare-only',action='store_true')
    args=parser.parse_args()
    paths=[Path(__file__),Path('stock_session_entries_nov19.py'),Path('us_cash_session_nov19.py'),
        Path('liquidity_trap_joint_nov19_audit.py'),Path('phase_scenarios_nov19_audit.py'),
        Path('price_grid_bracket_nov19.py'),Path('price_bracket_path_nov19.py'),
        Path('crypto_native_price_pool_nov19.py'),Path('crypto_minimum_bank_runner_nov19_audit.py'),
        Path('crypto_active_entry_runner_nov19_audit.py'),Path('banked_runner_path_nov19.py'),
        Path('account_cash_guard_nov17_audit.py'),Path('account_cash_guard_nov17.py'),
        Path('cash_tp1_policy_nov19.py'),Path('src/tp1_plan.py'),Path('src/okx_trader.py'),
        STOCK_ROOT/'stock_module_cluster_audit.py',STOCK_MANIFEST,CRYPTO_MANIFEST,SPEC_PATH,META,SNAPSHOT,
        EXPANSION/'selection.json',EXPANSION/'manifest.json']
    sources={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    protocol=dict(declared_at=datetime.now(timezone.utc).isoformat(),sources=dict(sources),
        routes={k:sorted(v) for k,v in ROUTES.items()},calendar_sources=list(CALENDAR_SOURCES),
        calendar_year=2026,session='NY09:30 regular open; official holidays and13:00 early closes',
        entry_clock='after opening30min; before session open+5h and close-30min',
        opening_range_bars=2,range_width_fraction=[.002,.04],session_active_share=.75,
        atr='last14 active regular-session true ranges, including intersession gaps',
        vwap='native trade typical15m-bar price weighted by native contract volume, not underlying consolidated VWAP',
        opening_breakout_min_body_atr=.3,breakout_volume_min=1.2,retest_wait_bars=4,
        retest_body_atr=.3,retest_close_location=.6,retest_volume_min=.75,
        failure_body_atr=.4,failure_return_atr=.1,failure_volume_min=1.2,
        acceptance_wait_bars=2,acceptance_return_atr=.1,acceptance_close_location=.6,
        vwap_reclaim_body_atr=.3,vwap_reclaim_return_atr=.1,vwap_extension_atr=[.1,1.5],
        macro='QQQ andSPY same-session trade-return/weighted-price context, omit self, reverse SOXS exposure',
        stop='structure+0.15ATR, floor0.3%, capclip(1.5ATR,0.8%,1.5%)',
        target1='clip(1ATR,0.5%,1%)',target2='clip(2.5ATR,1%,2%), cap at known opposing range/prior-session levels',
        minimum_final_move=.008,minimum_rr=1.5,tick_grid='actual frozen exact contract; production rounding',
        market_price_stress=.0005,stock_horizon_bars=HORIZON,stock_time_exit='underlying cash session close',
        triggers=['trade','mark'],universes=['base','expanded_current_snapshot'],
        client_tp1_policies=TP1_POLICIES,margin_fraction=.03,leverage=10,capital=130,
        crypto_reference='frozen crypto proposals with pre-entry12/16 activity, checked BE-atTP1; fixed across stock routes',
        publication_rails='stock2/scan,4/hour,2/direction/hour,4/nominal-direction,4/economic-direction,3h cooldown,2losses,daily2R',
        stock_cash_gate='last20 closed nominal50% paper outcomes positive in both R and price return',
        selection_months=['2026-08','2026-09'],selection_minimum_each_bot_month=20,
        choose_stock='positive stock monthly contribution and>=20stock closes across both TP1 modes and priorities; trade trigger only',
        joint_requirement='same positive/count test for both bots; this is a research prerequisite, not the30% objective',
        later='only selected stock case; fresh October paper/cash account, no earlier cash seed',
        previously_inspected_history=True,objective_verified=False,live_activation_supported=False)
    declaration=ROOT/f'{PREFIX}_protocol.json'
    if declaration.exists():
        previous=json.loads(declaration.read_text(encoding='utf-8'))
        if previous['sources']!=sources:raise ValueError('session_declared_sources_changed')
        protocol=previous
    else:declaration.write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    if args.declare_only:
        print('session_protocol_frozen',protocol['declared_at'],flush=True);return
    instruments=json.loads(SNAPSHOT.read_text(encoding='utf-8'))['instruments']
    specs=json.loads(SPEC_PATH.read_text(encoding='utf-8'));datasets={}
    for symbol,item in json.loads(STOCK_MANIFEST.read_text(encoding='utf-8'))['symbols'].items():
        datasets[symbol]=dict(mark=load_frame(STOCK_ROOT/item['path'],item['sha256']),
            trade=load_frame(STOCK_ROOT/item['trade_path'],item['trade_sha256']),origin='base')
        specs[symbol]=with_exact_tick(item['spec'],instruments)
    expansion=json.loads((EXPANSION/'manifest.json').read_text(encoding='utf-8'))
    if not expansion['complete']:raise ValueError('session_expansion_not_complete')
    for symbol,item in expansion['symbols'].items():
        if item['category']!='3':continue
        if symbol in datasets:raise ValueError('duplicate_session_asset')
        datasets[symbol]=dict(mark=load_frame(Path(item['mark']['path']),item['mark']['sha256']),
            trade=load_frame(Path(item['trade']['path']),item['trade']['sha256']),origin='expansion')
        specs[symbol]=with_exact_tick(item['spec'],instruments)
    contexts={s:SessionContext(d['mark'],d['trade']) for s,d in datasets.items()}
    end=min(min(d['mark']['time'][-1],d['trade']['time'][-1])+900 for d in datasets.values())
    cutoff=end-HORIZON*900;records=dict(trade=[],mark=[]);diagnostics={}
    for symbol,context in sorted(contexts.items()):
        diag={};refs=[contexts[s] for s in ('QQQUSDT','SPYUSDT') if s!=symbol]
        events=candidates(context,symbol,specs[symbol],references=refs,diagnostics=diag)
        events=[dict(r,origin=datasets[symbol]['origin']) for r in events if r['entry_time']<=cutoff]
        diagnostics[symbol]=diag
        for r in events:
            records['trade'].append(attach_outcome(r,context.trade,True))
            records['mark'].append(attach_outcome(dict(r,entry_index=bisect.bisect_left(context.mark['time'],r['entry_time'])),context.mark,False))
        print('session_candidates',symbol,len(events),dict(Counter(r['family'] for r in events)),flush=True)
    frames={}
    for kind,rows in records.items():
        f=frame_of(rows)
        if 'origin' not in f:f['origin']=pd.Series(dtype=str)
        if f.duplicated(['symbol','entry_time']).any():raise ValueError('duplicate_session_entry')
        path=ROOT/f'{PREFIX}_{kind}.pkl';f.to_pickle(path);sources[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
        frames[kind]=f
    pool,_=load_pool();crypto=[]
    manifest=json.loads(CRYPTO_MANIFEST.read_text(encoding='utf-8'))
    for symbol,g in pool[pool.is_frozen_candidate].groupby('symbol'):
        item=manifest['symbols'][symbol];c=load_frame(Path(item['path']),item['sha256'])
        lookup={int(t):i for i,t in enumerate(c['time'])}
        for row in g.to_dict('records'):
            i=lookup[int(row['entry_time'])]
            if activity_gate(c,i):crypto.append(checked_row(row,c,i,'be_tp1',.0005))
    admitted,unknown_crypto=admitted_records(crypto)
    if unknown_crypto:raise ValueError('session_crypto_reference_admitted_unknown')
    cf=pd.DataFrame(admitted);cf['signed_return']=cf['risk_pct']*cf['net_r']
    selection={};stock_eligible=[];joint_eligible=[];start=stamp('2026-08-01');split=stamp('2026-10-01')
    for universe in ('base','expanded_current_snapshot'):
        for kind,f in frames.items():
            for name in ROUTES:
                key='|'.join((universe,kind,name));sf=route(f,name,universe,start,split)
                unknown=int((~sf.known.astype(bool)).sum())
                if unknown:selection[key]=dict(status='UNSCORABLE_UNKNOWN_PATHS',unknown=unknown);continue
                pf=paper_portfolio(sf);ready=stock_ready(pf)
                cc=cf[cf.entry_time.ge(start)&(cf.entry_time+48*900<split)&cf.exit_time.lt(split)]
                cases=cash_cases(cc,ready,specs,start)
                stock_ok,stock_score=qualifying(cases,'stocks');crypto_ok,crypto_score=qualifying(cases,'crypto')
                selection[key]=dict(status='RESOLVED_RESEARCH_CASH',proposals=len(sf),paper_signals=len(pf),
                    ready_entries=len(ready),stock_qualified=stock_ok,joint_qualified=stock_ok and crypto_ok,
                    stock_score=stock_score,crypto_score=crypto_score,cash=cases)
                if kind=='trade' and stock_ok:stock_eligible.append((stock_score,key))
                if kind=='trade' and stock_ok and crypto_ok:joint_eligible.append((min(stock_score,crypto_score),key))
    chosen=max(stock_eligible)[1] if stock_eligible else None;later=None
    if chosen:
        universe,kind,name=chosen.split('|');sf=route(frames[kind],name,universe,split)
        unknown=int((~sf.known.astype(bool)).sum())
        if unknown:later=dict(status='UNSCORABLE_UNKNOWN_PATHS',unknown=unknown)
        else:
            pf=paper_portfolio(sf);ready=stock_ready(pf)
            later=dict(status='RESOLVED_RESEARCH_CASH',paper_signals=len(pf),ready_entries=len(ready),
                cash=cash_cases(cf[cf.entry_time.ge(split)],ready,specs,split))
    report=dict(status='CALENDAR_AWARE_STOCK_SESSION_JOINT_RESEARCH',protocol=protocol,sources=sources,
        signal_counts={kind:dict(total=len(f),known=int(f.known.astype(bool).sum()),
            families=f.family.value_counts().to_dict(),unknown_reasons=f.loc[~f.known.astype(bool),'unknown_reason'].value_counts().to_dict()
            if 'unknown_reason' in f else {}) for kind,f in frames.items()},
        diagnostics=diagnostics,selection=selection,chosen_stock=chosen,joint_qualified=len(joint_eligible),later=later,
        objective_verified=False,live_activation_supported=False,limitations=[
            'Previously inspected history/current-snapshot universe; no independent holdout or historical investable universe.',
            'Native volume-weighted typical15m price is not real consolidated stock VWAP or executable book.',
            'Regular cash calendar is a strategy context; X-Perp availability and holidays differ.',
            'SOXS inverse sign handled; QQQ/SPY are imperfect proxies for sector/single-stock and leverage risk.',
            'Actual current contracts/ticks/lots assumed historically; macro quote freshness within bar and bid/ask not reconstructed.',
            'Unknown raw stock path makes that selection case unscorable; never dropped or assigned zero PnL.',
            'Paper readiness nominal50% is proposed signal health; private5% or minimum50% cash uses real lot rounding.',
            'Cash starts fresh August and laterOctober; no floating-equity guard,funding,fees,liquidation or private fills.',
            'Crypto activity/checked-BE reference is research, not a newly deployed strategy; stock LIVE restriction unchanged.',
            'November partial; positive/count gate is a prerequisite, not proof of30% each bot each month.'])
    (ROOT/f'{PREFIX}.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    print('session_results',report['signal_counts'],'stock_eligible',len(stock_eligible),'joint_eligible',len(joint_eligible),'chosen',chosen,flush=True)


if __name__=='__main__':main()
