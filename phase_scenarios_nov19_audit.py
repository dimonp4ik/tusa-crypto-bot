"""Two closed market scenarios on both native venues, with joint cash selection."""
from datetime import datetime,timezone
import hashlib
import json
import math
import ast
import collections
import heapq
from functools import lru_cache
from typing import Callable
from pathlib import Path
import pickle
import argparse
import pandas as pd

from phase_entry_scenarios_nov19 import candidates,FAMILIES,HORIZON
from price_bracket_path_nov19 import resolve
from account_cash_guard_nov17 import calendar_months,stamp
from account_cash_guard_nov17_audit import runtime_gate
from cash_tp1_policy_nov19 import simulate

ROOT=Path('reports/audit_2026_09_08')
STOCK_ROOT=Path('../tusa stocks')
CRYPTO_MANIFEST=ROOT/'xperp_15m_nov14_merged/manifest.json'
STOCK_MANIFEST=STOCK_ROOT/'reports/audit_2026_09_08/stock_mark_matched_nov18/manifest.json'
SPEC_PATH=ROOT/'shared_xperp_specs_nov14.json'
ROUTES=('range_only','trend_only','phase_union')
TP1_POLICIES=dict(requested5=dict(tp1_pct=5,minimum_close=False),half_minimum=dict(tp1_pct=50,minimum_close=True))
BASE_COLUMNS=['symbol','entry_time','exit_time','direction','family','signal_strength','known','risk_pct',
    'net_r','signed_return','sequential_outcome','sequential_banked','near_exit_time',
    'runner_price_return','bank_price_return','entry','sl','tp1','tp2']


def frame_of(rows):return pd.DataFrame(rows) if rows else pd.DataFrame(columns=BASE_COLUMNS)


def route(frame,name):
    if name=='phase_union':return frame.copy()
    family=FAMILIES[0] if name=='range_only' else FAMILIES[1]
    return frame[frame.family.eq(family)].copy()


def mature_before(frame,cutoff):return (frame.entry_time+HORIZON*900<cutoff)&frame.exit_time.lt(cutoff)


@lru_cache(maxsize=1)
def stock_portfolio_function():
    path=STOCK_ROOT/'stock_module_cluster_audit.py'
    tree=ast.parse(path.read_text(encoding='utf-8'))
    node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='portfolio_gate')
    namespace=dict(math=math,collections=collections,heapq=heapq,Callable=Callable)
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),namespace)
    return namespace['portfolio_gate']


def portfolio(frame,bot):
    # Proposed signal-health model uses nominal50% banking, independently of
    # a client's later lot-rounded choice. This is not the deployed profile.
    rows=[dict(r,module=r['family'],source_priority=-r['signal_strength'],module_priority=0,
        signed_price_return=r['signed_return'],outcome='SL' if r['sequential_outcome']=='SL' else 'TP')
        for r in frame.to_dict('records')]
    if bot=='crypto':accepted=runtime_gate(rows,latch_loss_pause=False)
    else:
        accepted=stock_portfolio_function()(rows,None,rolling_hour_cap=4,rolling_direction_hour_cap=2,
            rank_key=lambda r:-r['signal_strength'],loss_streak_cap=2,direction_cap=4,daily_loss_r_cap=2.)
    return frame_of(accepted)


def stock_ready(frame):
    closed=frame.sort_values(['exit_time','entry_time','symbol']).to_dict('records');out=[]
    from bisect import bisect_left
    times=[r['exit_time'] for r in closed]
    for row in frame.to_dict('records'):
        stop=bisect_left(times,row['entry_time']);history=closed[max(0,stop-20):stop]
        if len(history)==20 and math.fsum(r['net_r'] for r in history)>1e-10 and math.fsum(r['signed_return'] for r in history)>1e-12:
            out.append(row)
    return frame_of(out)


def cash_rows(frame,bot):
    rows=[]
    for row in frame.to_dict('records'):
        unit=2*row['bank_price_return']
        if not unit>0:raise ValueError('nonpositive_first_target_cash_unit')
        # The cash engine's .5R bank is normalized to this row's actual fixed
        # TP1 price return. Actual stop distance stays in actual_risk_fraction.
        rows.append(dict(row,bot=bot,actual_risk_fraction=row['risk_pct'],risk_pct=unit,
            sequential_runner_r=row['runner_price_return']/unit))
    return rows


def cash(crypto,stock,specs,*,policy,priority,start):
    rows=cash_rows(crypto,'crypto')+cash_rows(stock,'stocks')
    value=simulate(rows,specs,capital=130.,slots=12,bot_slots=dict(crypto=4,stocks=8),
        simultaneous_writers=2,release_delay=900,tie_key=lambda r:(r['bot']!=priority,r['symbol']),**TP1_POLICIES[policy])
    if value['peak_positions']>12 or value['peak_reserved_margin_pct']>90+1e-9:
        raise ValueError('phase_shared_capacity_violation')
    months={m:v for m,v in calendar_months(value).items() if stamp(m+'-01')>=start}
    for month,item in months.items():
        begin,end=stamp(month+'-01'),stamp(f'2026-{int(month[-2:])+1:02}-01')
        for bot in ('crypto','stocks'):
            closed=[r for r in value['ledger'] if r['bot']==bot and begin<=r['exit_time']<end]
            pnl=[r['bank_pnl']+r['runner_pnl'] for r in closed]
            item[bot+'_closed']=len(closed)
            item[bot+'_winrate_pct']=100*sum(v>1e-12 for v in pnl)/len(pnl) if pnl else None
            item[bot+'_tp1_unbanked']=sum(r['tp1_hit'] and r['planned_partial']==0 for r in closed)
    return dict(summary={k:v for k,v in value.items() if k not in ('ledger','cash_events','guard_rejections')},calendar_months=months)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile',choices=('long_context','intraday_context'),default='intraday_context')
    args=parser.parse_args();intraday=args.profile=='intraday_context'
    prefix='phase_scenarios_intraday_v2b_nov19' if intraday else 'phase_scenarios_long_context_v2b_nov19'
    source_paths=(Path(__file__),Path('phase_entry_scenarios_nov19.py'),Path('price_bracket_path_nov19.py'),
        Path('cash_tp1_policy_nov19.py'),Path('src/tp1_plan.py'),Path('account_cash_guard_nov17_audit.py'),
        STOCK_ROOT/'stock_module_cluster_audit.py',CRYPTO_MANIFEST,STOCK_MANIFEST,SPEC_PATH,
        ROOT/'phase_scenarios_nov19_protocol.json',ROOT/'phase_scenarios_nov19_v1b_protocol.json',
        ROOT/'phase_scenarios_intraday_nov19_protocol.json')
    sources={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths}
    protocol=dict(declared_at=datetime.now(timezone.utc).isoformat(),sources=dict(sources),
        implementation_revision='v2b: session TIME diagnostic bar is last included candle, not first excluded one; trading rules unchanged from v2',
        profile=args.profile,
        routes=ROUTES,stock_triggers=['trade','mark'],crypto_trigger='trade',
        target1_rule='clip(1ATR/entry,0.005,0.01)' if intraday else 'fixed0.006',
        target2_rule='clip(2.5ATR/entry,0.01,0.02),cap at unbroken4h/opposite16h level' if intraday else 'fixed0.018,cap at1h/all pivots and opposite5h level',
        final_target_minimum=.008 if intraday else .01,
        stop_rule='floor0.003,max distance clip(1.5ATR/entry,0.008,0.015)' if intraday else 'range0.003to0.008',
        scenario_context='1h phase,4h no strong opposition' if intraday else '4h phase and1h alignment',
        trend_confirmation='closed retest after breakout' if intraday else 'retest then separate close extension',
        minimum_final_reward_risk=1.5,horizon_bars=HORIZON,entry_exit_price_stress=.0005,
        minimum_active_trade_bars_last16=12,setup_volume_ratio=1.2,confirmation_volume_ratio=.75,
        htf_bars=[4,16],htf_ema_spans=[20,50],htf_ema_window=64,
        stock_clock='NY weekday10:00-15:30 entries,16:00 time exit; no holiday/early-close calendar',
        selection_months=['2026-08','2026-09'],later_start='2026-10-01',
        minimum_cash_closed_each_bot_month=20,both_bots_must_profit_each_selection_month=True,
        stock_paper_gate='fresh20 completed signals,positive sum actual-R and price,nominal50percent bank',
        client_tp1_policies=TP1_POLICIES,unknown_paths_make_case_unscorable=True,
        objective_verified=False,live_activation_supported=False)
    declaration=ROOT/f'{prefix}_protocol.json'
    if declaration.exists():
        prior=json.loads(declaration.read_text(encoding='utf-8'))
        if prior['sources']!=sources:raise ValueError('phase_declared_sources_changed')
        protocol=prior
    else:declaration.write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    crypto={};stock={};specs=json.loads(SPEC_PATH.read_text(encoding='utf-8'))
    for symbol,item in json.loads(CRYPTO_MANIFEST.read_text(encoding='utf-8'))['symbols'].items():
        p=Path(item['path']);raw=p.read_bytes()
        if hashlib.sha256(raw).hexdigest()!=item['sha256'] or specs[symbol]['instId']!=item['instrument']:
            raise ValueError('phase_crypto_source_or_contract_changed')
        crypto[symbol]=pickle.loads(raw);sources[str(p)]=item['sha256']
    for symbol,item in json.loads(STOCK_MANIFEST.read_text(encoding='utf-8'))['symbols'].items():
        pair=[]
        for key,sha in (('path','sha256'),('trade_path','trade_sha256')):
            p=STOCK_ROOT/Path(item[key]);raw=p.read_bytes()
            if hashlib.sha256(raw).hexdigest()!=item[sha]:raise ValueError('phase_stock_source_changed')
            pair.append(pickle.loads(raw));sources[str(p)]=item[sha]
        if item['spec']['instId']!=item['instrument']:raise ValueError('phase_stock_contract_changed')
        specs[symbol]=item['spec'];stock[symbol]=pair
    end=min([c['time'][-1]+900 for c in crypto.values()]+[min(m['time'][-1],t['time'][-1])+900 for m,t in stock.values()])
    cutoff=end-HORIZON*900;rows=dict(crypto=[],stock_trade=[],stock_mark=[]);diagnostics={}
    def resolve_row(row,path,trade_volume):
        result=resolve(path,row['entry_index'],direction=row['direction'],entry=row['entry'],stop=row['sl'],
            tp1=row['tp1'],tp2=row['tp2'],expire_at=row['expire_at'],require_trade_volume=trade_volume)
        out=dict(row,known=result['known'])
        if not result['known']:return dict(out,unknown_reason=result['reason'],exit_time=row['entry_time']+HORIZON*900)
        price=.5*result['bank_price_return']+.5*result['runner_price_return'] if result['banked'] else result['runner_price_return']
        return dict(out,exit_time=result['exit_time'],runner_price_return=result['runner_price_return'],
            bank_price_return=result['bank_price_return'],signed_return=price,net_r=price/row['risk_pct'],
            sequential_outcome={'TARGET':'TP_RUNNER','TIME':'TP_EXPIRED' if result['banked'] else 'EXPIRED'}.get(result['outcome'],result['outcome']),
            sequential_banked=result['banked'],near_exit_time=result['near_exit_time'] if result['near_exit_time'] is not None else math.nan)
    for symbol,c in sorted(crypto.items()):
        diag={}
        events=[r for r in candidates(c,c,symbol,references=[crypto['BTCUSDT']],profile=args.profile,diagnostics=diag)
            if r['entry_time']<=cutoff]
        diagnostics['crypto:'+symbol]=diag
        rows['crypto'].extend(resolve_row(r,c,True) for r in events)
        print('phase_candidates','crypto',symbol,len(events),flush=True)
    for symbol,(mark,trade) in sorted(stock.items()):
        diag={}
        events=[r for r in candidates(mark,trade,symbol,references=[stock['QQQUSDT'][0],stock['SPYUSDT'][0]],stocks=True,
            profile=args.profile,diagnostics=diag)
            if r['entry_time']<=cutoff]
        diagnostics['stocks:'+symbol]=diag
        rows['stock_trade'].extend(resolve_row(r,trade,True) for r in events)
        for row in events:
            # Matched sources have identical timestamps, but never rely on an
            # accidental numerical row index when switching trigger series.
            import bisect
            aligned=dict(row,entry_index=bisect.bisect_left(mark['time'],row['entry_time']))
            rows['stock_mark'].append(resolve_row(aligned,mark,False))
        print('phase_candidates','stocks',symbol,len(events),flush=True)
    frames={key:frame_of(value) for key,value in rows.items()}
    for key,frame in frames.items():
        path=ROOT/f'{prefix}_{key}.pkl';frame.to_pickle(path)
        sources[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
    start=stamp('2026-08-01');select_end=stamp('2026-10-01');selection={};eligible=[]
    for crypto_route in ROUTES:
        for stock_kind in ('stock_trade','stock_mark'):
            for stock_route in ROUTES:
                key=f'{crypto_route}__{stock_kind}__{stock_route}'
                cf=route(frames['crypto'],crypto_route);sf=route(frames[stock_kind],stock_route)
                cc=cf[cf.entry_time.ge(start)&mature_before(cf,select_end)]
                sc=sf[sf.entry_time.ge(start)&mature_before(sf,select_end)]
                unknown=int((~cc.known.astype(bool)).sum()+(~sc.known.astype(bool)).sum())
                if unknown:
                    selection[key]=dict(status='UNSCORABLE_UNKNOWN_PATHS',unknown=unknown);continue
                c=portfolio(cc,'crypto');s=stock_ready(portfolio(sc,'stocks'));results={};cases=[]
                for policy in TP1_POLICIES:
                    results[policy]={}
                    for priority in ('crypto','stocks'):
                        value=cash(c,s,specs,policy=policy,priority=priority,start=start)
                        results[policy][priority]=value
                        for month in ('2026-08','2026-09'):
                            item=value['calendar_months'][month]
                            cases.extend((item['return_contribution_pct'][bot],item[bot+'_closed']) for bot in ('crypto','stocks'))
                valid=all(p>0 and count>=20 for p,count in cases)
                score=min(p for p,_ in cases)
                selection[key]=dict(status='RESOLVED_CASH',qualified=valid,score=score,cash=results)
                if valid:eligible.append((score,key))
    chosen=max(eligible)[1] if eligible else None;later_result=None
    if chosen:
        cr,sk,sr=chosen.split('__');cf=route(frames['crypto'],cr);sf=route(frames[sk],sr)
        cf=cf[cf.entry_time.ge(select_end)];sf=sf[sf.entry_time.ge(select_end)]
        unknown=int((~cf.known.astype(bool)).sum()+(~sf.known.astype(bool)).sum())
        if unknown:later_result=dict(status='UNSCORABLE_UNKNOWN_PATHS',unknown=unknown)
        else:
            c=portfolio(cf,'crypto');s=stock_ready(portfolio(sf,'stocks'))
            later_result=dict(status='RESOLVED_CASH',cash={p:{q:cash(c,s,specs,policy=p,priority=q,start=select_end)
                for q in ('crypto','stocks')} for p in TP1_POLICIES})
    report=dict(status='PHASE_SCENARIO_JOINT_RESEARCH',objective_verified=False,live_activation_supported=False,
        protocol=protocol,mature_last_entry=cutoff,diagnostics=diagnostics,signals={k:dict(total=len(f),known=int(f.known.astype(bool).sum()),
            families=f.family.value_counts().to_dict()) for k,f in frames.items()},selection=selection,chosen=chosen,
        later=later_result,sources=sources,limitations=[
            'All historical eras inspected before; fixed new hypotheses are not an independent holdout.',
            'Closed known references plus fixed5bp entry/exit stresses are scenarios, not actual bid/ask or private fills.',
            'Stock mark-trigger is a different hypothetical contract; current live orders use last-trigger.',
            'Fresh20 stock-paper signal gate uses nominal50% price/R; client cash separately rounds real requested TP1 volumes.',
            'Unknown/zero-volume/stale time exits make the affected case unscorable, never a free flat or discarded loser.',
            'Weekday NY clock is not an official holiday/early-close calendar; 4h bars include off-hours price references.',
            'No current full-equity guard/minute recovery,floating PnL,funding,fees,latency or liquidation reconstruction.',
            'Current matched specs assumed historically; start coverage/November partial; no margin change or profit promise.'])
    (ROOT/f'{prefix}.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    print('phase_signals',report['signals'],'qualified',len(eligible),'chosen',chosen,flush=True)


if __name__=='__main__':main()
