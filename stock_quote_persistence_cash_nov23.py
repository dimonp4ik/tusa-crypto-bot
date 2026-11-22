"""Audited stock trade supplements chosen with strictly prior quote evidence."""

from collections import Counter
from functools import lru_cache
import json

from okx_post_close_cash_nov23 import trade_difference
from okx_selective_single_target_nov23 import cash_summary, key
from okx_unbankable_tp1_nov21 import recent_inputs
from okx_xperp_cash_tick_nov19 import CAPITAL, CLOSE_LAGS, FRACTION, LEVERAGE, simulate
from okx_xperp_l2_entry_replay_nov19 import attempt
from okx_xperp_multiday_replay_nov20 import exit_books, file_hash
from post_close_replay_nov23 import PostCloseRule
from stock_liquid_quote_routes_nov21 import (ARCHIVES, DATA as ORIGINAL, ROOT, limit_bursts,
    load_replay, stressed_exit_penalty)
from stock_liquidity_first_phases_nov22 import quote_at
from stock_quote_continuation_holdout_nov21 import DATA as FOLLOWUP


PREFIX = 'stock_quote_persistence_cash_nov23'
PAST_BUCKETS = 4
REQUIRED_PASSES = 3
POLICIES = {
    'native_control': (None,False,False),
    'relaxed_reversal': ('reversal_z1_25',False,False),
    'relaxed_reversal_with_prior_liquidity': ('reversal_z1_25',True,False),
    'continuation': ('aligned_continuation',False,False),
    'continuation_with_prior_liquidity': ('aligned_continuation',True,False),
    'continuation_regular_hours': ('aligned_continuation',False,True),
    'continuation_with_prior_liquidity_regular_hours': ('aligned_continuation',True,True),
}
EXACT_POLICIES = ('native_control','continuation_regular_hours','continuation_with_prior_liquidity_regular_hours')


@lru_cache(maxsize=16)
def snapshot(folder,profile,bucket):
    path = folder/f'bucket_{bucket}.json'
    if not path.exists():
        return None
    try:
        raw = path.read_bytes()
        scan = json.loads(raw)
    except (OSError,ValueError):
        return None
    if scan['profile']!=profile or scan['bucket']!=bucket:
        raise ValueError('observer profile or bucket changed')
    summary=scan['summary']
    if not (scan['completed_at'] and summary.get('history_complete') and summary.get('protocol_ok')):
        return None
    rows={row['symbol']:row for row in scan['observations']}
    if len(rows)!=summary['expected'] or len(rows)!=len(scan['observations']):
        raise ValueError('observer window has no complete universe')
    return scan,rows,file_hash(path)


def evidence_for(row,folder,profile,digests):
    symbol,direction,current=row['symbol'],row['direction'],row['epoch']
    first=snapshot(folder,profile,current)
    if first is None:
        raise ValueError('the recorded trade has no matching observer bundle')
    observed,_,digest=first
    digests[str(current)]=digest
    decision=float(observed['summary']['decision_at'])
    if decision*1000>row['entry_ms']:
        raise ValueError('entry book recorded before the decision')
    # Use the original candidates known before any books from this scan arrive.
    native_present=any(item['payload']['native_candidate'] for item in observed['observations'])
    checks=[]
    for minutes in range(PAST_BUCKETS,0,-1):
        previous=current-minutes*900
        payload=snapshot(folder,profile,previous)
        if payload is None or payload[0]['completed_at']>=decision:
            checks.append('unavailable_before_decision')
            continue
        history,observations,digest=payload
        digests[str(previous)]=digest
        item=observations.get(symbol)
        book=quote_at(previous,item) if item else None
        if book is None or not item['received_at']<decision:
            checks.append('missing_valid_prior_quote')
            continue
        result=attempt(book,item['received_at']*1000,direction,item['payload']['spec'])
        checks.append(result['status'])
    return dict(native_present=native_present,checks=checks,passes=checks.count('eligible'))


def proposals(rows,rules,known):
    route,require_liquidity,regular_only=rules
    baseline=[row for row in rows if row['route']=='native']
    if route is None:
        return baseline,[]
    first={}
    for row in sorted(rows,key=lambda r:(r['entry_ms'],r['symbol'])):
        if row['route']!=route:
            continue
        prior=known[key(row)]
        if prior['native_present'] or require_liquidity and prior['passes']<REQUIRED_PASSES:
            continue
        if regular_only and row['phase']!='regular':
            continue
        first.setdefault(row['epoch'],row)
    return baseline+list(first.values()),list(first.values())


def summarize_prior_passes(plan_rows,known):
    result={}
    for route in ('native','reversal_z1_25','aligned_continuation'):
        counts=Counter()
        for row in plan_rows:
            if row['route']==route:
                flags=known[key(row)]
                counts[f"{row['phase']}:{flags['passes']}"]+=1
        result[route]=counts
    return result


def run_exact(pending,capture,specs,ticks,books):
    rows=[row for policies in pending.values() for chosen,_ in policies.values() for row in chosen
          if row['entry_status']=='eligible']
    downloads,checks=exit_books(rows,capture,specs,books)
    out={}
    for name,policies in pending.items():
        out[name]={}
        for lag in CLOSE_LAGS:
            outcomes={}
            for label,(chosen,extra) in policies.items():
                options={'combined':chosen}
                if label!='native_control':
                    options['supplement_only']=[row for row in chosen if key(row) in extra]
                outcomes[label]={mode:cash_summary(simulate(rows,specs,ticks,books,clock=60,lag=lag,
                                 setting='requested5',max_positions=12),extra) for mode,rows in options.items()}
            out[name][str(lag)]=outcomes
            print('exact',name,lag,{k:{mode:(r.get('cash_pnl_usd'),r.get('reason'))
                                     for mode,r in v.items()} for k,v in outcomes.items()},flush=True)
    return dict(cash=out,book_checks=checks,archive_sha256={k:v['sha256'] for k,v in downloads.items()})


def joint_original(exact_inputs,specs,ticks,books):
    from okx_cross_bot_overlap_nov21 import START
    from okx_joint_exact_exit_nov21 import ENTRY_END

    original,common_specs,common_ticks,common_books,_,_,_=recent_inputs()
    crypto=[row for row in original if row['bot']=='crypto']
    for symbol, observed in books.items():
        if symbol in common_specs and common_specs[symbol]['instId']!=specs[symbol]['instId']:
            raise ValueError('stock instrument differs from the shared replay')
        for stamp,book in observed.items():
            # Stock rows here are tied to their own observed quote session.
            common_books[symbol][stamp]=book
    outcomes={}
    for cap in (6,12):
        for lag in CLOSE_LAGS:
            parameters=dict(clock=60,lag=lag,setting='requested5',max_positions=cap)
            cases={}
            for label in EXACT_POLICIES:
                stocks=[]
                extra=set()
                for policies in exact_inputs.values():
                    chosen,additions=policies[label]
                    stocks.extend(row for row in chosen if START<=row['epoch']<ENTRY_END)
                    extra.update(additions)
                replay=simulate(crypto+stocks,common_specs|specs,common_ticks|ticks,common_books,**parameters)
                reported=cash_summary(replay,extra)
                reported['stock_proposals']=len(stocks)
                if label=='native_control':
                    anchor=replay
                elif 'closed' in replay and 'closed' in anchor:
                    reported['compared_with_native']=trade_difference(anchor,replay)
                cases[label]=reported
            outcomes[f'cap{cap}_{lag}ms']=cases
            print('joint',cap,lag,{k:(v.get('cash_pnl_usd'),v.get('reason'),v.get('stock_proposals'))
                                  for k,v in cases.items()},flush=True)
    return dict(start=START,stop=ENTRY_END,cases=outcomes,
                limitation='Only the simultaneous window where the verified crypto candidate book is complete.')


def run_cohorts(data):
    plan,capture,_,specs,ticks,books,tape,tapes,_=load_replay(data,keep_tapes=True)
    observations_by_name={}
    cash_by_name={}
    input_digests={}
    exact_inputs={}
    for name,rows in plan['cases'].items():
        archive='contract_followup' if name=='unseen_contract_observer' else name
        folder,profile=ARCHIVES[archive]
        digests={}
        known={key(r):evidence_for(r,folder,profile,digests) for r in rows}
        input_digests[name]=digests
        observations_by_name[name]=summarize_prior_passes(rows,known)
        exact_inputs[name]={}
        for policy in EXACT_POLICIES:
            selected,added=proposals(rows,POLICIES[policy],known)
            exact_inputs[name][policy]=(list(limit_bursts(selected,max_same_side=2)),{key(r) for r in added})
        values={}
        for bps in (5,15,30,50):
            model=dict(clock=60,lag=200,setting='requested5',max_positions=12,
                       exit_price=stressed_exit_penalty(tapes,specs,bps))
            for policy,rules in POLICIES.items():
                use,extra=proposals(rows,rules,known)
                selected=list(limit_bursts(use,max_same_side=2))
                selected_keys={key(r) for r in extra}
                regular=simulate(selected,specs,ticks,books,**model)
                with_pause=simulate(selected,specs,ticks,books,**model,
                    post_close_rule=PostCloseRule(direction_loss_seconds=3600,direction_loss_bot='stocks'))
                for method,replay in (('original_cadence',regular),('stock_loss_pause_60m',with_pause)):
                    view=cash_summary(replay,selected_keys)
                    if 'closed' in replay:
                        view['status']='SCORED_PUBLIC_TRADE_EXIT_STRESS'
                        view['supplement_cash_usd']=sum(t['bank_pnl']+t['runner_pnl']
                            for t in replay['closed'] if key(t) in selected_keys)
                        view['supplement_exits']=Counter(t['code'] for t in replay['closed'] if key(t) in selected_keys)
                        if method!='original_cadence' and 'closed' in regular:
                            view['reallocated']=trade_difference(regular,replay)
                    values.setdefault(str(bps),{}).setdefault(policy,{})[method]=dict(view,
                        source_native=sum(row['route']=='native' for row in use),supplement_candidates=len(extra),
                        symbols=Counter(r['symbol'] for r in extra))
            print(name,bps,{k:(v['original_cadence'].get('trades'),v['original_cadence'].get('cash_pnl_usd'),
                              v['original_cadence'].get('reason'),v['original_cadence']['supplement_candidates'])
                            for k,v in values[str(bps)].items()},flush=True)
        cash_by_name[name]=values
    exact=run_exact(exact_inputs,capture,specs,ticks,books)
    result=dict(cash=cash_by_name,exact_books=exact,quote_history=observations_by_name,
                observer_sha256=input_digests,plan_sha256=capture['plan_sha256'],
                capture_sha256=file_hash(data/'capture.json'),tape=tape)
    if data==ORIGINAL:
        result['shared_with_crypto']=joint_original(exact_inputs,specs,ticks,books)
    return result


def main():
    trial = run_cohorts(ORIGINAL)
    later = run_cohorts(FOLLOWUP)
    report=dict(initial_cash_usd=CAPITAL,margin_fraction=FRACTION,leverage=LEVERAGE,policies=POLICIES,
                required_passing_quotes=REQUIRED_PASSES,prior_buckets=PAST_BUCKETS,
                discovery=trial,holdout=later,live_activation_supported=False,limitations=[
                    'An additional stock setup is allowed only in a scan with no original native candidate, as established before the current book quotes.',
                    'No current or future book is used for the history filter. The first eligible candidate in receipt order takes the one supplement slot.',
                    'All candidate entries retain the original quote and 5 bp cap. Both stock groups use experimental frozen geometry.',
                    'Observed entry books with trade-print exit stress, not private fills. This is a stock-only comparison with the same total 12-slot cap.',
                    'Exact book cases are included separately. A missing exit archive makes that case unscorable.',
                    'Older and later samples have different archived volume series. They are compared separately and do not establish monthly performance.',
                ])
    (ROOT/f'{PREFIX}.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8',newline='\n')


if __name__=='__main__':
    main()
