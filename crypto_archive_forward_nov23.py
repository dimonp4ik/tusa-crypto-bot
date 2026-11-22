"""Verify frozen crypto candidates in the next eight hours of public L2 data."""

from collections import Counter
from datetime import datetime, timezone
from unittest.mock import patch

import okx_cross_bot_overlap_nov21 as joint
from crypto_scan_priority_nov22 import POLICIES as RANKINGS
from crypto_scan_priority_nov22 import attribute, book_cadence, reconstruct_rank
from okx_crypto_risk_range_nov22 import attach_entry_quotes
from okx_delayed_trailing_cash_nov23 import paths_for
from okx_post_close_cash_nov23 import trade_difference
from okx_selective_single_target_nov23 import cash_summary, key
from okx_sweep_context_cash_nov23 import POLICIES as SWEEP_POLICIES
from okx_sweep_context_cash_nov23 import contexts, refuses
from okx_unbankable_tp1_nov21 import read_json
from okx_xperp_cash_tick_nov19 import CAPITAL, CLOSE_LAGS, FRACTION, LEVERAGE, simulate
from okx_xperp_multiday_capture_nov20 import HOLD
from okx_xperp_multiday_replay_nov20 import ROOT, exit_books, file_hash, resolve_all
from stock_liquid_quote_routes_nov21 import write_json


STEM='crypto_archive_forward_nov23'
DATA=ROOT/STEM
START=int(datetime(2026,11,20,4,tzinfo=timezone.utc).timestamp())
ENTRY_STOP=START+8*3600
BOOK_SCHEDULES={
    'observed_retry_sequence': (10_000,[0,200,800]),
    'same_time_11s': (11_000,[0]),
}
PROTECTION=('be_after_075r','trail_1r_distance_050r')


def load_candidates():
    manifest=DATA/'crypto_capture.json'
    with patch.multiple(joint,START=START,ENTRY_END=ENTRY_STOP,DATA=DATA):
        if not manifest.exists():
            joint.capture_crypto()
        captured=read_json(manifest)
        missing=[symbol for symbol,record in captured['trades'].items() if not record['complete']]
        if missing:
            raise ValueError(f'cash holdout has incomplete public trades for {missing}')
        reconstructed,annotations,checks=reconstruct_rank(captured)
    rows=[dict(r,**annotations[r['symbol'],r['epoch']]) for r in reconstructed]
    if len({key(dict(r,entry_ms=r['epoch']*1000)) for r in rows})!=len(rows):
        raise ValueError('the frozen crypto replay has duplicate instruments in a scan')
    if any(not START<=row['epoch']<ENTRY_STOP for row in rows):
        raise ValueError('frozen candidate outside the disjoint entry window')
    return captured,rows,checks,file_hash(manifest)


def material(captured,rows,schedule,available):
    offset,retries=schedule
    specs=captured['specs']
    ticks={symbol:float(spec['tickSz']) for symbol,spec in specs.items()}
    plan=dict(rows=rows,specs=specs,start=START,coverage_end=ENTRY_STOP+HOLD['crypto'],
              parameters=dict(crypto_offset_ms=offset,retry_offsets_ms=retries))
    proposals,books,entry_checks=attach_entry_quotes(rows,plan,available,ticks)
    trades,references=captured['trades'],captured['references']
    verification=resolve_all(proposals,{'files':trades},{'symbols':references},ticks)
    if any(check.get('error') or check.get('mismatches') for check in verification.values()):
        raise ValueError('frozen candidate trade tape did not match native bars')
    holdouts=proposals
    alternatives=[]
    alt_checks={}
    if len(retries)>1:
        holdouts,paths,alt_checks=paths_for(proposals,ticks,trades,references)
        alternatives=[row for row in paths if row['trail_rule'] in PROTECTION]
    new_archives,exit_checks=exit_books(proposals+alternatives,{'books':available},specs,books)
    evidence=dict(entry_checks=entry_checks,exit_checks=exit_checks,trade_checks=verification,
                  alternate_trade_checks=alt_checks,quote_status=Counter(r['entry_status'] for r in proposals),
                  downloaded_exit_sha256={k:v['sha256'] for k,v in new_archives.items()})
    return holdouts,specs,ticks,books,evidence


def cash_run(rows,specs,ticks,books,lag,*,rule=None):
    if any(r['entry_status']=='unknown_quote' for r in rows):
        return dict(status='UNSCORABLE_UNKNOWN_ENTRY',
                    missing=[key(r) for r in rows if r['entry_status']=='unknown_quote'])
    return simulate(rows,specs,ticks,books,clock=60,lag=lag,setting='requested5',
                    max_positions=12,unbankable_runner=rule)


def score(book_schedule,rows,specs,ticks,books,features):
    results={}
    picked={policy:list(book_cadence(rows,policy)) for policy in RANKINGS}
    context_rows = {}
    selected = picked['alphabetical']
    for policy in SWEEP_POLICIES:
        if policy == 'baseline':
            continue
        rejected = {
            key(row) for row in selected
            if row['entry_status'] == 'eligible' and refuses(policy, features[key(row)])
        }
        subset = [
            dict(row, entry_status='sweep_context_filter') if key(row) in rejected else row
            for row in selected
        ]
        context_rows[policy] = subset, rejected
    for lag in CLOSE_LAGS:
        control=cash_run(picked['alphabetical'],specs,ticks,books,lag)
        cases={}
        for name,subset in picked.items():
            replay=control if name=='alphabetical' else cash_run(subset,specs,ticks,books,lag)
            summary=cash_summary(replay,{})
            if 'closed' in replay:
                summary['by_market_alignment']=attribute(replay['closed'],subset)
                if name == 'alphabetical':
                    summary['closed_positions'] = replay['closed']
                elif 'closed' in control:
                    summary['trade_changes']=trade_difference(control,replay)
            cases[name]=summary
        for policy, (subset, rejected) in context_rows.items():
            replay = cash_run(subset, specs, ticks, books, lag)
            summary = cash_summary(replay, {})
            summary['rejected'] = [
                dict(symbol=row['symbol'], direction=row['direction'], entry_ms=row['entry_ms'],
                     context=features[key(row)]) for row in subset if key(row) in rejected
            ]
            if 'closed' in replay and 'closed' in control:
                summary['trade_changes'] = trade_difference(control, replay)
            cases[f'sweep_context:{policy}'] = summary
        if book_schedule=='observed_retry_sequence':
            for rule in PROTECTION:
                subset=picked['alphabetical']
                replay=cash_run(subset,specs,ticks,books,lag,rule=rule)
                summary=cash_summary(replay,{})
                if 'closed' in replay and 'closed' in control:
                    summary['trade_changes']=trade_difference(control,replay)
                cases[f'alphabetical_with_{rule}']=summary
        results[str(lag)]=cases
        print(book_schedule,lag,{name:(v.get('trades'),v.get('cash_pnl_usd'),v.get('reason'))
                                 for name,v in cases.items()},flush=True)
    return results


def main():
    captured,rows,checks,digest=load_candidates()
    print('Frozen crypto proposals',len(rows),'scans',len({row['epoch'] for row in rows}),flush=True)
    available={}
    results={}
    evidence={}
    for name,schedule in BOOK_SCHEDULES.items():
        proposals,specs,ticks,books,evidence[name]=material(captured,rows,schedule,available)
        features,extra=contexts(proposals,specs,captured['references'])
        evidence[name]['sweep_context_coverage']=Counter(ctx['sweep'] if ctx else 'unknown' for ctx in features.values())
        evidence[name]['sweep_context_extra_candles']=extra
        results[name]=score(name,proposals,specs,ticks,books,features)
    report=dict(start=START,entry_stop=ENTRY_STOP,outcome_coverage_end=ENTRY_STOP+HOLD['crypto'],
                capital_usd=CAPITAL,margin=FRACTION,leverage=LEVERAGE,source_sha256=digest,
                model_source_sha256=captured['source_digests'],book_schedules=BOOK_SCHEDULES,
                trade_sha256={symbol: {day: item['sha256'] for day, item in record['days'].items()}
                              for symbol, record in captured['trades'].items()},
                candle_sha256={symbol: item['sha256'] for symbol, item in captured['references'].items()},
                frozen_rankings=RANKINGS,frozen_protection=PROTECTION,frozen_sweep_context=SWEEP_POLICIES,
                proposal_count=len(rows),
                candidate_checks=checks,public_market_checks=evidence,
                entry_archive_hashes={k:v['sha256'] for k,v in available.items() if 'sha256' in v},
                results=results,live_activation_supported=False,limitations=[
                    'The eight-hour crypto window begins after the earlier November 20 holdout.',
                    'The original retry model is complemented by one synchronized 11-second book with no look-ahead across the scans.',
                    'A protective variant only acts if the modeled position cannot place the requested TP1 partial in whole lots.',
                    'Sweep context uses closed candles and applies the earlier vetoes after chronological publication admission.',
                    'Historical public order books and trades price market execution. Private fills, daily loss brakes and the stock bot are outside this holdout.',
                    'This is an eight-hour sample using a common-account simulation. It does not establish calendar-month performance.',
                ])
    write_json(ROOT/f'{STEM}.json',report)


if __name__=='__main__':
    main()
