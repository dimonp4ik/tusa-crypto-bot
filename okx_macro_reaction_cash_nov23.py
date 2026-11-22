"""Reprice scheduled release pauses and short-term observed reactions."""

from bisect import bisect_right
from collections import Counter
import csv
from datetime import datetime
import json
from statistics import median
from zoneinfo import ZoneInfo

from crypto_minimum_bank_runner_nov19_audit import admitted_records, cash_result
from crypto_native_price_pool_nov19 import MANIFEST
from okx_delayed_trailing_cash_nov23 import stock_sources
from okx_post_close_cash_nov23 import trade_difference
from okx_selective_single_target_nov23 import EARLIER,JOINT,cash_summary,key
from okx_spot_confirmation_nov22 import load_native
from okx_sweep_context_cash_nov23 import context_windows,holdout_inputs,window
from okx_swing_target_cash_nov23 import historical_brackets
from okx_unbankable_tp1_nov21 import old_inputs,recent_inputs
from okx_xperp_cash_tick_nov19 import CAPITAL,CLOSE_LAGS,FRACTION,LEVERAGE,simulate
from okx_xperp_multiday_replay_nov20 import ROOT,file_hash


PREFIX='okx_macro_reaction_cash_nov23'
ET=ZoneInfo('America/New_York')
BLS_2026='https://www.bls.gov/schedule/2026/home.htm'
FED_MONTH='https://www.federalreserve.gov/newsevents/2026-{}.htm'
POLICIES=('baseline','scheduled_15m_before_45m_after','scheduled_15m_before_90m_after',
          'after_adverse_15m_reaction','after_any_large_15m_reaction','adverse_15m_move_any_day')
REACTION=1.5


def events():
    location=ROOT/'official_macro_events_nov1'
    raw=location/'events.csv'
    metadata=json.loads((location/'manifest.json').read_bytes())
    if file_hash(raw)!=metadata['events_sha256']:
        raise ValueError('frozen official calendar changed')
    with raw.open(encoding='utf-8',newline='') as source:
        values=[dict(kind=row['event_type'], published_at=int(datetime.fromisoformat(
                     row['event_time_utc'].replace('Z','+00:00')).timestamp()),source=row['source_url'])
                for row in csv.DictReader(source) if '2026-07-01'<=row['event_time_utc']<'2026-11-01']
    additions = [
        ('EMPLOYMENT_SITUATION','11-06','08:30',BLS_2026),
        ('CPI','11-10','08:30',BLS_2026),
        *[('PPI',stamp,'08:30',BLS_2026) for stamp in ('07-15','08-13','09-10','10-15','11-13')],
        *[('FOMC_MINUTES',stamp,'14:00',FED_MONTH.format(month)) for stamp,month in
          (('07-08','july'),('08-19','august'),('10-07','october'),('11-18','november'))],
    ]
    for event,day,clock,source in additions:
        released=datetime.fromisoformat(f'2026-{day}T{clock}').replace(tzinfo=ET)
        values.append(dict(kind=event,published_at=int(released.timestamp()),source=source))
    values.sort(key=lambda row:row['published_at'])
    stamps=[row['published_at'] for row in values]
    if len(stamps)!=len(set(stamps)) or any(stamp%900 for stamp in stamps):
        raise ValueError('official release times are ambiguous or off the candle grid')
    return values,file_hash(raw)


def volatility(bars,side):
    if bars is None or len(bars)<18 or bars[-1][4]<=0:
        return None
    prior=bars[-18:-1]
    ranges=[max(high-low,abs(high-previous[3]),abs(low-previous[3]))
            for previous,(_,high,low,_,_) in zip(prior,prior[1:])]
    typical=median(ranges)
    if typical<=0:
        return None
    opening,_,_,close,_=bars[-1]
    return (close-opening)/typical*(1 if side=='LONG' else -1)


def observe(at,direction,bars,calendar):
    stamps=calendar[0]
    i=bisect_right(stamps,at)
    previous=at-stamps[i-1] if i else None
    upcoming=stamps[i]-at if i<len(stamps) else None
    last=calendar[1][i-1]['kind'] if previous is not None and previous<=5400 else None
    next_kind=calendar[1][i]['kind'] if upcoming is not None and upcoming<=900 else None
    return dict(before=upcoming,after=previous,previous_kind=last,
                upcoming_kind=next_kind,signed_closed_move=volatility(bars,direction))


def refuse(policy,observation):
    minutes={'scheduled_15m_before_45m_after':45,'scheduled_15m_before_90m_after':90}
    if policy in minutes:
        pre,post=observation['before'],observation['after']
        return (pre is not None and pre<=900) or (post is not None and post<=minutes[policy]*60)
    move=observation['signed_closed_move']
    if policy=='baseline' or move is None:
        return False
    if policy=='adverse_15m_move_any_day':
        return move<=-REACTION
    elapsed=observation['after']
    # Prior to the first close after release, the policy has no reaction evidence.
    if elapsed is None or not 900<=elapsed<=5400:
        return False
    return move<=-REACTION if policy=='after_adverse_15m_reaction' else abs(move)>=REACTION


def print_results(label,outcomes):
    print(label,{name:(v.get('trades'),round(v['cash_pnl_usd'],4) if 'cash_pnl_usd' in v else v.get('reason'))
                 for name,v in outcomes.items()},flush=True)


def cash_compare(cohort,specs,ticks,books,params,observations,expected):
    outcomes={}
    original=simulate(cohort,specs,ticks,books,**params)
    fields=('trades','return_pct','drawdown_pct','contribution_pct') if 'closed' in original else ('symbol','reason')
    if any(expected[field]!=original[field] for field in fields):
        raise ValueError('published baseline differs from the news comparison control')
    for policy in POLICIES:
        skipped={key(row) for row in cohort if key(row) in observations and refuse(policy,observations[key(row)])}
        amended=[dict(r,entry_status='news_context_filter') if key(r) in skipped else r for r in cohort]
        replay=original if policy=='baseline' else simulate(amended,specs,ticks,books,**params)
        summary=cash_summary(replay,{})
        removed=[r for r in original.get('closed',[]) if key(r) in skipped]
        summary['rejected_proposals']=len(skipped)
        summary['directly_filtered_baseline']=dict(outcomes=Counter(r['code'] for r in removed),
            cash_pnl_usd=sum(r['bank_pnl']+r['runner_pnl'] for r in removed),
            positions=[dict(symbol=r['symbol'],bot=r['bot'],entry_ms=r['entry_ms'],code=r['code'],
                          macro_context=observations[key(r)]) for r in removed])
        if policy!='baseline' and 'closed' in original and 'closed' in replay:
            summary['replacement_trades']=trade_difference(original,replay)
        outcomes[policy]=summary
    return outcomes


def evaluate(material,cohorts,caps,base,calendar,*,joint=False):
    rows,specs,ticks,books,_,trades,reference=material
    if joint:
        _,reference=stock_sources(trades,reference)
    bars,downloads=context_windows(rows,specs,reference)
    observations={key(row):observe(row['epoch'],row['direction'],bars[key(row)],calendar)
                  for row in rows if key(row) in bars}
    cases={}
    for name,select in cohorts.items():
        cohort=[row for row in rows if select(row)]
        cases[name]={}
        for lag in CLOSE_LAGS:
            params=dict(clock=60,lag=lag,setting='requested5',max_positions=caps[name])
            outcomes=cash_compare(cohort,specs,ticks,books,params,observations,base[name][str(lag)]['baseline'])
            cases[name][str(lag)]=outcomes
            print_results((name,lag),outcomes)
    return cases,dict(downloaded_candle_windows=downloads,
                       missing_reaction=sum(ctx['signed_closed_move'] is None for ctx in observations.values()))


def historical(calendar,expected):
    history,specs,_,_,metadata,*_=historical_brackets()
    candles=load_native(json.loads(MANIFEST.read_bytes())['symbols'])
    proposals=history['baseline']
    observations=[observe(int(r['entry_time']),r['direction'],window(candles[r['symbol']],int(r['entry_time'])),calendar)
                  for r in proposals]
    result={}
    for policy in POLICIES:
        accepted=[row for row,ctx in zip(proposals,observations,strict=True) if not refuse(policy,ctx)]
        admitted,unknown=admitted_records(accepted)
        if unknown:
            result[policy]=dict(status='UNSCORABLE',admitted_unknown=unknown)
            continue
        cash=cash_result(admitted,specs,'requested5')
        if policy=='baseline' and cash!=expected:
            raise ValueError('published candle baseline differs from calendar control')
        result[policy]=dict(status='CANDLE_PROXY',cash=cash,filtered=len(proposals)-len(accepted))
        print('jul_oct',policy,cash['summary']['trades'],cash['summary']['return_pct'],
              cash['summary']['cash_drawdown_pct'],flush=True)
    return dict(cases=result,candidate_pool_sha256=metadata['sha256'],source_frames_sha256=file_hash(MANIFEST))


def stock_followup(calendar):
    from stock_liquid_quote_routes_nov21 import limit_bursts,load_replay,stressed_exit_penalty
    from stock_quote_continuation_holdout_nov21 import DATA,STEM

    plan,capture,rows,specs,ticks,books,_,tapes,_=load_replay(DATA,keep_tapes=True)
    chosen=list(limit_bursts((r for r in rows if r['route']=='aligned_continuation'),max_same_side=2))
    bars,downloads=context_windows(chosen,specs,capture['reference'])
    contexts={key(row):observe(row['epoch'],row['direction'],bars[key(row)],calendar)
              for row in chosen if key(row) in bars}
    path=ROOT/f'{STEM}_exit_stress.json'
    expected=json.loads(path.read_bytes())['results']
    cases={}
    for name,cohort in plan['cases'].items():
        keys={key(row) for row in cohort}
        sample=[row for row in chosen if key(row) in keys]
        cases[name]={}
        for bps in (5,15,30,50):
            params=dict(clock=60,lag=200,setting='requested5',
                        exit_price=stressed_exit_penalty(tapes,specs,bps))
            results=cash_compare(sample,specs,ticks,books,params,contexts,
                                 expected[name][f'aligned_continuation|stock_cadence|stress_{bps}bps'])
            for result in results.values():
                if 'cash_pnl_usd' in result:
                    result['status']='SCORED_PUBLIC_TRADE_EXIT_STRESS'
            cases[name][str(bps)]=results
            print_results((name,bps),results)
    return dict(cases=cases,source_sha256=file_hash(path),downloaded_candle_windows=downloads,
                known_reactions=sum(v['signed_closed_move'] is not None for v in contexts.values()),
                missing_reactions=sum(v['signed_closed_move'] is None for v in contexts.values()))


def main():
    published,calendar_digest=events()
    calendar=([row['published_at'] for row in published],published)
    source=ROOT/'okx_swing_target_cash_nov23.json'
    baseline=json.loads(source.read_bytes())
    cash=baseline['cash_cases']
    earlier,old_evidence=evaluate(old_inputs(),EARLIER,{k:12 for k in EARLIER},cash,calendar)
    cap12='nov17_19_joint_cap12'
    joint,joint_evidence=evaluate(recent_inputs(),JOINT|{cap12:lambda _:True},
                                  {'nov17_19_joint':6,cap12:12},cash,calendar,joint=True)
    fresh='nov20_four_hours'
    holdout,fresh_evidence=evaluate(holdout_inputs(),{fresh:lambda _:True},{fresh:12},cash,calendar)
    report=dict(capital_usd=CAPITAL,margin=FRACTION,leverage=LEVERAGE,
                official_release_times=published,calendar_sha256=calendar_digest,baseline_sha256=file_hash(source),
                reaction_min_median_ranges=REACTION,policies=POLICIES,
                exact_cash=earlier|joint|holdout,archives=dict(older=old_evidence,joint=joint_evidence,new=fresh_evidence),
                crypto_older_candle_proxy=historical(calendar,baseline['monthly_candle_proxy']['cases']['baseline']['cash']),
                later_stock_stress=stock_followup(calendar),live_activation_supported=False,limitations=[
                    'A replay around official US releases, not a claim about unscheduled events or earnings translation.',
                    'Only entries can be blocked. A reaction uses the last fully closed 15-minute trade candle and earlier ranges.',
                    'The exact cash sample keeps the original exits and lagged public quote execution.',
                    'The joint stock route is hypothetical, and the later independent stock sample uses stressed trade-print exits.',
                    'The crypto candle proxy and short November quote windows do not establish monthly performance.',
                ])
    (ROOT/f'{PREFIX}.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8',newline='\n')


if __name__=='__main__':
    main()
