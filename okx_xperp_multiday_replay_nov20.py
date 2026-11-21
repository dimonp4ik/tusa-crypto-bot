"""Score the declared six-day cohort with the pilot's exact public replay."""
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path
import pickle

from okx_xperp_cash_tick_nov19 import AMEND_CLOCKS, CLOSE_LAGS, TP1_SETTINGS, read_book_job, simulate
from okx_xperp_l2_capture_nov19 import catalogue, download
from okx_xperp_l2_entry_replay_nov19 import attempt
from okx_xperp_multiday_capture_nov20 import FOLDER, HOLD, file_hash
from okx_xperp_tick_path_nov19 import load_trade_tape, reconcile, resolve_ticks


ROOT=FOLDER.parents[1]
PILOT_DAY='2026-11-13'


def utc_day(stamp):
    return datetime.fromtimestamp(stamp,timezone.utc).date().isoformat()


def entry_offset(row,plan):
    return plan['parameters']['stock_offset_ms' if row['bot']=='stocks' else 'crypto_offset_ms']


def key_for(symbol,ms):
    return f'{ms//86_400_000*86400}:{symbol}'


def pull_samples(jobs, *, stage):
    books=defaultdict(dict)
    checked={}
    with ProcessPoolExecutor(max_workers=4) as pool:
        for job,(symbol,result,evidence) in zip(jobs,pool.map(read_book_job,jobs)):
            if set(books[symbol])&set(result):
                raise ValueError('duplicate L2 sample within stage')
            books[symbol].update(result)
            checked[f'{job[1]}:{symbol}']=evidence
            if len(checked)%10==0 or len(checked)==len(jobs):
                print(stage,len(checked),'/',len(jobs),flush=True)
    return books,checked


def entries(plan,capture):
    requests=defaultdict(set)
    for row in plan['rows']:
        first=int(row['epoch']*1000)+entry_offset(row,plan)
        for delay in plan['parameters']['retry_offsets_ms']:
            stamp=first+delay
            requests[key_for(row['symbol'],stamp)].add(stamp)
    jobs=[]
    for key,points in sorted(requests.items()):
        item=capture['books'][key]
        if 'path' in item:
            jobs.append((item['symbol'],item['day'],item,sorted(points)))
    books,checks=pull_samples(jobs,stage='entry archives reconstructed')
    proposals=[]
    for row in plan['rows']:
        symbol=row['symbol']
        first=int(row['epoch']*1000)+entry_offset(row,plan)
        attempts=[]
        stamp=first
        for delay in plan['parameters']['retry_offsets_ms']:
            stamp=first+delay
            value=attempt(books[symbol].get(stamp),stamp,row['direction'],plan['specs'][symbol],
                          row['stop'],row['target'])
            attempts.append(dict(ms=stamp,**value))
            if value['status']!='wide':
                break
        status=attempts[-1]['status']
        result=dict(row,entry_status=status,entry_ms=stamp,tries=attempts,
                    stock_reference_ready=row['paper_reference_ready'])
        if status=='eligible':
            result['reference_entry']=attempts[-1]['average']
        proposals.append(result)
    return proposals,books,checks


def cached_entries(plan,capture,inputs):
    path=FOLDER/'entry_samples.pkl'
    source_files=inputs[:2]+(Path(__file__),Path('okx_xperp_l2_entry_replay_nov19.py'))
    signature=[file_hash(p) for p in source_files]
    if path.exists():
        saved=pickle.loads(path.read_bytes())
        if saved['signature']==signature:
            print('verified saved entry quotes',len(saved['values'][0]),flush=True)
            return saved['values']
    result=entries(plan,capture)
    temp=path.with_suffix('.pkl.tmp')
    temp.write_bytes(pickle.dumps(dict(signature=signature,values=result),protocol=5))
    temp.replace(path)
    return result


def resolve_all(rows,trades,reference,ticks,*,trade_loader=load_trade_tape):
    by_symbol=defaultdict(list)
    for row in rows:
        by_symbol[row['symbol']].append(row)
    evidence={}
    for symbol,candidates in by_symbol.items():
        try:
            tape=trade_loader(trades['files'][symbol])
            times=[event.ms for event in tape]
            item=reference['symbols'][symbol]
            path=Path(item['path'])
            if file_hash(path)!=item['sha256']:
                raise ValueError('15m source digest mismatch')
            frame=pickle.loads(path.read_bytes())
            begin=min(r['epoch'] for r in candidates)
            end=max(r['epoch']+HOLD[r['bot']] for r in candidates)
            mismatches=reconcile(tape,frame,start=begin,end=end)
        except (OSError,ValueError,KeyError) as exc:
            evidence[symbol]=dict(error=f'{type(exc).__name__}: {exc}')
            for r in candidates:
                if r['entry_status']=='eligible':
                    r['path']={clock:dict(known=False,reason='trade_archive_incomplete') for clock in AMEND_CLOCKS}
            continue

        relevant={at for r in candidates for at in range(r['epoch'],r['epoch']+HOLD[r['bot']],900)}
        errors={at:code for at,code in mismatches.items() if at in relevant}
        evidence[symbol]=dict(tape_trades=len(tape),reconciled_bars=len(relevant),mismatches=errors)

        def checked(row,at,px,clock):
            result=resolve_ticks(tape,times,signal=row,entered_at=at,entry_px=px,tick_size=ticks[symbol],
                                 amend_seconds=clock)
            if result['known']:
                first=at//900_000*900
                last=(result['end_ms']-int(result['code']=='TIME'))//900_000*900
                invalid=[t for t in errors if first<=t<=last]
                if invalid:
                    return dict(known=False,reason='trade_tape_candle_disagreement',bars=invalid)
            return result

        for r in candidates:
            r['paper_path']=checked(r,r['epoch']*1000,r['entry'],900)
            if r['entry_status']=='eligible':
                r['path']={clock:checked(r,r['entry_ms'],r['reference_entry'],clock) for clock in AMEND_CLOCKS}
    return evidence


def exit_books(rows,capture,specs,books):
    needed=defaultdict(set)
    for row in rows:
        # The shared cash model keeps the declared reference readiness gate.
        if row['entry_status']!='eligible' or row['bot']=='stocks' and not row['stock_reference_ready']:
            continue
        for path in row['path'].values():
            if not path['known']:
                continue
            for when in (path['end_ms'],path['bank']['ms'] if path['bank'] else None):
                if when is not None:
                    for lag in CLOSE_LAGS:
                        stamp=when+lag
                        if stamp not in books[row['symbol']]:
                            needed[key_for(row['symbol'],stamp)].add(stamp)
    extra=defaultdict(set)
    for key in needed:
        if 'path' not in capture['books'].get(key,{}):
            day,symbol=key.split(':',1)
            extra[int(day)].add(symbol)
    downloads={}
    for day,symbols in sorted(extra.items()):
        families={s:specs[s]['instId'].rsplit('-',1)[0] for s in sorted(symbols)}
        files=catalogue(sorted(set(families.values())),day_start=day)
        for symbol,family in families.items():
            if family in files:
                downloads[f'{day}:{symbol}']=dict(download(family,files[family],day_start=day),
                                                   instrument=specs[symbol]['instId'])
    jobs=[]
    for key,points in sorted(needed.items()):
        day,symbol=key.split(':',1)
        item=capture['books'].get(key) or {}
        if 'path' not in item:
            item=downloads.get(key,{})
        if item:
            jobs.append((symbol,int(day),item,sorted(points)))
    processed,checks=pull_samples(jobs,stage='exit archives reconstructed')
    for symbol,samples in processed.items():
        books[symbol].update(samples)
    return downloads,checks


def cases_for(rows,specs,ticks,books):
    unknown=[(r['symbol'],r['entry_ms']) for r in rows if r['entry_status']=='unknown_quote']
    if unknown:
        return dict(status='UNSCORABLE_ENTRY_ARCHIVE',unknown=unknown)
    result={}
    for clock in AMEND_CLOCKS:
        for lag in CLOSE_LAGS:
            for policy in TP1_SETTINGS:
                key=f'BE{clock}s_exit{lag}ms_{policy}'
                result[key]=simulate(rows,specs,ticks,books,clock=clock,lag=lag,setting=policy)
    return result


def main():
    inputs=(FOLDER/'plan.json',FOLDER/'capture.json',ROOT/'paired_contract_volume_nov19/manifest.json',
            ROOT/'native_universe_nov19_snapshot.json')
    plan,capture,ref,snapshot=(json.loads(path.read_bytes()) for path in inputs)
    if capture['plan_sha256']!=file_hash(inputs[0]):
        raise ValueError('frozen multiday inputs changed')
    spec=plan['specs']
    by_instrument={r['instId']:float(r['tickSz']) for r in snapshot['instruments'] if r.get('tickSz')}
    ticks={s:by_instrument[v['instId']] for s,v in spec.items()}
    rows,books,entry_checks=cached_entries(plan,capture,inputs)
    paths=resolve_all(rows,{'files':capture['trades']},ref,ticks)
    by_day=defaultdict(Counter)
    for row in rows:
        key=utc_day(row['epoch'])
        by_day[key][row['bot']+':'+row['entry_status']]+=1
    print('eligible entries by day',json.dumps(by_day),flush=True)
    extra,exit_checks=exit_books(rows,capture,spec,books)
    full=cases_for(rows,spec,ticks,books)
    later=[r for r in rows if utc_day(r['epoch'])!=PILOT_DAY]
    holdout=cases_for(later,spec,ticks,books)
    pilot=[r for r in rows if utc_day(r['epoch'])==PILOT_DAY]
    parity=cases_for(pilot,spec,ticks,books)
    prior=json.loads((ROOT/'okx_xperp_cash_tick_nov19.json').read_bytes())['cases']
    if set(parity)!=set(prior) or any(parity[key]!=value for key,value in prior.items()):
        raise ValueError('multiday run changed the existing one-day control')
    report=dict(status='MULTIDAY_NATIVE_PUBLIC_DEPTH',input_hashes={str(p):file_hash(p) for p in inputs},
        scope=dict(start=plan['start'],end=plan['coverage_end'],proposals=len(rows)),entry_counts=by_day,
        source_checks=dict(tape=paths,entry=entry_checks,exit=exit_checks,additional_book_files=extra),
        pilot_parity=True, full=full, excluding_original_pilot=holdout,paths=rows,
        objective_verified=False,live_activation_supported=False,
        limitations=['Historical, fixed headroom, quote- and tape-based simulation. No private fills or execution acknowledgments.',
            'Same 3% allocation and 10x, one realized-cash account, local position caps 4/8, common 12/90%, and 15m delayed slot release.',
            'Historical stock readiness frozen before replay; one continuous cash path per cohort and policy. No fees or floating equity pause.',
            'Nov14 partial coverage. The pilot-excluded result is a holdout for the first comparison, not a never-seen date range.',
            'Entry and exit ordering, book gaps and amended BE clock proxies inherit the pilot limitations.'])
    target=ROOT/'okx_xperp_multiday_cash_nov20.json'
    target.write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    for label,cases in (('full',full),('excluding_original_pilot',holdout)):
        print(label,json.dumps({k:{q:v for q,v in item.items() if q in ('status','return_pct','drawdown_pct','reason','symbol','trades','contribution_pct')}
                                for k,item in cases.items() if isinstance(item,dict)}),flush=True)


if __name__=='__main__':main()
