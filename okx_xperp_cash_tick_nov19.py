"""Reprice the fixed OKX pilot with verified trade order and L2 market legs."""
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
import hashlib
import heapq
import json
import math
from pathlib import Path

from okx_xperp_l2_capture_nov19 import START, END, ROOT_L2, catalogue, download
from okx_xperp_l2_entry_replay_nov19 import archive
from okx_xperp_tick_path_nov19 import load_trade_tape, reconcile, resolve_ticks
from src.execution_guard import execution_quote
from src.okx_trader import calc_contracts, round_to_tick
from src.tp1_plan import plan_tp1_close


ROOT = ROOT_L2.parent
CAPITAL = 130.
FRACTION = .03
LEVERAGE = 10
CAPS = {'crypto':4,'stocks':8}
RELEASE_MS = 900_000
CLOSE_LAGS = (200,1000)
AMEND_CLOCKS = (60,900)
TP1_SETTINGS = {'requested5':(5.,False),'half_minimum':(50.,True)}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(entries, tapes, frames, ticks):
    import pickle
    result=[]
    tape_checks={}
    for symbol,view in entries['instruments'].items():
        if not view.get('proposals'):
            continue
        item=tapes['files'][symbol]
        tape=load_trade_tape(item)
        times=[row.ms for row in tape]
        cache=frames['symbols'][symbol]
        frame_path=Path(cache['path'])
        if digest(frame_path)!=cache['sha256']:
            raise ValueError('15m benchmark digest changed')
        frame=pickle.loads(frame_path.read_bytes())
        end=END+57_600
        errors=reconcile(tape,frame,start=START,end=end)
        tape_checks[symbol]=dict(trades=len(tape),checked_bars=(end-START)//900,
                                 discrepancy_count=len(errors),discrepancies=errors)

        def checked_outcome(row, opened, price, clock):
            outcome=resolve_ticks(tape,times,signal=row,entered_at=opened,entry_px=price,
                                  tick_size=ticks[symbol],amend_seconds=clock)
            if outcome['known']:
                begin=opened//900_000*900
                last=(outcome['end_ms']-int(outcome['code']=='TIME'))//900_000*900
                if any(begin<=ts<=last for ts in errors):
                    return dict(known=False,reason='trade_tape_candle_disagreement')
            return outcome

        for row in view['proposals']:
            latency='10000' if row['bot']=='crypto' else '70000'
            accepted=row['variants'][latency][-1]
            proposal=dict(row,entry_status=accepted['status'],entry_ms=int(row['epoch']*1000)+accepted['offset_ms'])
            proposal['paper_path']=checked_outcome(row,int(row['epoch']*1000),row['entry'],900)
            if accepted['status']=='eligible':
                proposal['reference_entry']=accepted['average']
                proposal['path']={clock:checked_outcome(row,proposal['entry_ms'],accepted['average'],clock)
                                  for clock in AMEND_CLOCKS}
            result.append(proposal)
    return result,tape_checks


def sample_points(rows):
    points=defaultdict(set)
    for row in rows:
        if row['entry_status']!='eligible':
            continue
        symbol=row['symbol']
        points[symbol].add(row['entry_ms'])
        for p in row['path'].values():
            if not p['known']:
                continue
            times=[p['end_ms']]
            if p['bank']:
                times.append(p['bank']['ms'])
            for t in times:
                for lag in CLOSE_LAGS:
                    points[symbol].add(t+lag)
    return points


def read_book_job(job):
    symbol,day_start,metadata,points=job
    data,verification=archive(Path(metadata['path']),metadata['instrument'],metadata['sha256'],points,
                              day_start=day_start)
    return symbol,data,verification


def collect_books(rows, manifest):
    jobs=[]
    extra={}
    for symbol,points in sample_points(rows).items():
        item=manifest['files'][symbol]
        same=[t for t in points if t<END*1000]
        next_day=[t for t in points if t>=END*1000]
        jobs.append((symbol,START,item,same))
        if next_day:
            extra[symbol]=(item,next_day)
    extra_files={}
    if extra:
        families=sorted({item['family'] for item,_ in extra.values()})
        following=catalogue(families,day_start=END)
        for symbol,(before,points) in sorted(extra.items()):
            market_file=following.get(before['family'])
            if market_file is None:
                raise ValueError('missing next-day archive for an open pilot trade')
            new=dict(download(before['family'],market_file,day_start=END),instrument=before['instrument'])
            jobs.append((symbol,END,new,points))
            extra_files[symbol]=new
    books=defaultdict(dict)
    evidence={}
    with ProcessPoolExecutor(max_workers=3) as pool:
        for symbol,samples,check in pool.map(read_book_job,jobs):
            if set(books[symbol])&set(samples):
                raise ValueError('duplicate L2 instant across daily files')
            books[symbol].update(samples)
            evidence.setdefault(symbol,[]).append(check)
            print('entry/exit books',symbol,len(samples),flush=True)
    return books,extra_files,evidence


def close_quote(book, stamp, side, contracts):
    if book is None:
        raise ValueError('missing exit L2 sample')
    opposite='SHORT' if side=='LONG' else 'LONG'
    # A running position has to unwind its size even when entry gates would refuse.
    return execution_quote(book,opposite,contracts,stamp,max_spread=1.,max_slippage=1.).average


def simulate(rows, specs, ticks, books, *, clock, lag, setting):
    cash=peak=CAPITAL
    drawdown=0.
    max_used=0
    max_reserved=0.
    active={}
    events=[]
    closed=[]
    skipped=Counter()
    contribution={bot:0. for bot in CAPS}

    def drain(until):
        nonlocal cash,peak,drawdown
        while events and (until is None or events[0][0]<until):
            _,index,kind,symbol,pnl=heapq.heappop(events)
            if kind=='release':
                del active[symbol]
            else:
                cash+=pnl
                contribution[closed[index]['bot']]+=pnl
                peak=max(peak,cash)
                drawdown=max(drawdown,(peak-cash)/peak)

    for row in sorted(rows,key=lambda item:(item['entry_ms'],item['bot'],item['symbol'])):
        symbol,bot,opened,side=(row[k] for k in ('symbol','bot','entry_ms','direction'))
        drain(opened)
        if row['entry_status']!='eligible':
            skipped[bot+':'+row['entry_status']]+=1
            continue
        if bot=='stocks' and not row['stock_reference_ready']:
            skipped['stocks:paper_guard']+=1
            continue
        if symbol in active or len(active)>=12 or sum(r[0]==bot for r in active.values())>=CAPS[bot]:
            skipped[bot+':position_cap']+=1
            continue
        reserved=math.fsum(r[1] for r in active.values())
        if cash<=0 or reserved+cash*FRACTION>cash*.9:
            skipped[bot+':margin_cap']+=1
            continue
        path=row['path'][clock]
        if not path['known']:
            return dict(status='UNSCORABLE',symbol=symbol,entry_ms=opened,reason=path['reason'])
        spec=specs[symbol]
        book=books[symbol].get(opened)
        try:
            if book is None:
                raise ValueError('missing entry L2 sample')
            entry=execution_quote(book,side,None,opened,
                size_for_price=lambda price:calc_contracts(cash*FRACTION,LEVERAGE,price,spec))
            entry.validate_bracket(row['stop'],row['target'])
            if bot=='crypto' and round_to_tick(entry.average,ticks[symbol])!=path['be_level']:
                raise ValueError('compounded size changed the precomputed BE level')
            tp1=round_to_tick(row['entry']+.5*(row['entry']-row['stop']),ticks[symbol])
            if bot=='crypto' and (1 if side=='LONG' else -1)*(entry.average-tp1)>=0:
                raise ValueError('compounded entry is already beyond TP1')
            partial,minimum=TP1_SETTINGS[setting]
            bank_size=(plan_tp1_close(entry.contracts,partial,spec,ensure_minimum=minimum)['size']
                       if path['bank'] else 0.)
            remainder=max(0.,entry.contracts-bank_size)
            bank_at=(path['bank']['ms']+lag) if bank_size else None
            bank_px=close_quote(books[symbol].get(bank_at),bank_at,side,bank_size) if bank_size else None
            end_at=path['end_ms']+lag if remainder>1e-12 else bank_at
            runner_px=close_quote(books[symbol].get(end_at),end_at,side,remainder) if remainder>1e-12 else None
        except ValueError as exc:
            return dict(status='UNSCORABLE',symbol=symbol,entry_ms=opened,reason=str(exc))
        sign=1 if side=='LONG' else -1
        unit=float(spec['ctVal'])*sign
        bank_pnl=unit*bank_size*(bank_px-entry.average) if bank_px is not None else 0.
        runner_pnl=unit*remainder*(runner_px-entry.average) if runner_px is not None else 0.
        index=len(closed)
        closed.append(dict(symbol=symbol,bot=bot,entry_ms=opened,exit_ms=end_at,code=path['code'],
                           entry_px=entry.average,qty=entry.contracts,bank_qty=bank_size,
                           bank_px=bank_px,runner_px=runner_px,bank_pnl=bank_pnl,runner_pnl=runner_pnl))
        if bank_at is not None:
            heapq.heappush(events,(bank_at,index,'bank',symbol,bank_pnl))
        if runner_px is not None:
            heapq.heappush(events,(end_at,index,'runner',symbol,runner_pnl))
        heapq.heappush(events,(end_at+RELEASE_MS,index,'release',symbol,0.))
        margin=entry.contracts*float(spec['ctVal'])*max(entry.best,entry.worst)/LEVERAGE
        active[symbol]=(bot,margin)
        max_used=max(max_used,len(active))
        max_reserved=max(max_reserved,(reserved+margin)/cash)
    drain(None)
    if active or abs(cash-CAPITAL-math.fsum(contribution.values()))>1e-9 or max_used>12 or max_reserved>.9:
        raise ValueError('shared account ledger did not reconcile')
    return dict(status='SCORED_PUBLIC_L2_SIMULATION',trades=len(closed),skipped=skipped,
                return_pct=100*(cash/CAPITAL-1),drawdown_pct=100*drawdown,
                contribution_pct={k:100*v/CAPITAL for k,v in contribution.items()},
                peak_reservations=max_used,peak_margin_pct=100*max_reserved,closed=closed)


def main():
    entry_path=ROOT/'okx_xperp_l2_entry_replay_nov19.json'
    l2_path=ROOT_L2/'manifest.json'
    tape_path=ROOT_L2/'trades/manifest.json'
    frame_path=ROOT/'paired_contract_volume_nov19/manifest.json'
    spec_path=ROOT/'shared_xperp_specs_nov14.json'
    snapshot_path=ROOT/'native_universe_nov19_snapshot.json'
    entries,l2,tapes,frames,specs,snapshot=[json.loads(p.read_bytes()) for p in
        (entry_path,l2_path,tape_path,frame_path,spec_path,snapshot_path)]
    if entries['manifest_sha256']!=digest(l2_path) or not tapes['complete'] or not frames['complete']:
        raise ValueError('missing or mixed pilot sources')
    by_instrument={row['instId']:float(row['tickSz']) for row in snapshot['instruments'] if row.get('tickSz')}
    ticks={symbol:by_instrument[item['instId']] for symbol,item in specs.items() if item['instId'] in by_instrument}
    rows,checks=prepare(entries,tapes,frames,ticks)
    eligible=[r for r in rows if r['entry_status']=='eligible']
    transitions=Counter(tuple(r['path'][k].get('code','UNKNOWN') for k in AMEND_CLOCKS) for r in eligible)
    print('tape checks',len(checks),sum(v['discrepancy_count'] for v in checks.values()),
          'eligible',len(eligible),'transitions',dict(transitions),flush=True)
    books,extra,book_checks=collect_books(rows,l2)
    cases={}
    for clock in AMEND_CLOCKS:
        for lag in CLOSE_LAGS:
            for policy in TP1_SETTINGS:
                key=f'BE{clock}s_exit{lag}ms_{policy}'
                cases[key]=simulate(rows,specs,ticks,books,clock=clock,lag=lag,setting=policy)
                summary={k:v for k,v in cases[key].items() if k!='closed'}
                print(key,summary,flush=True)
    for row in eligible:
        row['path']={str(k):v for k,v in row['path'].items()}
    report=dict(status='FIXED_OKX_TICK_AND_DEPTH_PILOT',sources={str(p):digest(p) for p in
                (entry_path,l2_path,tape_path,frame_path,spec_path,snapshot_path)},
                objective_verified=False,live_activation_supported=False,tape_checks=checks,
                entry_count=len(rows),entry_eligible=len(eligible),paths=rows,
                amendment_comparison={'->'.join(k):v for k,v in transitions.items()},
                next_day_book_files=extra,book_checks=book_checks,cases=cases,
                limitations=[
                    'One UTC day. Retrospective public prices and displayed depth, no client orders or fills.',
                    'BE activation at next minute or next 15m boundary are explicit modeled clocks, not actual exchange amendment acknowledgments.',
                    'Order protection is assumed armed at entry. Trigger books sampled strictly before trigger+200ms or +1s. Same-tick double target exits are aggregated.',
                    'Shared cash is realized-only, with an atomic single entry writer, stock historical readiness, 12 total and 4/8 bot slots, 90% margin limit, 15m release.',
                    'No commissions, floating-equity drawdown pause, cancellations or trade network simulation. Current exact-contract size/tick specifications used historically.',
                    'Any missing accepted price or inconsistent path makes its whole cash scenario unscorable.',
                ])
    (ROOT/'okx_xperp_cash_tick_nov19.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')


if __name__=='__main__':main()
