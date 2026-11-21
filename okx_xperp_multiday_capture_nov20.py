"""Freeze and capture the six-day X-Perp entry cohort before scoring it."""
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
import hashlib
import json
from pathlib import Path
import time

from account_cash_guard_nov17 import stamp
from okx_xperp_l2_capture_nov19 import ROOT_L2, catalogue, download
from okx_xperp_trade_capture_nov19 import TRADE_ROOT, TRADE_TZ, fetch_family, get_catalogue
from shared_admission_replay_nov19 import records


FOLDER = ROOT_L2 / 'nov9_14'
START = int(stamp('2026-11-09'))
DAY_MS = 86_400_000
HOLD = {'crypto':43_200,'stocks':57_600}


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def local_trade_days(start,end):
    first=datetime.fromtimestamp(start,TRADE_TZ).replace(hour=0,minute=0,second=0,microsecond=0)
    last=datetime.fromtimestamp(end-1,TRADE_TZ).replace(hour=0,minute=0,second=0,microsecond=0)
    return {int(day.timestamp()*1000):day.date().isoformat()
            for day in (first+timedelta(days=i) for i in range((last-first).days+1))}


def freeze(*, start=START, entry_end=None):
    crypto,ready,stock,specs,end=records()
    if entry_end is not None:
        end = min(end, entry_end + max(HOLD.values()))
    entry_end = entry_end if entry_end is not None else end
    ready_keys={(r['symbol'],r['entry_time']) for r in ready}
    proposals=[]
    for r in crypto+stock:
        at=int(r['entry_time'])
        if not start<=at<entry_end:
            continue
        bot=r['bot']; e=r['entry']; side=r['direction']; sign=1 if side=='LONG' else -1
        stop=r['sl'] if bot=='crypto' else e*(1-sign*r['risk_pct'])
        target=e+2*(e-stop) if bot=='crypto' else e*(1+sign*r['target_pct']/100)
        if at+HOLD[bot]>end:
            raise ValueError('unmatured signal in historical cohort')
        proposals.append(dict(epoch=at,entry=e,stop=stop,target=target,bot=bot,direction=side,
            symbol=r['symbol'],paper_reference_ready=(r['symbol'],r['entry_time']) in ready_keys))
    proposals.sort(key=lambda r:(r['epoch'],r['bot'],r['symbol']))
    if len(proposals)!=len({(r['bot'],r['symbol'],r['epoch']) for r in proposals}):
        raise ValueError('duplicate frozen proposal')
    return dict(start=start,coverage_end=int(end),specs={s:specs[s] for s in sorted({r['symbol'] for r in proposals})},
        rows=proposals,parameters=dict(capital=130.,margin=.03,leverage=10,crypto_offset_ms=10_000,
            stock_offset_ms=70_000,exit_lags_ms=[200,1000],be_seconds=[60,900],
            retry_offsets_ms=[0,200,800],account_cap=12,bot_caps={'crypto':4,'stocks':8},
            margin_cap=.9,release_delay_ms=900_000), research_only=True)


def book_files(plan):
    days=defaultdict(set)
    for row in plan['rows']:
        days[row['epoch']//86400*86400].add(row['symbol'])
    result={}
    with ThreadPoolExecutor(max_workers=5) as pool:
        requests={}
        for day,symbols in sorted(days.items()):
            by_symbol={s:plan['specs'][s]['instId'] for s in sorted(symbols)}
            by_family={s:inst.rsplit('-',1)[0] for s,inst in by_symbol.items()}
            exchange=catalogue(sorted(set(by_family.values())),day_start=day)
            for symbol,family in by_family.items():
                key=f'{day}:{symbol}'
                info=dict(day=day,symbol=symbol,instrument=by_symbol[symbol],family=family)
                result[key]=info
                if family in exchange:
                    requests[pool.submit(download,family,exchange[family],day_start=day)]=key
                else:
                    info['error']='missing archived book day'
            time.sleep(.5)
        for task in as_completed(requests):
            key=requests[task]
            try:
                result[key].update(task.result())
            except Exception as exc:
                result[key]['error']=f'{type(exc).__name__}: {exc}'
            if len([x for x in result.values() if 'path' in x or 'error' in x])%10==0:
                print('entry archives finished',sum('path' in x or 'error' in x for x in result.values()),flush=True)
    return result


def trade_files(plan):
    days=local_trade_days(plan['start'],plan['coverage_end'])
    families={s:v['instId'].rsplit('-',1)[0] for s,v in plan['specs'].items()}
    catalog=get_catalogue(sorted(set(families.values())),trade_days=days)
    requirement=defaultdict(dict)
    for row in plan['rows']:
        at=row['epoch']//900*900
        requirement[row['symbol']].update(local_trade_days(at, at+HOLD[row['bot']]))
    TRADE_ROOT.mkdir(parents=True,exist_ok=True)
    jobs=[]
    for symbol,family in sorted(families.items()):
        required=requirement[symbol]
        existing=[r for r in catalog.get(family,[]) if int(r['dateTs']) in required]
        jobs.append(((symbol,plan['specs'][symbol]['instId'],existing),required))
    with ThreadPoolExecutor(max_workers=5) as pool:
        return dict(pool.map(lambda pair:fetch_family(pair[0],trade_days=pair[1]),jobs))


def main():
    FOLDER.mkdir(parents=True,exist_ok=True)
    planned=freeze()
    plan_path=FOLDER/'plan.json'
    if plan_path.exists() and json.loads(plan_path.read_bytes())!=planned:
        raise ValueError('multiday cohort or policies changed')
    if not plan_path.exists():
        plan_path.write_text(json.dumps(planned,indent=2,allow_nan=False),encoding='utf-8')
    print('frozen proposals',len(planned['rows']),flush=True)
    books=book_files(planned)
    trades=trade_files(planned)
    capture=dict(plan_sha256=file_hash(plan_path),books=books,trades=trades,
        complete=all('path' in x for x in books.values()) and all(x['complete'] for x in trades.values()))
    (FOLDER/'capture.json').write_text(json.dumps(capture,indent=2),encoding='utf-8')
    print('multiday captured',capture['complete'],len(books),sum(len(x['days']) for x in trades.values()),flush=True)


if __name__=='__main__':main()
