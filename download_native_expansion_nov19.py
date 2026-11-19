"""Immutable public native-history expansion, separate from production universes."""
from concurrent.futures import ThreadPoolExecutor,as_completed
import hashlib
import json
import math
from pathlib import Path
import pickle
import threading
import time

from src.binance_client import _okx_get

ROOT=Path('reports/audit_2026_09_08')
SNAPSHOT=ROOT/'native_universe_nov19_snapshot.json'
FOLDER=ROOT/'native_expansion_nov19'
SELECTION=FOLDER/'selection.json'
EXCLUDED_REVISIT=('ADA','DOT','NEAR','TAO')
US_EQUITIES=('SNDK','TTWO','AMD','CRCL','SOXS','NBIS','PLTR','COIN','HOOD','AAPL','META')
LISTED_BEFORE=1782864000 # July1 UTC, known listing metadata, not a PnL filter.
START=1777593600 # May1 UTC.


class Public:
    def __init__(self):self.lock=threading.Lock();self.next=0.
    def get(self,path,params):
        with self.lock:
            delay=self.next-time.monotonic()
            if delay>0:time.sleep(delay)
            self.next=time.monotonic()+.26
        return _okx_get(path,params,timeout=10)


def strict_instruments(rows):
    out={}
    for r in rows:
        inst=r.get('instId','')
        if ('_UM_XPERP-' not in inst or r.get('state')!='live' or r.get('ruleType')!='xperp'
                or r.get('ctType')!='linear' or r.get('settleCcy')!='USD'):continue
        base=inst.split('-')[0]
        if base not in out or inst>out[base]['instId']:out[base]=r
    return out


def parse(rows,kind,start,end):
    values={}
    for row in rows:
        size=9 if kind=='trade' else 6
        if len(row)!=size:raise ValueError('expanded_history_schema_changed')
        if str(row[-1])!='1':continue
        stamp=int(row[0])//1000
        if int(row[0])%900000:raise ValueError('unaligned_expanded_history')
        if not start<=stamp<end:continue
        price=[float(v) for v in row[1:5]]
        if not all(math.isfinite(v) and v>0 for v in price) or not price[2]<=min(price[0],price[3])<=max(price[0],price[3])<=price[1]:
            raise ValueError('invalid_expanded_ohlc')
        if kind=='trade':
            volume=float(row[5])
            if not math.isfinite(volume) or volume<0:raise ValueError('invalid_expanded_volume')
            price.append(volume)
        if stamp in values and values[stamp]!=price:raise ValueError('conflicting_expanded_candle_revision')
        values[stamp]=price
    times=sorted(values)
    frame=dict(time=times,**{key:[values[t][i] for t in times] for i,key in enumerate(
        ('open','high','low','close','volume') if kind=='trade' else ('open','high','low','close'))})
    gaps=[(a,b) for a,b in zip(times,times[1:]) if b-a!=900]
    return frame,gaps


def choose(snapshot):
    specs=strict_instruments(snapshot['instruments'])
    old=json.loads((ROOT/'xperp_15m_nov14_merged/manifest.json').read_text(encoding='utf-8'))
    old_bases={s[:-4] for s in old['symbols']}
    ranked=[r for r in snapshot['ranking'] if r['category']=='1' and r['base'] not in old_bases
        and r['base'] not in EXCLUDED_REVISIT and r['base'] in specs
        and int(specs[r['base']]['listTime'])/1000<LISTED_BEFORE]
    extra=[r['base'] for r in sorted(ranked,key=lambda r:(-r['turnover_reference'],r['base']))[:6]]
    end=min(v['end'] for v in old['symbols'].values())+900
    stock_bases=[];late={}
    for base in US_EQUITIES:
        r=specs.get(base)
        if r is None:late[base]='current_contract_unavailable';continue
        if int(r['listTime'])/1000+1072*900>=end:
            late[base]='current_contract_listed_too_late_for_historical_warmup_and_exit';continue
        stock_bases.append(base)
        if len(stock_bases)==6:break
    if len(stock_bases)!=6:raise ValueError('insufficient_declared_US_contract_history')
    selected={}
    for base in (*EXCLUDED_REVISIT,*extra,*stock_bases):
        r=specs.get(base)
        if r is None:raise ValueError('declared_expansion_contract_unavailable_'+base)
        category='1' if base not in stock_bases else '3'
        if r['instCategory']!=category:raise ValueError('expanded_asset_category_conflict')
        selected[base+'USDT']=dict(instrument=r['instId'],category=category,spec=r,
            listed_after_July1=int(r['listTime'])/1000>=LISTED_BEFORE,
            kinds=['trade'] if category=='1' else ['trade','mark'])
    return dict(selected=selected,start=START,end=end,known_listing_exclusions=late,
        snapshot_sha256=hashlib.sha256(SNAPSHOT.read_bytes()).hexdigest(),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        selection_basis='four previously excluded crypto plus six highest current snapshot turnover untested crypto listed before July1; first six explicit US equities with current-contract history before research end',
        limitations=['Current-universe/current-turnover selection is hindsight for old months, not a historical universe or independent temporal holdout.',
            'Strict payload category/ruleType/linear/USD checks; query parameters alone do not filter the returned catalogue.',
            'No production universe change, no keys/orders; current specs assumed historically.'])


def fetch(symbol,item,kind,selection,source):
    instrument=item['instrument'];folder=FOLDER/'pages'/symbol/kind;folder.mkdir(parents=True,exist_ok=True)
    start=max(selection['start'],int(math.ceil(int(item['spec']['listTime'])/1000/900))*900)
    end=selection['end'];after=end*1000;rows=[];pages=[]
    endpoint='/api/v5/market/history-candles' if kind=='trade' else '/api/v5/market/history-mark-price-candles'
    for _ in range(250):
        params=dict(instId=instrument,bar='15m',limit=100,after=str(after));path=folder/f'{after}.json'
        if path.exists():
            raw=path.read_bytes();stored=json.loads(raw)
            if stored['endpoint']!=endpoint or stored['params']!=params:raise ValueError('expanded_cached_request_changed')
            body=stored['response']
        else:
            body=None
            for attempt in range(3):
                try:body=source.get(endpoint,params);break
                except Exception:
                    if attempt==2:raise
                    time.sleep(1+attempt)
            if str(body.get('code','0'))!='0':raise ValueError('expanded_exchange_error')
            raw=json.dumps(dict(endpoint=endpoint,params=params,received_at=time.time(),response=body),
                sort_keys=True,separators=(',',':')).encode();path.write_bytes(raw)
        pages.append(dict(path=str(path),sha256=hashlib.sha256(raw).hexdigest()))
        page=body.get('data',[])
        if not page:break
        oldest=min(int(r[0]) for r in page)
        if oldest>=after:raise ValueError('expanded_pagination_not_advancing')
        if max(int(r[0]) for r in page)>end*1000:raise ValueError('expanded_range_ignored')
        rows.extend(page);after=oldest
        if oldest<=start*1000 or len(page)<100:break
    else:raise ValueError('expanded_page_bound_exceeded')
    frame,gaps=parse(rows,kind,start,end)
    if not frame['time']:raise ValueError('empty_expanded_history')
    if gaps:raise ValueError('expanded_candle_gaps')
    payload=pickle.dumps(frame,protocol=pickle.HIGHEST_PROTOCOL);output=FOLDER/f'{symbol}_{kind}_15m.pkl'
    if output.exists() and output.read_bytes()!=payload:raise ValueError('immutable_expanded_frame_changed')
    if not output.exists():output.write_bytes(payload)
    return dict(status='OK',path=str(output),sha256=hashlib.sha256(payload).hexdigest(),bars=len(frame['time']),
        start=frame['time'][0],end=frame['time'][-1],requested_start=start,requested_end=end,
        partial_start=frame['time'][0]>start,partial_end=frame['time'][-1]+900<end,pages=pages)


def main():
    FOLDER.mkdir(parents=True,exist_ok=True);snapshot=json.loads(SNAPSHOT.read_text(encoding='utf-8'))
    proposed=choose(snapshot)
    if SELECTION.exists():
        selection=json.loads(SELECTION.read_text(encoding='utf-8'))
        if selection!=proposed:raise ValueError('expanded_selection_source_changed')
    else:
        selection=proposed;SELECTION.write_text(json.dumps(selection,indent=2),encoding='utf-8')
    source=Public();current=strict_instruments(source.get('/api/v5/public/instruments',{'instType':'FUTURES'})['data'])
    for symbol,item in selection['selected'].items():
        if current.get(symbol[:-4],{}).get('instId')!=item['instrument']:raise ValueError('expanded_contract_rolled_or_unavailable')
    manifest=dict(selection=selection,selection_sha256=hashlib.sha256(SELECTION.read_bytes()).hexdigest(),symbols={},complete=False)
    print('expanded_selected',[(s,v['category']) for s,v in selection['selected'].items()],flush=True)
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures={pool.submit(fetch,s,item,kind,selection,source):(s,kind) for s,item in selection['selected'].items() for kind in item['kinds']}
        for future in as_completed(futures):
            symbol,kind=futures[future]
            try:value=future.result()
            except Exception as exc:value=dict(status='ERROR',reason=str(exc),error_type=type(exc).__name__)
            manifest['symbols'].setdefault(symbol,dict(instrument=selection['selected'][symbol]['instrument'],
                category=selection['selected'][symbol]['category'],spec=selection['selected'][symbol]['spec']))[kind]=value
            (FOLDER/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
            print('expanded_history',symbol,kind,value['status'],value.get('bars'),value.get('reason'),flush=True)
    manifest['complete']=all(manifest['symbols'].get(s,{}).get(kind,{}).get('status')=='OK'
        for s,item in selection['selected'].items() for kind in item['kinds'])
    (FOLDER/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print('expanded_complete',manifest['complete'],flush=True)


if __name__=='__main__':main()
