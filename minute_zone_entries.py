"""Minute-level reconstruction of watched-zone entry timing on a bounded sample."""
import csv,json,time,hashlib
from pathlib import Path
import requests


def fetch_minutes(inst,start,end,folder):
    path=folder/f'{inst}_{start}_{end}.json'
    if path.exists():body=json.loads(path.read_text())
    else:
        allrows=[];after=end*1000
        for page in range((end-start)//18000+3):
            response=requests.get('https://www.okx.com/api/v5/market/history-candles',
              params={'instId':inst,'bar':'1m','after':after,'limit':300},timeout=15)
            response.raise_for_status();part=response.json()
            if part.get('code')!='0':raise ValueError(str(part))
            batch=part.get('data',[])
            if not batch:break
            allrows.extend(batch);oldest=min(int(x[0]) for x in batch)
            if oldest<=start*1000:break
            if oldest>=after:raise ValueError('Pagination did not advance')
            after=oldest;time.sleep(.12)
        body={'code':'0','data':allrows}
        path.write_text(json.dumps(body),encoding='utf-8')
    data={int(x[0])//1000:[float(v) for v in x[1:5]] for x in body['data'] if x[-1]=='1' and start<=int(x[0])//1000<end}
    if sorted(data)!=list(range(start,end,60)):raise ValueError('Incomplete confirmed minute window')
    return data,hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    folder=Path('reports/audit_2026_09_08');cache=folder/'entry_minute_cache';cache.mkdir(exist_ok=True)
    with (folder/'venue_swap_tagged_raw.csv').open() as f:rows=list(csv.DictReader(f))
    chosen=[]
    for symbol in ('BTCUSDT','ETHUSDT'):
        chosen.extend([r for r in rows if r['symbol']==symbol and r['entry_is_intrabar']=='True' and float(r['entry_time'])>=1785542400][:6])
    results=[]
    for row in chosen:
        start=int(float(row['entry_time']));end=start+960;base=row['symbol'][:-4]
        item={'source_row_index':next(i for i,r in enumerate(rows) if r is row),'symbol':row['symbol'],'signal_bar':start,'planned_entry':float(row['entry'])}
        try:
            source,sh=fetch_minutes(base+'-USDT-SWAP',start,end,cache)
            actual,xh=fetch_minutes(base+'-USD_UM_XPERP-310404',start,end,cache)
            level=float(row['entry']);sign=1 if row['direction']=='LONG' else -1
            triggers=[t for t in range(start,start+900,60) if source[t][2]<=level<=source[t][1]]
            if not triggers:raise ValueError('No minute-level level touch')
            trigger=triggers[0];fill=trigger+60;sp=source[fill][0];xp=actual[fill][0]
            item.update(trigger_minute=trigger,modeled_fill_time=fill,swap_fill=sp,xperp_fill=xp,
                adverse_basis_bps=sign*(xp/sp-1)*10000,
                adverse_vs_planned_bps=sign*(xp/level-1)*10000,source_sha256=sh,xperp_sha256=xh,
                within_bracket=(float(row['sl'])<xp<float(row['tp1']) if sign==1 else float(row['tp1'])<xp<float(row['sl'])))
        except Exception as exc:item['error']=str(exc)
        results.append(item);print(json.dumps(item),flush=True)
    report={'status':'ENTRY_TIMING_DIAGNOSTIC','selection':'First six intrabar entries per BTC/ETH after Aug 1, selected without outcomes.',
      'assumption':'First confirmed minute range containing the modeled entry on SWAP, market fill at next minute open (up to one minute assumed delay).',
      'limitations':['Not exact zone-range/polling reconstruction; exact live trigger timestamps not exported.',
       'Historical opens are not executable bid/ask. No profitability claim; exits and funding not replayed.'],
      'results':results,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (folder/'minute_zone_entry_review.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
if __name__=='__main__':main()
