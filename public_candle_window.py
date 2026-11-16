"""Immutable bounded confirmed OKX windows for execution-timing research."""
import hashlib
import json
import math
import time
from pathlib import Path
import requests

FIELDS=('open','high','low','close','volume')


def validate_rows(rows,start,end,seconds):
    found={}
    for raw in rows:
        if not isinstance(raw,list) or len(raw)<9 or raw[8]!='1':
            continue
        stamp=int(raw[0])//1000
        if not start<=stamp<end:
            continue
        values=tuple(float(v) for v in raw[1:6])
        o,h,l,c,v=values
        if (stamp%seconds or not all(math.isfinite(x) for x in values)
                or min(o,h,l,c)<=0 or v<0 or not l<=min(o,c)<=max(o,c)<=h):
            raise ValueError('invalid confirmed candle')
        if stamp in found and found[stamp]!=values:
            raise ValueError('conflicting confirmed candle revision')
        found[stamp]=values
    if sorted(found)!=list(range(start,end,seconds)):
        raise ValueError('incomplete confirmed candle window')
    return dict(time=list(range(start,end,seconds)),**{
        key:[found[t][i] for t in range(start,end,seconds)] for i,key in enumerate(FIELDS)})


def fetch_window(inst,start,end,seconds,folder):
    if seconds not in (1,60) or not isinstance(start,int) or not isinstance(end,int):
        raise ValueError('unsupported window')
    if not 0<end-start<=86400 or start%seconds or end%seconds:
        raise ValueError('unbounded or unaligned window')
    bar='1s' if seconds==1 else '1m'
    path=Path(folder)/f'{inst}_{bar}_{start}_{end}.json'
    if path.exists():
        body=json.loads(path.read_text(encoding='utf-8'))
    else:
        rows=[];after=end*1000
        for _ in range(math.ceil((end-start)/seconds/300)+2):
            response=requests.get('https://www.okx.com/api/v5/market/history-candles',
                params=dict(instId=inst,bar=bar,after=after,limit=300),timeout=15)
            response.raise_for_status();body=response.json()
            if body.get('code')!='0':raise ValueError('public candle request refused')
            batch=body.get('data',[])
            if not batch:break
            rows.extend(batch);oldest=min(int(x[0]) for x in batch)
            if oldest<=start*1000:break
            if oldest>=after:raise ValueError('pagination did not advance')
            after=oldest;time.sleep(.15)
        body=dict(code='0',data=rows)
        validate_rows(rows,start,end,seconds)  # do not freeze an incomplete download
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(body),encoding='utf-8')
    return validate_rows(body['data'],start,end,seconds),hashlib.sha256(path.read_bytes()).hexdigest()
