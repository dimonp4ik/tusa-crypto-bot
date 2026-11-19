"""Download public OKX trade tapes for both bots' fixed L2 pilot day."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
import hashlib
import json
import re
from pathlib import Path
import time
from urllib.parse import urlsplit

import requests

from okx_xperp_l2_capture_nov19 import CATALOG, END, ROOT_L2, START, cohort


TRADE_ROOT = ROOT_L2 / 'trades'
TRADE_TZ = timezone(timedelta(hours=8))
DAY0 = datetime.fromtimestamp(START, TRADE_TZ).replace(hour=0, minute=0, second=0, microsecond=0)
DAY1 = datetime.fromtimestamp(END + 57600 - 1, TRADE_TZ).replace(hour=0, minute=0, second=0, microsecond=0)
TRADE_DAYS = {int(day.timestamp()*1000): day.date().isoformat() for day in (DAY0, DAY1)}


def get_catalogue(families):
    result = {}
    for i in range(0, len(families), 5):
        batch = families[i:i+5]
        query = dict(module=1,instType='FUTURES',instFamilyList=','.join(batch),
                     dateAggrType='daily',begin=min(TRADE_DAYS),end=max(TRADE_DAYS))
        response = requests.get(CATALOG,params=query,timeout=12)
        response.raise_for_status()
        body = response.json()
        if body.get('code') != '0':
            raise ValueError(f'OKX trade catalogue code {body.get("code")}')
        for page in body['data']:
            for group in page['details']:
                family = group['instFamily']
                if family not in batch or family in result:
                    raise ValueError('catalogue family mismatch')
                result[family] = group['groupDetails']
        time.sleep(.65)
    return result


def fetch_family(job):
    symbol, instrument, files = job
    family = instrument.rsplit('-',1)[0]
    found = {}
    for item in files:
        stamp = int(item['dateTs'])
        day = TRADE_DAYS.get(stamp)
        if day is None or day in found:
            raise ValueError('unexpected date in trade catalogue')
        name = family + f'-futureschain-trades-{day}.zip'
        url = urlsplit(item['url'])
        route = '/cdn/okex/traderecords/trades/daily/' + day.replace('-', '') + '/' + name
        if (item['filename'] != name or url.scheme != 'https' or url.netloc != 'static.okx.com'
                or url.path != route or (url.query and not re.fullmatch(r'v=\d{1,8}', url.query)) or url.fragment):
            raise ValueError('unexpected trade archive URL')
        limit = 32*1024**2
        path = TRADE_ROOT/name
        if path.exists():
            if not 100 <= path.stat().st_size <= limit:
                raise ValueError('unexpected cached ZIP size')
            raw = path.read_bytes()
        else:
            data = bytearray()
            with requests.get(item['url'],stream=True,timeout=(10,25),allow_redirects=False) as response:
                if response.status_code != 200:
                    raise ValueError(f'trade file HTTP {response.status_code}')
                for chunk in response.iter_content(chunk_size=256*1024):
                    data.extend(chunk)
                    if len(data)>limit:
                        raise ValueError('trade archive exceeds pilot bound')
            raw = bytes(data)
            if len(raw)<100 or not raw.startswith(b'PK'):
                raise ValueError('trade archive is not ZIP')
            temp = path.with_suffix('.zip.part')
            temp.write_bytes(raw)
            temp.replace(path)
        found[day] = dict(path=str(path),sha256=hashlib.sha256(raw).hexdigest(),
                          utc_start_ms=stamp,bytes=len(raw),catalogue=item)
    return symbol,dict(instrument=instrument,days=found,complete=set(found)==set(TRADE_DAYS.values()))


def main():
    rows,_,instruments=cohort()
    symbols=sorted({row['symbol'] for row in rows})
    families=[instruments[symbol].rsplit('-',1)[0] for symbol in symbols]
    catalogue=get_catalogue(families)
    TRADE_ROOT.mkdir(parents=True,exist_ok=True)
    jobs=[(symbol,instruments[symbol],catalogue.get(family,[])) for symbol,family in zip(symbols,families)]
    with ThreadPoolExecutor(max_workers=4) as pool:
        entries=dict(pool.map(fetch_family,jobs))
    report=dict(endpoint=CATALOG,module=1,trade_days_utc8=list(TRADE_DAYS.values()),
                files=entries,complete=all(item['complete'] for item in entries.values()))
    (TRADE_ROOT/'manifest.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('trade capture',report['complete'],len(entries),sum(len(item['days']) for item in entries.values()),flush=True)


if __name__=='__main__':main()
