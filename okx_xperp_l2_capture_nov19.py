"""Capture daily OKX L2 files for all frozen proposals in a pilot session."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import time
from urllib.parse import urlsplit

import requests

from account_cash_guard_nov17 import stamp
from shared_admission_replay_nov19 import ROOT, records


ROOT_L2 = ROOT / 'okx_xperp_l2_nov19'
CATALOG = 'https://www.okx.com/api/v5/public/market-data-history'
DAY = '2026-11-13'
START = int(stamp(DAY))
END = START + 86400
LIQUIDITY_COMPARATORS = ('NVDAUSDT', 'INTCUSDT', 'SPCXUSDT', 'SOXLUSDT')


def cohort():
    crypto, ready, stocks, specs, _ = records()
    entry_rows = [r for r in crypto + stocks if START <= r['entry_time'] < END]
    universe = set(LIQUIDITY_COMPARATORS) | {r['symbol'] for r in entry_rows}
    instruments = {s: specs[s]['instId'] for s in sorted(universe)}
    readiness = {(r['symbol'], r['entry_time']) for r in ready}
    return entry_rows, readiness, instruments


def validate_file(family, record, *, day_start=START):
    day = datetime.fromtimestamp(day_start,timezone.utc).date().isoformat()
    name = family + f'-futureschain-L2orderbook-400lv-{day}.tar.gz'
    url = urlsplit(record['url'])
    if (name != record['filename'] or url.scheme != 'https' or url.netloc != 'static.okx.com'
            or url.path != '/cdn/okx/match/orderbook/pro/L2/400lv/daily/'
                            + day.replace('-', '') + '/' + name
            or url.query or url.fragment or int(record['dateTs']) != 1000 * day_start):
        raise ValueError('unexpected OKX archive URL or date')
    size = float(record['sizeMB'])
    if not math.isfinite(size) or not 0 < size <= 500:
        raise ValueError('archive outside declared pilot size range')
    return name, math.ceil((1.24 * size + 2) * 1024**2)


def catalogue(families, *, day_start=START):
    files = {}
    for i in range(0, len(families), 5):
        batch = families[i:i+5]
        query = dict(module=4, instType='FUTURES', instFamilyList=','.join(batch),
                     dateAggrType='daily', begin=1000 * day_start, end=1000 * day_start)
        response = requests.get(CATALOG, params=query, timeout=12)
        response.raise_for_status()
        body = response.json()
        if body['code'] != '0':
            raise ValueError(f'OKX catalogue code {body["code"]}')
        for data in body['data']:
            for group in data['details']:
                family = group['instFamily']
                items = group['groupDetails']
                if family not in batch or family in files or len(items) != 1:
                    raise ValueError('unexpected catalogue group')
                validate_file(family, items[0],day_start=day_start)
                files[family] = items[0]
        time.sleep(.65)
    return files


def download(family, record, *, day_start=START):
    name, limit = validate_file(family, record,day_start=day_start)
    target = ROOT_L2 / name
    digest = hashlib.sha256()
    count = 0
    if target.exists():
        with target.open('rb') as stream:
            for content in iter(lambda: stream.read(512 * 1024), b''):
                count += len(content)
                digest.update(content)
    else:
        temp = target.with_suffix('.gz.part')
        with requests.get(record['url'], stream=True, timeout=(10, 35), allow_redirects=False) as response:
            if response.status_code != 200:
                raise ValueError(f'download HTTP {response.status_code}')
            with temp.open('wb') as stream:
                for content in response.iter_content(chunk_size=512 * 1024):
                    count += len(content)
                    if count > limit:
                        raise ValueError('download exceeds catalogue bound')
                    digest.update(content)
                    stream.write(content)
        if count <= 1024:
            raise ValueError('download unexpectedly small')
        temp.replace(target)
    if not 1024 < count <= limit:
        raise ValueError('cached download size outside bounds')
    return dict(path=str(target), sha256=digest.hexdigest(), bytes=count, catalogue=record)


def main():
    rows, _, instruments = cohort()
    families = {s: value.rsplit('-', 1)[0] for s, value in instruments.items()}
    ROOT_L2.mkdir(parents=True, exist_ok=True)
    files = catalogue(sorted(set(families.values())))
    manifest = dict(day=DAY, requested_from=datetime.now(timezone.utc).isoformat(),
                    source=CATALOG, proposed_count=len(rows), files={})
    with ThreadPoolExecutor(max_workers=4) as pool:
        tasks = {}
        for symbol, family in families.items():
            base = dict(instrument=instruments[symbol], family=family)
            manifest['files'][symbol] = base
            if family in files:
                tasks[pool.submit(download, family, files[family])] = symbol
            else:
                base['error'] = 'no catalogue files for requested day'
                print(symbol, base['error'], flush=True)
        for future in as_completed(tasks):
            symbol = tasks[future]
            try:
                manifest['files'][symbol].update(future.result())
            except Exception as exc:
                manifest['files'][symbol]['error'] = f'{type(exc).__name__}: {exc}'
            item = manifest['files'][symbol]
            print(symbol, item.get('bytes', item.get('error')), flush=True)
    manifest['complete'] = all('path' in item for item in manifest['files'].values())
    (ROOT_L2 / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print('complete', manifest['complete'], len(manifest['files']), flush=True)


if __name__ == '__main__':
    main()
