"""Capture both OKX volume fields for a fixed cohort of X-Perp proposals."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import pickle
import time

from account_cash_guard_nov17 import stamp
from download_native_expansion_nov19 import Public, strict_instruments
from shared_admission_replay_nov19 import MANIFEST, ROOT, STOCK_MANIFEST, records


FOLDER = ROOT / 'paired_contract_volume_nov19'
START = int(stamp('2026-11-04'))
COHORT_START = int(stamp('2026-11-05'))
STEP = 900
ENDPOINT = '/api/v5/market/history-candles'


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def selection():
    crypto, _, stocks, _, end = records()
    manifests = {'crypto': MANIFEST, 'stocks': STOCK_MANIFEST}
    items = {}
    for bot, rows in (('crypto', crypto), ('stocks', stocks)):
        path = manifests[bot]
        manifest = json.loads(path.read_bytes())['symbols']
        for symbol in sorted({row['symbol'] for row in rows if row['entry_time'] >= COHORT_START}):
            item = manifest[symbol]
            items[symbol] = dict(bot=bot, instrument=item['instrument'],
                                 archive_sha256=item['sha256'], archive_path=item['path'])
    return dict(start=START, cohort_start=COHORT_START, end=end, symbols=items,
                manifests={str(path): digest(path.read_bytes()) for path in manifests.values()})


def parse(rows, start, end):
    fields = ('open', 'high', 'low', 'close', 'volume_contracts', 'volume_base', 'volume_quote')
    data = {}
    for row in rows:
        if len(row) != 9 or row[8] != '1':
            raise ValueError('expected confirmed native trade candles')
        millis = int(row[0])
        if millis % (STEP * 1000):
            raise ValueError('unaligned candle timestamp')
        at = millis // 1000
        if not start <= at < end:
            continue
        values = tuple(float(x) for x in row[1:8])
        opening, high, low, close, *volumes = values
        if (not all(math.isfinite(x) for x in values)
                or not 0 < low <= min(opening, close) <= max(opening, close) <= high
                or min(volumes) < 0):
            raise ValueError('invalid native OHLC or volume')
        if at in data and data[at] != values:
            raise ValueError('conflicting duplicate candle')
        data[at] = values
    times = sorted(data)
    if times != list(range(start, end, STEP)):
        raise ValueError(f'incomplete historical interval ({len(times)} bars)')
    return dict(time=times, **{key: [data[at][i] for at in times] for i, key in enumerate(fields)})


def fetch(public, symbol, instrument, start, end):
    folder = FOLDER / 'pages' / symbol
    folder.mkdir(parents=True, exist_ok=True)
    all_rows = []
    evidence = []
    after = end * 1000
    for _ in range(25):
        params = dict(instId=instrument, bar='15m', after=str(after), limit=300)
        page = folder / f'{after}.json'
        if page.exists():
            raw = page.read_bytes()
            payload = json.loads(raw)
            if payload['params'] != params:
                raise ValueError('cached page is for another instrument or interval')
        else:
            for attempt in range(3):
                try:
                    body = public.get(ENDPOINT, params)
                    break
                except Exception:
                    if attempt == 2:
                        raise
                    time.sleep(attempt + 1)
            payload = dict(params=params, response=body)
            raw = json.dumps(payload, separators=(',', ':')).encode()
            page.write_bytes(raw)
        evidence.append(dict(path=str(page), sha256=digest(raw)))
        if payload['response'].get('code') != '0':
            raise ValueError('OKX history error')
        rows = payload['response']['data']
        if not rows:
            break
        newest = max(int(row[0]) for row in rows)
        if newest >= after:
            raise ValueError('pagination cursor boundary ignored')
        all_rows.extend(rows)
        after = min(int(row[0]) for row in rows)
        if after <= start * 1000:
            break
    frame = parse(all_rows, start, end)
    path = FOLDER / f'{symbol}.pkl'
    raw = pickle.dumps(frame, protocol=5)
    if path.exists() and path.read_bytes() != raw:
        raise ValueError('decoded immutable history revised')
    if not path.exists():
        path.write_bytes(raw)
    return dict(path=str(path), sha256=digest(raw), bars=len(frame['time']), pages=evidence)


def main():
    FOLDER.mkdir(parents=True, exist_ok=True)
    declaration = FOLDER / 'selection.json'
    spec = dict(selection(), source_sha256=digest(Path(__file__).read_bytes()))
    if declaration.exists() and json.loads(declaration.read_bytes()) != spec:
        raise ValueError('declared capture selection or source changed')
    if not declaration.exists():
        declaration.write_text(json.dumps(spec, indent=2), encoding='utf-8')
    api = Public()
    expected = strict_instruments(api.get('/api/v5/public/instruments', {'instType': 'FUTURES'})['data'])
    manifest = dict(selection_sha256=digest(declaration.read_bytes()),
                    checked_at=datetime.now(timezone.utc).isoformat(), symbols={})
    with ThreadPoolExecutor(max_workers=4) as pool:
        jobs = {}
        for symbol, item in spec['symbols'].items():
            current = expected.get(symbol.removesuffix('USDT'), {})
            if current.get('instId') != item['instrument']:
                manifest['symbols'][symbol] = dict(error='declared instrument unavailable or rolled')
                continue
            if int(current['listTime']) > spec['start'] * 1000:
                manifest['symbols'][symbol] = dict(error='instrument listed after capture start')
                continue
            jobs[pool.submit(fetch, api, symbol, item['instrument'], spec['start'], spec['end'])] = symbol
        for job in as_completed(jobs):
            symbol = jobs[job]
            try:
                manifest['symbols'][symbol] = job.result()
            except Exception as exc:
                manifest['symbols'][symbol] = dict(error=f'{type(exc).__name__}: {exc}')
            result = manifest['symbols'][symbol]
            print(symbol, result.get('bars', result.get('error')), flush=True)
    manifest['complete'] = all('path' in row for row in manifest['symbols'].values())
    (FOLDER / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print('complete', manifest['complete'], len(manifest['symbols']), flush=True)


if __name__ == '__main__':
    main()
