"""Bounded public trade evidence, reconciled to a confirmed one-second candle."""
import hashlib
import json
import math
from pathlib import Path
import re
import time

import requests


def near(a, b):
    return math.isclose(float(a), float(b), rel_tol=1e-8, abs_tol=1e-10)


def validate_second_trades(rows, start, expected, *, instrument=None):
    found = {}
    for row in rows:
        stamp, identity = int(row['ts']), int(row['tradeId'])
        price, size = float(row['px']), float(row['sz'])
        if (not start * 1000 <= stamp < (start + 1) * 1000 or identity < 0
                or not math.isfinite(price) or not math.isfinite(size) or min(price, size) <= 0
                or (instrument is not None and row.get('instId') != instrument)):
            raise ValueError('invalid public trade')
        value = (stamp, identity, price, size)
        if identity in found and found[identity] != value:
            raise ValueError('conflicting public trade revision')
        found[identity] = value
    ordered = sorted(found.values())
    if not near(sum(r[3] for r in ordered), expected['volume']):
        raise ValueError('public trade and second contract volume disagree')
    if not ordered:
        if expected['volume'] != 0:
            raise ValueError('missing public trades')
        return []
    ohlc = (ordered[0][2], max(r[2] for r in ordered), min(r[2] for r in ordered), ordered[-1][2])
    if not all(near(a, expected[k]) for a, k in zip(ohlc, ('open', 'high', 'low', 'close'))):
        raise ValueError('public trades and second OHLC disagree')
    return ordered


def fetch_second_trades(instrument, start, expected, folder):
    if (not isinstance(start, int) or isinstance(start, bool) or start + 1 > time.time()
            or not re.fullmatch(r'[A-Z0-9_-]{1,96}', instrument)):
        raise ValueError('invalid closed trade second')
    path = Path(folder) / f'{instrument}_trades_{start}.json'
    if path.exists():
        body = json.loads(path.read_text(encoding='utf-8'))
    else:
        params = dict(instId=instrument, type='2', after=str((start + 1) * 1000), limit=100)
        found = {}
        previous = None
        for _ in range(4):
            time.sleep(.12)
            response = requests.get('https://www.okx.com/api/v5/market/history-trades',
                                    params=params, timeout=15)
            response.raise_for_status()
            answer = response.json()
            if answer.get('code') != '0':
                raise ValueError('public trade request refused')
            batch = answer.get('data', [])
            if not batch:
                raise ValueError('public trade coverage unavailable')
            oldest_id = min(int(r['tradeId']) for r in batch)
            earliest = min(int(r['ts']) for r in batch)
            if previous is not None and oldest_id >= previous:
                raise ValueError('public trade pagination did not advance')
            for row in batch:
                stamp = int(row['ts'])
                if stamp >= (start + 1) * 1000:
                    raise ValueError('public trade beyond requested end')
                if stamp >= start * 1000:
                    identity = str(row['tradeId'])
                    if identity in found and found[identity] != row:
                        raise ValueError('conflicting public trade revision')
                    found[identity] = row
            if earliest < start * 1000:
                body = dict(instrument=instrument, start=start, covered_before_ms=earliest,
                            volume_unit='contracts', data=list(found.values()))
                validate_second_trades(body['data'], start, expected, instrument=instrument)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(body), encoding='utf-8')
                break
            previous = oldest_id
            params = dict(instId=instrument, type='1', after=str(oldest_id), limit=100)
        else:
            raise ValueError('public trade coverage limit reached')
    if (body.get('instrument') != instrument or body.get('start') != start
            or body.get('volume_unit') != 'contracts'
            or not isinstance(body.get('covered_before_ms'), int)
            or body['covered_before_ms'] >= start * 1000):
        raise ValueError('unproven public trade coverage')
    result = validate_second_trades(body['data'], start, expected, instrument=instrument)
    return result, hashlib.sha256(path.read_bytes()).hexdigest()


def validate_minute_seconds(parent, seconds, start):
    if seconds['time'] != list(range(start, start + 60)):
        raise ValueError('incomplete second minute')
    live = [i for i, volume in enumerate(seconds['volume']) if volume > 0]
    ohlc = ((seconds['open'][live[0]], max(seconds['high'][i] for i in live),
             min(seconds['low'][i] for i in live), seconds['close'][live[-1]])
            if live else (seconds['close'][0],) * 4)
    if not all(near(value, parent[key]) for value, key in zip(ohlc, ('open', 'high', 'low', 'close'))):
        raise ValueError('minute and traded seconds OHLC disagree')
    if not near(sum(seconds['volume']), parent['volume']):
        raise ValueError('minute and seconds contract volume disagree')
