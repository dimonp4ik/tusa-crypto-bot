"""Measure the last closed mark candle against qualified stock quote prices."""

from concurrent.futures import ThreadPoolExecutor
from math import isfinite

import requests

from okx_unbankable_tp1_nov21 import read_json
from okx_xperp_multiday_replay_nov20 import ROOT, file_hash
from stock_liquid_quote_routes_nov21 import write_json


WIDE = 'two_quotes_up_to_15bp_with_2r'
MARK_HISTORY = 'https://www.okx.com/api/v5/market/history-mark-price-candles'


def previous_closed_mark(response, entry_ms):
    last_open = entry_ms // 60_000 * 60_000 - 60_000
    rows = response.get('data') or []
    if response.get('code') != '0' or len(rows) != 1:
        return None
    row = rows[0]
    try:
        if len(row) != 6 or row[-1] != '1' or int(row[0]) != last_open:
            return None
        value = float(row[4])
    except (TypeError, ValueError):
        return None
    return value if value > 0 and isfinite(value) else None


def measure(row):
    stamp = row['entry_ms']
    response = requests.get(MARK_HISTORY, timeout=12, params=dict(
        instId=row['instrument'], bar='1m', after=stamp//60_000*60_000, limit=1))
    response.raise_for_status()
    mark_row = response.json()
    mark = previous_closed_mark(mark_row, stamp)
    sign = 1 if row['direction'] == 'LONG' else -1
    basis = sign*(row['entry_px']/mark-1)*10_000 if mark is not None else None
    return dict(row, mark=mark, adverse_mark_basis_bps=basis, source=mark_row)


def main():
    older = ROOT / 'stock_contract_spread_holdout_nov23.json'
    later = ROOT / 'stock_spread_shared_nov23.json'
    recovery = ROOT / 'stock_spread_recovery_nov23.json'
    prior_spec = ROOT / older.stem / 'capture.json'
    recent_spec = ROOT / 'stock_native_geometry_match_nov22' / 'capture.json'
    fills = read_json(older)['result']['fills']
    specs = read_json(prior_spec)['specs']
    current = {(r['symbol'], r['epoch']) for r in fills['current']}
    rows = [dict(symbol=r['symbol'], instrument=specs[r['symbol']]['instId'], entry_ms=r['entry_ms'],
                 entry_px=r['reference_entry'], direction=r['direction'], path=r['path']['60']['code'],
                 cohort='holdout_current' if (r['symbol'], r['epoch']) in current else 'holdout_added') for r in fills[WIDE]]
    later_data = read_json(later)
    specs = read_json(recent_spec)['specs']
    directions = {(r['symbol'], r['entry_ms']): r['direction'] for r in read_json(recovery)['attempts'][WIDE]
                  if r['entry_status'] == 'eligible'}
    for cohort in ('nov20', 'nov21'):
        for row in later_data['cases'][cohort]['results']['middle']['200'][WIDE]['closed_stock_positions']:
            symbol = row['symbol']
            rows.append(dict(symbol=symbol, instrument=specs[symbol]['instId'], entry_ms=row['entry_ms'],
                entry_px=row['entry_px'], direction=directions[symbol, row['entry_ms']], path=row['code'], cohort=cohort))
    with ThreadPoolExecutor(max_workers=3) as pool:
        measured = list(pool.map(measure, rows))
    write_json(ROOT / 'stock_spread_mark_basis_nov23.json', dict(
        observations=measured,
        inputs={p.relative_to(ROOT).as_posix(): file_hash(p) for p in (older, later, recovery, prior_spec, recent_spec)},
        limitations=[
            'Older paths are qualified proposals before simultaneous slot and capacity decisions.',
            'The mark close is not the executable entry quote and can differ in time by up to a minute.',
            'This is a diagnostic of a previously analyzed group, not a test of a deployed gate.',
        ]))


if __name__ == '__main__':
    main()
