"""Test BTC-adjusted crypto continuation and snapback with fixed cash constraints."""
import hashlib
import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from liquidity_trap_joint_nov19_audit import load_frame, with_exact_tick, attach_outcome, verified_cash
from phase_scenarios_nov19_audit import ROOT, CRYPTO_MANIFEST, frame_of, TP1_POLICIES
from account_cash_guard_nov17_audit import runtime_gate
from account_cash_guard_nov17 import stamp
from price_grid_bracket_nov19 import rounded_bracket

END_TRAIN = stamp('2026-10-01')
START = stamp('2026-08-01')
FAMILIES = ('continuation', 'snapback')
HORIZON = 32


def setup_at(prices, index):
    """Closed return estimates use 64 preceding bars, then a separate impulse."""
    asset, btc = prices
    if index < 69 or not (np.isfinite(asset[index-69:index+1]).all()
                          and np.isfinite(btc[index-69:index+1]).all()):
        return None
    a = np.diff(asset[index-68:index+1]); b = np.diff(btc[index-68:index+1])
    x, y = b[:-4], a[:-4]
    variance = np.dot(x-x.mean(), x-x.mean())
    if variance <= 1e-12:
        return None
    beta = float(np.clip(np.dot(x-x.mean(), y-y.mean())/variance, 0., 3.))
    residual_history = y-beta*x
    sigma = float(residual_history.std())
    if sigma <= 1e-6:
        return None
    residual = a[-4:]-beta*b[-4:]-residual_history.mean()
    z = float(residual.sum()/(2*sigma))
    trend = float(b[-16:].sum())
    if abs(z) < 1.75:
        return None
    sign = 1 if z > 0 else -1
    if trend*sign >= .004 and residual[-1]*sign > 0:
        return 'continuation', sign, abs(z), beta
    if abs(trend) <= .0025 and residual[-1]*sign <= -.35*sigma:
        return 'snapback', -sign, abs(z), beta
    return None


def proposals(symbol, candle, benchmark, spec):
    t = np.asarray(candle['time'], dtype=np.int64)
    if len(t)<102 or (np.diff(t)!=900).any():
        raise ValueError('noncontinuous_relative_move_history')
    log_close = np.log(np.asarray(candle['close'], dtype=float))
    btc_close = np.asarray([benchmark.get(int(x), math.nan) for x in t])
    volume = np.asarray(candle['volume'], dtype=float)
    o, h, l, c = (np.asarray(candle[k], dtype=float) for k in ('open','high','low','close'))
    ranges = np.maximum.reduce((h[1:]-l[1:], abs(h[1:]-c[:-1]), abs(l[1:]-c[:-1])))
    records = []
    for index in range(69, len(t)-HORIZON):
        if volume[index] <= 0 or np.count_nonzero(volume[index-15:index+1]>0) < 12:
            continue
        if volume[index] < np.median(volume[index-16:index]):
            continue
        entry_time = int(t[index])+900
        if entry_time<START:
            continue
        result = setup_at((log_close, btc_close), index)
        if result is None:
            continue
        family, sign, score, beta = result
        if sign*(c[index]-o[index]) <= 0:
            continue
        atr = float(np.mean(ranges[index-14:index])); entry = float(c[index])*(1+sign*.0005)
        extreme = float(min(l[index-2:index+1]) if sign>0 else max(h[index-2:index+1]))
        risk = max(.003, sign*(entry-extreme)/entry + .1*atr/entry)
        cap = .01
        if risk > cap:
            continue
        raw = dict(symbol=symbol, family=family, direction='LONG' if sign>0 else 'SHORT',
                   entry=entry, sl=entry*(1-sign*risk), tp1=entry*(1+sign*.005),
                   tp2=entry*(1+sign*.015), risk_cap=cap, entry_time=entry_time,
                   entry_index=index+1, expire_at=None, signal_strength=score, beta=beta)
        row = rounded_bracket(raw, spec)
        if row is not None:
            records.append(row)
    return records


def assess(rows, specs, start, until=None):
    records = [dict(r, source_priority=-int(1000*r['signal_strength']), module_priority=0,
                    outcome=r.get('sequential_outcome','UNKNOWN')) for r in rows
               if r['entry_time']>=start and (until is None or r['entry_time']+HORIZON*900<until)]
    admitted = runtime_gate(records, latch_loss_pause=False)
    unknown = [r for r in admitted if not r['known']]
    result = dict(proposals=len(records), admitted=len(admitted),
                  unknown_reasons=dict(Counter(r['unknown_reason'] for r in unknown)))
    if unknown:
        return dict(result, status='UNSCORABLE_ADMITTED_UNKNOWN')
    crypto = frame_of(admitted)
    return dict(result, status='RESOLVED_PROXY_CASH', cash={
        policy: verified_cash(crypto, frame_of([]), specs, policy=policy, priority='crypto', start=start,
                              end_month='2026-09' if until else '2026-11')
        for policy in TP1_POLICIES})


def main():
    snapshot = ROOT/'native_universe_nov19_snapshot.json'
    manifest = json.loads(CRYPTO_MANIFEST.read_text(encoding='utf-8'))['symbols']
    spec_file = ROOT/'shared_xperp_specs_nov14.json'
    instruments = json.loads(snapshot.read_text(encoding='utf-8'))['instruments']
    specs = json.loads(spec_file.read_text(encoding='utf-8'))
    protocol_path = ROOT/'crypto_relative_move_nov19_protocol.json'
    sources = [Path(__file__).resolve(), snapshot, CRYPTO_MANIFEST, spec_file,
               Path('price_bracket_path_nov19.py'), Path('price_grid_bracket_nov19.py'),
               Path('account_cash_guard_nov17_audit.py'), Path('liquidity_trap_joint_nov19_audit.py'),
               Path('phase_scenarios_nov19_audit.py'), Path('cash_tp1_policy_nov19.py'),
               Path('src/okx_trader.py'), Path('src/tp1_plan.py')]
    fingerprints = {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    if protocol_path.exists():
        if json.loads(protocol_path.read_text())['sources'] != fingerprints:
            raise ValueError('relative_move_protocol_changed')
    else:
        protocol_path.write_text(json.dumps(dict(sources=fingerprints,
            declared_at=datetime.now(timezone.utc).isoformat(), families=FAMILIES,
            selection='August and September, both TP1 modes positive and at least 20 closed each month',
            later='October and partial November only if selection passes',
            asset_set='frozen base native crypto except BTC, which is the benchmark',
            account='130 USD, 3% margin, 10x, crypto cap 4 under shared cap 12 / 90%, stocks PAPER',
            historical_quotes=False, independent_holdout=False), indent=2), encoding='utf-8')
    btc = manifest['BTCUSDT']
    candle = load_frame(Path(btc['path']), btc['sha256'])
    benchmark = dict(zip(candle['time'], np.log(candle['close'])))
    rows = []
    for symbol,item in sorted(manifest.items()):
        if symbol == 'BTCUSDT':
            continue
        if specs[symbol]['instId']!=item['instrument']:
            raise ValueError('relative_move_wrong_contract')
        spec = with_exact_tick(specs[symbol], instruments)
        c = load_frame(Path(item['path']), item['sha256'])
        rows.extend(attach_outcome(r,c,True) for r in proposals(symbol,c,benchmark,spec))
    routes = {f:[r for r in rows if r['family']==f] for f in FAMILIES}
    routes['union'] = rows
    results = {key:assess(cohort, specs, START, END_TRAIN) for key,cohort in routes.items()}
    winners = []
    for key, result in results.items():
        periods = [x for p in result.get('cash',{}).values() for x in p['calendar_months'].values()]
        if len(periods)==4 and all(x['crypto_closed']>=20 and x['return_pct']>0 for x in periods):
            winners.append((min(x['return_pct']-x['cash_drawdown_pct'] for x in periods), key))
    chosen = max(winners)[1] if winners else None
    later = assess(routes[chosen], specs, END_TRAIN) if chosen else None
    output = dict(candidates=dict(Counter(row['family'] for row in rows)), selection=results,
                  chosen=chosen, later=later, live_activation_supported=False,
                  limits='Public last-trade candles, 5bp price stresses, no historical spread/depth or account equity replay.')
    (ROOT/'crypto_relative_move_nov19.json').write_text(json.dumps(output,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps(dict(candidates=output['candidates'],selection={k:[v['status'],v['admitted']] for k,v in results.items()},chosen=chosen)))


if __name__ == '__main__':
    main()
