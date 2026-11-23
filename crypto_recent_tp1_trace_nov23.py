"""Trace recent crypto TP1 excursions against confirmed public X-Perp candles."""

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

from okx_cross_bot_overlap_nov21 import declared_specs
from okx_xperp_multiday_capture_nov20 import file_hash
from public_candle_window import fetch_window
from src.crypto_profit_first_material_shadow_router import (
    ARTIFACT_PATH, REQUIRED_BTC_HISTORY_BARS, REQUIRED_SYMBOL_HISTORY_BARS, shadow_setups,
)
from src.okx_trader import calc_contracts
from src.tp1_plan import plan_tp1_close
from stock_liquid_quote_routes_nov21 import ROOT, write_json


STEM = 'crypto_recent_tp1_trace_nov23'
DATA = ROOT / STEM
CASES = (
    ('HYPEUSDT', 'SHORT', '22T05:30:12', 3.36, 3.36, None, '22T06:39:00', 'SL_HIT'),
    ('BTCUSDT', 'LONG', '22T10:00:13', -17.14, -21.44, '22T12:16:00', '22T12:30:10', 'BREAKEVEN'),
    ('ETHUSDT', 'LONG', '22T10:00:14', 0.82, -24.80, '22T12:18:00', '22T12:30:10', 'BREAKEVEN'),
    ('HYPEUSDT', 'SHORT', '23T00:00:13', 6.66, 4.44, '23T00:12:00', '23T02:28:00', 'TP2_HIT'),
    ('ZECUSDT', 'SHORT', '23T00:00:14', 8.17, 21.58, '23T00:19:00', '23T00:34:00', 'BREAKEVEN'),
    ('LINKUSDT', 'SHORT', '23T03:30:12', 1.13, 1.13, None, '23T05:46:00', 'SL_HIT'),
)
POLICIES = {'as_requested_5pct': (5, False), 'requested_25pct': (25, False),
            'half_with_minimum': (50, True), 'full_at_first_target': (100, False)}


def epoch(value):
    return int(datetime.fromisoformat(f'2026-11-{value}+00:00').timestamp())


def closed_context(inst, end, bars, sources):
    output = defaultdict(list)
    for start in range(end-bars*900, end, 86400):
        stop = min(start+86400, end)
        frame, digest = fetch_window(inst, start, stop, 900, DATA/'candles')
        sources[f'{inst}:15m:{start}:{stop}'] = digest
        for field, values in frame.items():
            output[field].extend(values)
    return dict(output, volume_contracts=output['volume'])


def execution_price_bounds(model, sign, vs_signal_bps, vs_model_bps):
    """The log rounds both relative prices to 0.01 bp. Bound that uncertainty."""
    actual = model*(1+sign*vs_model_bps/10000)
    actual_error = model*.005/10000
    ratio = 1+sign*vs_signal_bps/10000
    ratio_error = .005/10000
    return actual, ((actual-actual_error)/(ratio+ratio_error), (actual+actual_error)/(ratio-ratio_error))


def scan_case(case, specs):
    symbol, side, ack, vs_signal, vs_model, first, final, outcome = case
    start, end = epoch(ack), epoch(final)
    bucket = start//900*900
    sign = 1 if side == 'LONG' else -1
    spec, sources = specs[symbol], {}
    inst = spec['instId']
    btc = closed_context(specs['BTCUSDT']['instId'], bucket, REQUIRED_BTC_HISTORY_BARS, sources)
    own = closed_context(inst, bucket, REQUIRED_SYMBOL_HISTORY_BARS, sources)
    # The complete signal candle's open is the first reported trade, not its final close.
    opening, digest = fetch_window(inst, bucket, bucket+60, 60, DATA/'candles')
    sources[f'{inst}:1m:{bucket}:{bucket+60}'] = digest
    if opening['volume'][0] <= 0:
        raise ValueError('Missing trade in the forming opening minute')
    model = opening['open'][0]
    estimated_fill, (signal_lo, signal_hi) = execution_price_bounds(model, sign, vs_signal, vs_model)
    estimated_signal = (signal_lo+signal_hi)/2
    setup = shadow_setups(own, btc, entry_time=bucket, market_price=estimated_signal,
                          model_price=model, symbol=symbol)
    if len(setup) != 1 or setup[0]['direction'] != side:
        raise ValueError(f'Frozen model direction mismatch for {symbol} {ack}')
    setup = setup[0]
    risk = abs(setup['entry']-setup['sl'])
    # Omit the minute containing order acknowledgement and the minute containing
    # the observed model exit. Their OHLC alone does not give event order.
    own_start, own_end = (start//60+1)*60, end//60*60
    frame, digest = fetch_window(inst, own_start, own_end, 60, DATA/'candles')
    sources[f'{inst}:1m:{own_start}:{own_end}'] = digest
    target_edge = 'high' if sign == 1 else 'low'
    prices = [(t, p) for t, p, vol in zip(frame['time'], frame[target_edge], frame['volume'], strict=True) if vol > 0]
    if not prices:
        raise ValueError('No traded minute before observed model exit')
    maximum_at, maximum = max(prices, key=lambda p: sign*p[1])
    conservative_entry = signal_hi if sign == 1 else signal_lo
    best_r = sign*(maximum-conservative_entry)/risk
    exposure = calc_contracts(130*.03, 10, estimated_fill, spec)
    policies = {}
    for name, (pct, minimum) in POLICIES.items():
        result = plan_tp1_close(exposure, pct, spec, ensure_minimum=minimum)
        policies[name] = dict(contracts=result['size'], pct=result['actual_pct'])
    if first is not None:
        first_start = epoch(first)-60
        if not any(abs(t-first_start) <= 60 and sign*(p-conservative_entry)/risk >= .49 for t, p in prices):
            raise ValueError(f'Logged TP1 not reproduced to 0.01R on adjacent traded minutes for {symbol} {ack}')
    return dict(symbol=symbol, entry_scan=bucket, direction=side, model_outcome=outcome,
                model_first_observed_tp1=epoch(first) if first else None, signal_price_range=[signal_lo, signal_hi],
                model_assumed_forming_open=model, approximate_entry_fill=estimated_fill, initial_risk_price=risk,
                best_r_before_exit_minute=best_r, best_r_candle_open=maximum_at, simulated_contracts=exposure,
                modeled_tp1_quantities=policies, candle_sha256=sources)


def main():
    specs = declared_specs()
    with ThreadPoolExecutor(max_workers=3) as pool:
        rows = list(pool.map(lambda case: scan_case(case, specs), CASES))
    for row in rows:
        print(row['symbol'], row['model_outcome'], round(row['best_r_before_exit_minute'], 3),
              {k: round(v['pct'], 1) for k, v in row['modeled_tp1_quantities'].items()}, flush=True)
    reached = [r for r in rows if r['model_first_observed_tp1'] is not None]
    breakevens = [r for r in reached if r['model_outcome'] == 'BREAKEVEN']
    counts = dict(entry_count=len(rows), tp1_count=len(reached), breakeven_after_tp1=len(breakevens),
                  breakeven_before_one_r=sum(r['best_r_before_exit_minute'] < 1 for r in breakevens),
                  unbankable_5pct_after_tp1=sum(r['modeled_tp1_quantities']['as_requested_5pct']['contracts'] == 0
                                               for r in reached))
    write_json(ROOT/f'{STEM}.json', dict(rows=rows, capital_assumption_usd=130, margin_fraction=.03, leverage=10,
        counts=counts, live_activation_supported=False,
        sources={p.name: file_hash(p) for p in (Path(__file__), ARTIFACT_PATH)},
        limitations=[
            'Railway logged the six real entries and separate paper outcomes. Broker exits and private cash PnL were not retrieved.',
            'Estimated fill uses rounded live log ratios and assumes the forming-open reference matched the finalized public open.',
            'OHLC shows traded prices, not executable depth. It excludes the entry and exit minutes for the excursion calculation.',
            'TP1 sizing uses a common $130 balance assumption, not the actual personal account balance or user settings.',
        ]))


if __name__ == '__main__':
    main()
