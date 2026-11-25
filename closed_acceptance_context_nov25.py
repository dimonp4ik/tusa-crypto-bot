"""Test closed-bar price acceptance and peer pressure with frozen OKX entries."""

from collections import Counter
import math
from pathlib import Path
from statistics import median
from unittest.mock import patch

import cash_loss_pause_audit_nov24 as fresh_data
import first_print_confirmation_nov25 as entry_data
from okx_cross_bot_overlap_nov21 import closed
from okx_post_close_cash_nov23 import trade_difference
from okx_post_stop_reclaim_nov23 import with_actual_fill_path
from okx_selective_single_target_nov23 import EARLIER, cash_summary, key
from okx_spot_confirmation_nov22 import closed_index, load_native
from okx_unbankable_tp1_nov21 import old_inputs, read_json
from okx_xperp_cash_tick_nov19 import CLOSE_LAGS, simulate
from okx_xperp_multiday_replay_nov20 import ROOT, file_hash
from src.crypto_profit_first_material_shadow_router import (
    REQUIRED_BTC_HISTORY_BARS, REQUIRED_SYMBOL_HISTORY_BARS, shadow_setups,
)
from src.crypto_venue_router import ROBUST_SYMBOLS
from stock_liquid_quote_routes_nov21 import write_json


POLICIES = ('breakout_shallow_cross', 'breakout_weak_close',
            'breakout_weak_acceptance', 'breakout_peer_opposition',
            'stock_shallow_rejection', 'joint_pressure_confirmation')
LOCATION = .65
CUSHION_ATR = .10
REJECTION = .25
OPPOSED_FRACTION = 2 / 3
OPPOSED_ATR = .25


def prior_atr(frame, i):
    high, low, close = (frame[name] for name in ('high', 'low', 'close'))
    return math.fsum(max(high[j] - low[j], abs(high[j] - close[j-1]), abs(low[j] - close[j-1]))
                     for j in range(i-15, i-1)) / 14


class ClosedContext:
    def __init__(self, frames):
        self.frames = frames
        self.volume = {
            symbol: frame['volume'] if 'volume' in frame else frame['volume_contracts']
            for symbol, frame in frames.items()
        }
        self.cohorts = {
            bot: {s: f for s, f in frames.items() if (s in ROBUST_SYMBOLS) == (bot == 'crypto')}
            for bot in ('crypto', 'stocks')
        }
        self.moves = {}

    def market(self, bot, epoch):
        ident = bot, epoch
        if ident not in self.moves:
            moves = {}
            for symbol, frame in self.cohorts[bot].items():
                index = closed_index(frame, epoch, 16)
                if index is None or min(self.volume[symbol][index-2:index]) <= 0:
                    continue
                atr = prior_atr(frame, index)
                if atr > 0:
                    prices = frame['close']
                    moves[symbol] = (prices[index-1] - prices[index-2]) / atr
            self.moves[ident] = moves
        return self.moves[ident]

    def features(self, row):
        symbol, epoch, bot = (row[name] for name in ('symbol','epoch','bot'))
        frame = self.frames.get(symbol)
        i = closed_index(frame, epoch, 21 if bot == 'crypto' else 2)
        if i is None or self.volume[symbol][i-1] <= 0:
            return None
        high, low, close = (frame[name] for name in ('high','low','close'))
        top, bottom, final = high[i-1], low[i-1], close[i-1]
        if not bottom <= final <= top:
            raise ValueError('closed price lies outside its candle')
        side = 1 if row['direction'] == 'LONG' else -1
        location = (final-bottom)/(top-bottom) if top>bottom else .5
        if side < 0:
            location = 1-location
        cushion = None
        if bot == 'crypto' and row['family'] == 'breakout':
            boundary = max(high[i-21:i-1]) if side == 1 else min(low[i-21:i-1])
            atr = prior_atr(frame, i)
            if atr > 0 and side*(final-boundary) > 0:
                cushion = side*(final-boundary)/atr
        peers = [side*move for s,move in self.market(bot,epoch).items() if s != symbol]
        known = len(peers) >= max(6, (len(self.cohorts[bot])-1)//2)
        return dict(directional_close=location, breakout_cushion_atr=cushion, peers=len(peers),
                    peer_known=known, peer_median_atr=median(peers) if known else None,
                    peer_opposed_fraction=sum(move<0 for move in peers)/len(peers) if known else None)


def skip(name, row, features):
    if features is None:
        return False
    loc = features['directional_close']
    cushion = features['breakout_cushion_atr']
    stock = row['bot'] == 'stocks'
    shallow = cushion is not None and cushion < CUSHION_ATR
    weak_close = cushion is not None and loc < LOCATION
    weak_reversal = loc < REJECTION
    opposed = (features['peer_known'] and features['peer_median_atr'] <= -OPPOSED_ATR
               and features['peer_opposed_fraction'] >= OPPOSED_FRACTION)
    if name == 'breakout_shallow_cross':
        return shallow
    if name == 'breakout_weak_close':
        return weak_close
    if name == 'breakout_weak_acceptance':
        return shallow and weak_close
    if name == 'breakout_peer_opposition':
        return cushion is not None and opposed
    if name == 'stock_shallow_rejection':
        return stock and weak_reversal
    if name == 'joint_pressure_confirmation':
        return opposed and (weak_reversal if stock or row['family']=='range_reversion' else shallow or weak_close)
    raise ValueError('unknown acceptance policy')


def mark_old_family(rows):
    from shared_admission_replay_nov19 import records

    crypto, *_ = records()
    family = {(r['symbol'], int(r['entry_time']), r['direction']): r for r in crypto}
    for row in rows:
        if row['bot'] != 'crypto' or row['entry_status'] != 'eligible':
            continue
        candidate = family[row['symbol'], row['epoch'], row['direction']]
        if not all(math.isclose(candidate[source], row[target], rel_tol=1e-12)
                   for source,target in (('entry','entry'),('sl','stop'))):
            raise ValueError('older market entry does not match the original family')
        row['family'] = candidate['family']


def label_family(row, frames):
    if row['bot'] != 'crypto' or 'family' in row:
        return row
    symbol, stamp = row['symbol'], row['epoch']
    setups = shadow_setups(closed(frames[symbol], stamp, REQUIRED_SYMBOL_HISTORY_BARS),
                          closed(frames['BTCUSDT'], stamp, REQUIRED_BTC_HISTORY_BARS),
                          entry_time=stamp, symbol=symbol, market_price=row['entry'],
                          model_price=row['opening_anchor'])
    if (len(setups)!=1 or setups[0]['direction']!=row['direction']
            or not math.isclose(setups[0]['sl'], row['stop'], rel_tol=1e-12)):
        raise ValueError('reconstructed market setup differs from the frozen proposal')
    return dict(row, family=setups[0]['family'])


def exact(rows, specs, ticks, books, archives, trades, references, expected):
    frame = ClosedContext(load_native(references))
    eligible = [label_family(r, frame.frames) for r in rows if r['entry_status']=='eligible'
                and (r['bot']!='stocks' or r['stock_reference_ready'])]
    features = {key(r):frame.features(r) for r in eligible}
    all_rows = {key(row):row for row in eligible}
    refused = {name:{entry for entry, f in features.items() if skip(name, all_rows[entry], f)}
               for name in POLICIES}
    repricing = dict(repriced=[], extra_exit_checks={})
    replay = with_actual_fill_path(trades, archives, repricing)
    output = {}
    for lag in CLOSE_LAGS:
        params = dict(clock=60, lag=lag, setting='requested5')
        baseline = simulate(rows, specs, ticks, books, **params)
        if any(baseline.get(name) != expected[str(lag)].get(name) for name in entry_data.FIELDS):
            raise ValueError('original executable cash baseline changed')
        outcomes = dict(baseline=cash_summary(baseline, {}))
        for policy, blocks in refused.items():
            subset = [dict(r,entry_status=policy) if key(r) in blocks else r for r in rows]
            test = replay(subset, specs, ticks, books, **params) if blocks else baseline
            summary = cash_summary(test, {})
            original = [r for r in baseline['closed'] if key(r) in blocks]
            summary['rejected_baseline'] = [dict(symbol=r['symbol'],bot=r['bot'],entry_ms=r['entry_ms'],
                outcome=r['code'],cash_pnl_usd=r['bank_pnl']+r['runner_pnl'],
                **features[key(r)]) for r in original]
            if 'closed' in test:
                summary['difference'] = trade_difference(baseline, test)
            outcomes[policy] = dict(summary, rejected_candidates=len(blocks))
        output[str(lag)] = outcomes
        print(lag, {name:(r.get('trades'),r.get('return_pct'),r.get('rejected_candidates'),r.get('reason'))
                    for name,r in outcomes.items()}, flush=True)
    pattern = Counter()
    for entry, f in features.items():
        row = all_rows[entry]
        family = row['family'] if row['bot']=='crypto' else 'native_stock_reversal'
        pattern[f'{family}:candidates'] += 1
        if f is not None:
            pattern[f'{family}:own_context'] += 1
            pattern[f'{family}:peer_context'] += int(f['peer_known'])
            pattern[f'{family}:breakout_range_reproduced'] += int(f['breakout_cushion_atr'] is not None)
    return dict(results=output, feature_coverage=pattern, repricing=repricing)


def older():
    rows, *inputs = old_inputs()
    mark_old_family(rows)
    source = read_json(ROOT/'first_print_confirmation_older_nov25.json')['cases']
    return {name:exact([r for r in rows if keep(r)], *inputs,
                       {lag: case['baseline'] for lag,case in source[name]['results'].items()})
            for name, keep in EARLIER.items()}


def native_joint():
    with patch.object(entry_data, 'compare', exact):
        return entry_data.native_joint()


def fresh():
    prior = read_json(ROOT/'first_print_delay_holdout_nov25.json')['scores']['results']
    expected = {lag: case['baseline'] for lag,case in prior.items()}

    def evaluate(material, *_args, **_kwargs):
        return exact(*material, expected)

    with patch.object(fresh_data, 'compare', evaluate):
        case, evidence = fresh_data.fresh_crypto()
    return {'nov22_crypto_12h':dict(case,evidence=evidence)}


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    modes = {'older':older, 'native_joint':native_joint, 'fresh':fresh}
    parser.add_argument('mode', choices=modes)
    mode = parser.parse_args().mode
    results = modes[mode]()
    write_json(ROOT/f'closed_acceptance_context_{mode}_nov25.json', dict(
        cases=results, script_sha256=file_hash(Path(__file__)), policies=POLICIES,
        thresholds=dict(location=LOCATION, breakout_cushion_atr=CUSHION_ATR,
                        rejection=REJECTION, opposed_share=OPPOSED_FRACTION, opposed_atr=OPPOSED_ATR),
        live_activation_supported=False, limitations=[
            'All decision inputs come from closed bars strictly before the signal epoch.',
            'Breakout cushion uses the original 20-bar high/low and 14 true ranges ending one bar before the signal bar.',
            'Peer pressure excludes the candidate, uses only active contracts in the captured same-bot basket and leaves unknown context unfiltered.',
            'Historical publication and quote qualification are fixed. Shared cash and closings are recalculated.',
            'These public order-book cash simulations do not score private market fills or a complete month.',
        ]))


if __name__ == '__main__':
    main()
