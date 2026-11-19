"""Frozen account replay with causal traded-bar gap and activity filters."""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

from shared_admission_replay_nov19 import (
    CLIENTS, MANIFEST, ROOT, STOCK, STOCK_MANIFEST, case, load_frame, records,
)


BAR_SECONDS = 900
LOOKBACK = 16
MIN_ACTIVE = 6
MAX_GAP_R = 2.0
RULES = ('control', 'active_six_of_eight', 'gap_at_most_2r', 'active_and_gap')


def recent_trade_evidence(bars, index, risk_fraction):
    """Measure activity and gaps using archived base volume before entry."""
    if index < LOOKBACK or not math.isfinite(risk_fraction) or risk_fraction <= 0:
        raise ValueError('insufficient gap history or invalid bracket')
    start = index - LOOKBACK
    window = range(start, index)
    if any(bars['time'][i + 1] - bars['time'][i] != BAR_SECONDS for i in window):
        raise ValueError('noncontiguous gap history')
    volume = bars['volume']
    if any(not math.isfinite(volume[i]) or volume[i] < 0 for i in window):
        raise ValueError('invalid volume')
    active = sum(volume[i] > 0 for i in range(index - 8, index))
    traded = [i for i in window if volume[i] > 0]
    gaps = [abs(bars['open'][b] / bars['close'][a] - 1) / risk_fraction
            for a, b in zip(traded, traded[1:])]
    return active, max(gaps, default=0.0) if len(traded) >= 2 else None


def include_rule(evidence, rule):
    active, gap_r = evidence
    enough_trades = active >= MIN_ACTIVE
    bounded_gap = gap_r is not None and gap_r <= MAX_GAP_R
    return {'control': True, 'active_six_of_eight': enough_trades,
            'gap_at_most_2r': bounded_gap,
            'active_and_gap': enough_trades and bounded_gap}[rule]


def add_evidence(rows, manifest, *, data_root=Path('.')):
    by_symbol = json.loads(manifest.read_bytes())['symbols']
    for symbol in sorted({row['symbol'] for row in rows}):
        item = by_symbol[symbol]
        bars = load_frame(data_root / item['path'], item['sha256'])
        lookup = {int(stamp): index for index, stamp in enumerate(bars['time'])}
        for row in rows:
            if row['symbol'] != symbol:
                continue
            risk = (abs(row['entry'] - row['sl']) / row['entry']
                    if row['bot'] == 'crypto' else row['risk_pct'])
            row['gap_evidence'] = recent_trade_evidence(bars, lookup[int(row['entry_time'])], risk)


def counterfactuals(crypto, stock, specs):
    results = {}
    for bot, rows in (('crypto', crypto), ('stocks', stock)):
        bot_result = {}
        for rule in RULES:
            keep = [row for row in rows if include_rule(row['gap_evidence'], rule)]
            reject = [row for row in rows if not include_rule(row['gap_evidence'], rule)]
            keep_ids = {(row['symbol'], row['entry_time']) for row in keep}
            feed = [row for row in crypto + stock if row['bot'] != bot or
                    (row['symbol'], row['entry_time']) in keep_ids]
            label = 'sequential_outcome' if bot == 'crypto' else 'outcome'
            bot_result[rule] = dict(
                kept=len(keep), rejected=len(reject),
                rejected_outcomes=dict(Counter(row[label] for row in reject)),
                rejected_worse_than_2r=sum(row['net_r'] < -2 for row in reject),
                total_worse_than_2r=sum(row['net_r'] < -2 for row in rows),
                cash={policy: case(feed, specs, priority='crypto', policy=policy)
                      for policy in CLIENTS},
            )
        results[bot] = bot_result
    return results


def main():
    base = ROOT / 'native_gap_risk_nov19'
    inputs = [Path(__file__), Path('shared_admission_replay_nov19.py'),
              ROOT / 'shared_admission_replay_nov19_protocol.json', MANIFEST, STOCK_MANIFEST]
    sources = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in inputs}
    declaration = base.with_name(base.name + '_protocol.json')
    if declaration.exists():
        if json.loads(declaration.read_bytes())['sources'] != sources:
            raise ValueError('changed gap-risk protocol inputs')
    else:
        declaration.write_text(json.dumps(dict(
            declared_at=datetime.now(timezone.utc).isoformat(), sources=sources,
            rules=RULES, lookback_bars=LOOKBACK, minimum_active_of_last_8=MIN_ACTIVE,
            maximum_absolute_print_gap_r=MAX_GAP_R,
            shared_account='same frozen hypothetical readiness and admission as parent replay',
        ), indent=2), encoding='utf-8')
    crypto, stock, _, specs, end = records()
    add_evidence(crypto, MANIFEST)
    add_evidence(stock, STOCK_MANIFEST, data_root=STOCK.parents[1])
    result = counterfactuals(crypto, stock, specs)
    for bot, policies in result.items():
        for rule, row in policies.items():
            cash = row['cash']['half_minimum']
            monthly = [round(month['return_contribution_pct'][bot], 2)
                       for month in cash['monthly'].values()]
            print(bot, rule, row['kept'], row['rejected_worse_than_2r'],
                  row['total_worse_than_2r'], monthly, round(cash['cash_drawdown_pct'], 2))
    base.with_suffix('.json').write_text(json.dumps(dict(
        status='RETROSPECTIVE_GAP_DIAGNOSTIC', data_end_utc=end,
        objective_verified=False, live_activation_supported=False, results=result,
        limitations=[
            'Fixed previously admitted historical proposals. Excluded entries do not refill earlier publication slots.',
            'Stock readiness is the same fixed counterfactual as the parent replay, not recalculated after filtering.',
            'A 15m traded bar does not bound the time since last trade, and its worst intrabar gap may be invisible.',
            'Archived base volume can round positive contract volume to zero, particularly on thin books. This may change every liquidity-filter result.',
            'Previously inspected sample, no live fills, historical order books or account equity guard, partial November.',
            'Some inherited stock expiries can use carried terminal prices. No fitted or promoted policy.',
        ],
    ), indent=2, allow_nan=False), encoding='utf-8')


if __name__ == '__main__':
    main()
