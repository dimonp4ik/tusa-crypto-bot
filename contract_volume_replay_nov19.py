"""Pair base-volume and contract-volume X-Perp exits on the same price path."""
from bisect import bisect_left
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import pickle

from banked_runner_path_nov19 import resolve as resolve_crypto
from contract_volume_history_nov19 import COHORT_START, FOLDER, STEP, digest
from native_gap_risk_nov19 import RULES, include_rule, recent_trade_evidence
from shared_admission_replay_nov19 import CLIENTS, ROOT, STOCK, case, records
from src.tp1_volume_bank import candidate_r, next_open_observation


VOLUMES = ('volume_base', 'volume_contracts')


def checked_pickle(path, expected_sha256):
    raw = path.read_bytes()
    if digest(raw) != expected_sha256:
        raise ValueError(f'history digest mismatch: {path.name}')
    return pickle.loads(raw)


def frames():
    declaration = FOLDER / 'selection.json'
    manifest_path = FOLDER / 'manifest.json'
    selected = json.loads(declaration.read_bytes())
    manifest = json.loads(manifest_path.read_bytes())
    if (not manifest['complete'] or set(selected['symbols']) != set(manifest['symbols'])
            or digest(declaration.read_bytes()) != manifest['selection_sha256']):
        raise ValueError('capture is incomplete or does not match declared cohort')
    result = {}
    for symbol, item in selected['symbols'].items():
        fetched = manifest['symbols'][symbol]
        current = checked_pickle(Path(fetched['path']), fetched['sha256'])
        repository = Path('.') if item['bot'] == 'crypto' else STOCK.parents[1]
        previous = checked_pickle(repository / item['archive_path'], item['archive_sha256'])
        previous_index = {t: i for i, t in enumerate(previous['time'])}
        matches = Counter()
        for i, stamp in enumerate(current['time']):
            j = previous_index.get(stamp)
            if j is None:
                continue
            matches['overlap'] += 1
            matches['ohlc_changed'] += any(current[k][i] != previous[k][j]
                                           for k in ('open', 'high', 'low', 'close'))
            volume = previous['volume'][j]
            base, contracts = (current[key][i] for key in VOLUMES)
            matches['base_only'] += volume == base and base != contracts
            matches['contracts_only'] += volume == contracts and base != contracts
            matches['matches_both'] += volume == base == contracts
            matches['matches_neither'] += volume != base and volume != contracts
        counts = dict(matches, base_zero_contract_positive=sum(
            base == 0 < contracts for base, contracts in zip(*(current[key] for key in VOLUMES))),
            traded_contract_bars=sum(value > 0 for value in current['volume_contracts']))
        result[symbol] = (current, counts, fetched['sha256'])
    return result, digest(manifest_path.read_bytes())


def stock_exit(frame, index, row):
    entry = row['entry']
    sign = 1 if row['direction'] == 'LONG' else -1
    risk = row['risk_pct']
    target_fraction = row['target_pct'] / 100
    stop = entry * (1 - sign * risk)
    target = entry * (1 + sign * target_fraction)
    end = index + row['hold_bars']
    if end > len(frame['time']):
        raise ValueError('missing stock exit horizon')
    for i in range(index, end):
        if frame['volume'][i] <= 0:
            continue
        opening, high, low, close = (frame[key][i] for key in ('open', 'high', 'low', 'close'))
        if sign * (opening - stop) <= 0:
            code, price = 'SL', opening
            break
        if sign * (opening - target) >= 0:
            code, price = 'TP', target
            break
        if sign * ((low if sign > 0 else high) - stop) <= 0:
            code, price = 'SL', stop
            break
        if sign * ((high if sign > 0 else low) - target) >= 0:
            code, price = 'TP', target
            break
        if i == end - 1:
            code, price = 'TIME', close
            break
    else:
        return dict(known=False, reason='no_traded_terminal_candle')
    return dict(known=True, outcome=code, exit_time=frame['time'][i] + STEP,
                banked=False, near_exit_time=None, bank_price_return=0.,
                runner_price_return=sign * (price / entry - 1))


def exit_for(row, frame, index):
    if row['bot'] == 'stocks':
        return stock_exit(frame, index, row)
    return resolve_crypto(frame, index, direction=row['direction'], entry=row['entry'],
                          stop=row['sl'], policy='be_tp1', exit_stress=.0005)


def paired(rows, market):
    result = []
    for row in rows:
        frame, _, _ = market[row['symbol']]
        i = bisect_left(frame['time'], row['entry_time'])
        if i >= len(frame['time']) or frame['time'][i] != row['entry_time']:
            raise ValueError('missing entry candle')
        risk_fraction = (abs(row['entry'] - row['sl']) / row['entry']
                         if row['bot'] == 'crypto' else row['risk_pct'])
        evidence = {}
        for key in VOLUMES:
            candles = dict(frame, volume=frame[key])
            path = exit_for(row, candles, i)
            entry_evidence = recent_trade_evidence(candles, i, risk_fraction)
            evidence[key] = dict(path=path, gate={rule: include_rule(entry_evidence, rule) for rule in RULES})
        result.append((row, evidence))
    return result


def model_r(path, risk):
    fraction = .05 if path['banked'] else 0.
    return (fraction * path['bank_price_return'] + (1 - fraction) * path['runner_price_return']) / risk


def stats(pairs):
    transitions = Counter()
    gate_changes = Counter()
    unknown = Counter()
    delta = []
    changed = []
    for row, replay in pairs:
        base, contracts = (replay[key]['path'] for key in VOLUMES)
        for gate in RULES:
            gate_changes[gate] += replay[VOLUMES[0]]['gate'][gate] != replay[VOLUMES[1]]['gate'][gate]
        old, new = (path.get('outcome', 'UNKNOWN') for path in (base, contracts))
        transitions[f'{old}->{new}'] += 1
        for key, path in zip(VOLUMES, (base, contracts)):
            if not path['known']:
                unknown[f'{key}:{path["reason"]}'] += 1
        if base['known'] and contracts['known']:
            risk = abs(row['entry'] - row['sl']) / row['entry'] if row['bot'] == 'crypto' else row['risk_pct']
            base_r, contract_r = (model_r(path, risk) for path in (base, contracts))
            delta.append(contract_r - base_r)
        if base != contracts:
            changed.append(dict(symbol=row['symbol'], entry_time=row['entry_time'],
                                base=base, contracts=contracts))
    return dict(count=len(pairs), transitions=dict(transitions), gate_changes=dict(gate_changes),
                unknown=dict(unknown), comparable=len(delta), changed_count=len(changed),
                comparable_sum_delta_r=math.fsum(delta), changed=changed)


def next_open_shadow(pairs, market):
    """Compare the currently defined paper challenger without changing orders."""
    result = Counter()
    deltas = {key: [] for key in VOLUMES}
    for row, evidence in pairs:
        path = evidence['volume_contracts']['path']
        if not path['known'] or not path['banked']:
            continue
        result['banked'] += 1
        action = path['near_exit_time']
        if path['exit_time'] <= action:
            continue
        c, _, _ = market[row['symbol']]
        index = bisect_left(c['time'], action)
        if index >= len(c['time']) or c['time'][index] != action or c['volume_contracts'][index] <= 0:
            result['no_next_open_trade'] += 1
            continue
        entry = row['entry']
        sign = 1 if row['direction'] == 'LONG' else -1
        risk = abs(entry - row['sl'])
        action_r = sign * (c['open'][index] - entry) / risk
        if not 0 < action_r < 2:
            continue
        result['runner_open_at_action'] += 1
        runner_r = path['runner_price_return'] * entry / risk
        bank_r = path['bank_price_return'] * entry / risk
        baseline = candidate_r(tp1_fraction=.05, tp1_r=bank_r, extra_fraction=0.,
                               extra_r=0., runner_r=runner_r)
        applied = []
        for key in VOLUMES:
            observed = next_open_observation(dict(c, volume=c[key]), activation_ts=action - STEP,
                                             bar_seconds=STEP, now=action)
            extra = observed['extra_close_fraction'] if observed else 0.
            result[f'{key}_no_observation'] += observed is None
            result[f'{key}_applied'] += extra > 0
            applied.append(extra)
            r = candidate_r(tp1_fraction=.05, tp1_r=bank_r, extra_fraction=extra,
                            extra_r=action_r, runner_r=runner_r)
            deltas[key].append(r - baseline)
        result['decision_changes'] += applied[0] != applied[1]
    return dict(counts=dict(result), delta_r={key: math.fsum(values) for key, values in deltas.items()})


def cash_pairs(pairs, specs):
    missing = {key: sum(not pair[key]['path']['known'] for _, pair in pairs) for key in VOLUMES}
    if any(missing.values()):
        return dict(status='UNSCORABLE', unknown=missing)
    results = {}
    for key in VOLUMES:
        rows = []
        for row, paths in pairs:
            path = paths[key]['path']
            unit = 2 * path['bank_price_return'] if row['bot'] == 'crypto' else row['risk_pct']
            rows.append(dict(row, risk_pct=unit, exit_time=path['exit_time'],
                             near_exit_time=path['near_exit_time'], sequential_banked=path['banked'],
                             sequential_runner_r=path['runner_price_return'] / unit))
        results[key] = {policy: case(rows, specs, priority='crypto', policy=policy) for policy in CLIENTS}
    return dict(status='PAIRED_PUBLIC_PROXY', results=results)


def main():
    market, capture_hash = frames()
    crypto, ready, stocks, specs, end = records()
    crypto_pairs = paired([row for row in crypto if row['entry_time'] >= COHORT_START], market)
    stock_pairs = paired([row for row in stocks if row['entry_time'] >= COHORT_START], market)
    ready_ids = {(row['symbol'], row['entry_time']) for row in ready}
    ready_pairs = [(row, proof) for row, proof in stock_pairs if (row['symbol'], row['entry_time']) in ready_ids]
    proposal_stats = dict(crypto=stats(crypto_pairs), stocks=stats(stock_pairs), stocks_ready=stats(ready_pairs))
    sources = [Path(__file__), Path('contract_volume_history_nov19.py'),
               Path('native_gap_risk_nov19.py'), Path('banked_runner_path_nov19.py'),
               Path('shared_admission_replay_nov19.py'), ROOT / 'shared_admission_replay_nov19_protocol.json']
    report = dict(status='PAIRED_NATIVE_VOLUME_EVIDENCE', capture_manifest_sha256=capture_hash,
                  sources={str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources},
                  cohort_start=COHORT_START, coverage_end=end, objective_verified=False,
                  live_activation_supported=False,
                  archive_volume_correction=(
                      'The paired overlap directly identifies the frozen archive volume field. '
                      'On all verified overlaps it agrees with contracts, so the prior gap-risk '
                      'report warning about archived base-volume rounding does not apply to this overlap.'),
                  market={s: dict(info, sha256=sha) for s, (_, info, sha) in market.items()},
                  proposals=proposal_stats, crypto_next_open_shadow=next_open_shadow(crypto_pairs, market),
                  cash=dict(crypto_only=cash_pairs(crypto_pairs, specs),
                          frozen_readiness_common_account=cash_pairs(crypto_pairs + ready_pairs, specs)),
                  limitations=[
                      'Fixed prior proposals, 15m public paths, no historical quotes or realized client fills.',
                      'Only volume eligibility changes between paired exit arms; entry selection is fixed.',
                      'Stock paths lacking a real terminal trade stay unscored; no backfilling a last price.',
                      'Frozen historical stock readiness is unchanged even when recalculated outcomes differ.',
                      'Historical control uses the fixed prior account replay cohort, not full current production parity.',
                  ])
    target = ROOT / 'contract_volume_replay_nov19.json'
    target.write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    for name, values in proposal_stats.items():
        print(name, {k: v for k, v in values.items() if k != 'changed'}, flush=True)
    print('crypto shadow', report['crypto_next_open_shadow'])
    print('cash status', {name: value['status'] for name, value in report['cash'].items()})


if __name__ == '__main__':
    main()
