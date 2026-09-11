"""Read-only audit of bot exports. These are model signals, not exchange fills."""
from pathlib import Path
import argparse
import collections
import csv
import hashlib
import json
import statistics


def read_csv_rows(path):
    with path.open(encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))


def metrics(rows):
    rr = [r['_net'] for r in rows]
    wins = [v for v in rr if v > 0]
    losses = [v for v in rr if v < 0]
    equity = peak = drawdown = 0.0
    for row in sorted(rows, key=lambda item: float(item.get('closed_at') or item.get('opened_at') or 0)):
        equity += row['_net']
        peak = max(peak, equity)
        drawdown = max(drawdown, peak - equity)
    return dict(n=len(rr), net_r=sum(rr), wr=100*len(wins)/len(rr) if rr else None,
                avg_win=statistics.mean(wins) if wins else None,
                avg_loss=statistics.mean(losses) if losses else None,
                pf=sum(wins)/-sum(losses) if losses else None,
                max_drawdown_r=drawdown)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('signals', type=Path)
    p.add_argument('setups', type=Path)
    p.add_argument('--round-trip-bps', type=float, default=12.0)
    p.add_argument('--stress-bps', type=float, default=22.0)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    if a.round_trip_bps < 0 or a.stress_bps < a.round_trip_bps:
        p.error('costs must satisfy 0 <= round-trip-bps <= stress-bps')
    rows = read_csv_rows(a.signals)
    setups = read_csv_rows(a.setups)
    closed = [r for r in rows if r['status'] not in ('OPEN', 'TP1_PARTIAL')]
    unavailable = []
    for r in closed:
        if r.get('realized_r') in ('', None):
            unavailable.append(r['id'])
            continue
        e, sl = float(r['entry_price']), float(r['sl'])
        risk_pct = abs(e-sl)/e
        r['_risk_pct'] = risk_pct
        mult = float(r.get('size_mult') or 1)
        r['_gross'] = float(r['realized_r'])*mult
        r['_net'] = r['_gross'] - ((a.round_trip_bps / 10_000) / risk_pct)*mult
        r['_net_stress'] = r['_gross'] - ((a.stress_bps / 10_000) / risk_pct)*mult
    valid = [r for r in closed if '_net' in r]
    by_status = {k: metrics([r for r in valid if r['status'] == k]) for k in sorted({r['status'] for r in valid})}
    sls = [r for r in valid if r['status'] == 'SL_HIT']
    # Freeze joins by signal id. Do not join nearby times or mix unsent simulations.
    setup_ids = [s['signal_id'] for s in setups if s.get('signal_id')]
    joined = {s['signal_id']: s for s in setups if s.get('signal_id')}
    duplicate_setup_links = len(setup_ids) - len(joined)
    missing_setup_links = [r['id'] for r in valid if r['id'] not in joined]
    by_group = {}
    for field in ['symbol', 'direction', 'session', 'entry_source', 'trend_1h', 'trend_4h']:
        buckets = collections.defaultdict(list)
        for r in valid:
            key = r.get(field) or joined.get(r['id'], {}).get(field) or 'missing'
            buckets[key].append(r)
        by_group[field] = {k: metrics(v) for k,v in buckets.items()}
    gate_groups = collections.defaultdict(list)
    for r in valid:
        setup = joined.get(r['id'])
        if not setup:
            gate_groups['missing_setup'].append(r)
            continue
        accepted = setup.get('decision') in ('LONG', 'SHORT')
        gate_groups['accepted' if accepted else 'rejected'].append(r)
        trend = (setup.get('trend') or '').lower()
        aligned = ((r.get('direction') == 'LONG' and trend == 'bull') or
                   (r.get('direction') == 'SHORT' and trend == 'bear'))
        gate_groups['aligned' if aligned else 'not_aligned'].append(r)
        if accepted and aligned:
            gate_groups['accepted_and_aligned'].append(r)
    report = dict(signals=len(rows), closed=len(closed), missing_realized_r=unavailable,
        source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [a.signals,a.setups]},
        first_open=min(float(r['opened_at']) for r in rows), last_open=max(float(r['opened_at']) for r in rows),
        cost_model_bps=dict(base=a.round_trip_bps, stress=a.stress_bps),
        metrics=metrics(valid), gross_r=sum(r['_gross'] for r in valid),
        stress_net_r=sum(r['_net_stress'] for r in valid), by_status=by_status, by_group=by_group,
        gate_groups={key: metrics(value) for key,value in sorted(gate_groups.items())},
        join_integrity=dict(missing_setup_links=missing_setup_links,
                            duplicate_setup_signal_ids=duplicate_setup_links),
        stop_overshoot=dict(n=len(sls), beyond_1r=sum(float(r['realized_r']) < -1.0001 for r in sls),
            worst_unscaled_r=min((float(r['realized_r']) for r in sls),default=None)),
        worst_signals=[{k:r[k] for k in ['id','symbol','direction','status','opened_at','closed_at','realized_r','_net']} for r in sorted(valid,key=lambda r:r['_net'])[:10]],
        setups=dict(n=len(setups), source=dict(collections.Counter(s['source'] for s in setups)),
                    missing_net_r=sum(not s.get('net_r') for s in setups)),
        limitations=['Bot signal prices and R are synthetic accounting, not verified OKX executions.',
                     f'Fee/slippage estimate is {a.round_trip_bps:g} bps round trip; stress estimate is {a.stress_bps:g} bps. Funding excluded.',
                     'Legacy missing size multipliers fall back to 1 as in bot reporting; past risk configuration may differ.',
                     'Gate and alignment groups are same-window diagnostics, not validated trading rules.',
                     'Setup shadow outcomes are excluded from realized-R metrics.'])
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps({k:report[k] for k in ['signals','closed','metrics','gross_r','stress_net_r','stop_overshoot']},ensure_ascii=False))


if __name__ == '__main__':
    main()
