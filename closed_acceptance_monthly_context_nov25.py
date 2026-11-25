"""Replay the fixed acceptance rules on longer native shared-account cohorts."""

from collections import Counter
from pathlib import Path

import closed_acceptance_context_nov25 as rules
from okx_spot_confirmation_nov22 import load_native
from okx_unbankable_tp1_nov21 import read_json
from okx_xperp_multiday_replay_nov20 import ROOT, file_hash
from shared_admission_replay_nov19 import MANIFEST, STOCK_MANIFEST, case, records
from stock_liquid_quote_routes_nov21 import write_json


def candidate_context(frames, stock_root):
    stock = read_json(STOCK_MANIFEST)['symbols']
    crypto = read_json(MANIFEST)['symbols']
    instruments = {s:dict(v, path=stock_root / v['path']) for s,v in stock.items() if v.get('path')}
    instruments.update({s:v for s,v in crypto.items() if v.get('path')})
    if not frames.issubset(instruments):
        raise ValueError('unavailable native context for a historical candidate')
    return rules.ClosedContext(load_native(instruments))


def identity(row):
    return row['bot'], row['symbol'], row['entry_time']


def run(rows, specs, context, control):
    flags = {name:set() for name in rules.POLICIES}
    coverage = Counter()
    for row in rows:
        candidate = dict(row, epoch=int(row['entry_time']))
        features = context.features(candidate)
        bot = row['bot']
        coverage[f'{bot}:candidates'] += 1
        if features is not None:
            coverage[f'{bot}:own_context'] += 1
            coverage[f'{bot}:peer_context'] += int(features['peer_known'])
        for name in rules.POLICIES:
            if rules.skip(name, candidate, features):
                flags[name].add(identity(row))
    options = dict(priority='crypto', policy='requested5')
    baseline = case(rows, specs, **options)
    if baseline != control:
        raise ValueError('original longer shared account baseline changed')
    result = dict(baseline=baseline)
    for name, blocked in flags.items():
        selected = [r for r in rows if identity(r) not in blocked]
        result[name] = dict(case(selected, specs, **options), rejected=Counter(bot for bot,*_ in blocked))
    return dict(cases=result, coverage=coverage)


def main():
    crypto, ready, paper, specs, end = records()
    reference = ROOT/'shared_admission_replay_nov19.json'
    frozen = read_json(reference)
    context = candidate_context({r['symbol'] for r in crypto+paper}, Path('../tusa stocks'))
    output = {}
    for label, stocks in (('ready_one_writer',ready), ('all_stock_capacity_diagnostic',paper)):
        control = frozen['results'][label]['requested5']['crypto']
        output[label] = run(crypto+stocks, specs, context, control)
        for name, result in output[label]['cases'].items():
            print(label,name,round(result['return_pct'],4),round(result['cash_drawdown_pct'],4),
                  result.get('rejected'),{m:round(v['return_pct'],4) for m,v in result['monthly'].items()},
                  flush=True)
    sources = (Path(__file__), Path(rules.__file__), MANIFEST, STOCK_MANIFEST, reference)
    write_json(ROOT/'closed_acceptance_monthly_context_nov25.json', dict(
        cases=output, end=end, source_sha256={p.name if p not in (MANIFEST,STOCK_MANIFEST) else (
            'crypto_manifest' if p==MANIFEST else 'stock_manifest'):file_hash(p) for p in sources},
        live_activation_supported=False, limitations=[
            'Shared cash starts at $130 with the client 3% margin, leverage 10x and requested 5% TP1.',
            'Frozen historical publication and readiness remain fixed before the cash replay.',
            'The all-stock scenario diagnoses stock signals independent of their live-readiness decision.',
            'Candle price paths do not model entry depth, private fills or fees. November is partial.',
        ]))


if __name__ == '__main__':
    main()
