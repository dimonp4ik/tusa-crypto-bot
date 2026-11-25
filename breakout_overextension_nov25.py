"""Test whether avoiding extremely extended breakouts reduces failed follow-through."""

from pathlib import Path
from unittest.mock import patch

import closed_acceptance_context_nov25 as acceptance
import closed_acceptance_monthly_context_nov25 as monthly
from okx_unbankable_tp1_nov21 import read_json
from okx_xperp_multiday_replay_nov20 import ROOT, file_hash
from stock_liquid_quote_routes_nov21 import write_json


EXTENSION_ATR = 3.
PEER_PRESSURE_ATR = 3.
POLICIES = ('own_extension', 'broad_extension', 'shallow_or_own_extension', 'shallow_or_broad_extension')


def skip(policy, row, context):
    if context is None or context['breakout_cushion_atr'] is None:
        return False
    distance = context['breakout_cushion_atr']
    own = distance >= EXTENSION_ATR
    broad = (distance >= 1. and context['peer_known']
             and context['peer_median_atr'] >= PEER_PRESSURE_ATR
             and context['peer_opposed_fraction'] <= .25)
    if policy == 'own_extension':
        return own
    if policy == 'broad_extension':
        return broad
    shallow = distance < acceptance.CUSHION_ATR
    if policy == 'shallow_or_own_extension':
        return shallow or own
    if policy == 'shallow_or_broad_extension':
        return shallow or broad
    raise ValueError('unrecognized overextension policy')


def long_history():
    crypto, ready, paper, specs, end = monthly.records()
    reference = ROOT/'shared_admission_replay_nov19.json'
    frozen = read_json(reference)['results']
    context = monthly.candidate_context({r['symbol'] for r in crypto+paper}, Path('../tusa stocks'))
    return {name:monthly.run(crypto+stocks, specs, context, frozen[name]['requested5']['crypto'])
            for name,stocks in (('ready_one_writer',ready), ('all_stock_capacity_diagnostic',paper))}


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    modes = dict(older=acceptance.older, native_joint=acceptance.native_joint,
                 fresh=acceptance.fresh, monthly=long_history)
    parser.add_argument('mode', choices=modes)
    mode = parser.parse_args().mode
    with patch.multiple(acceptance, POLICIES=POLICIES, skip=skip):
        result = modes[mode]()
    source_files = (Path(__file__), Path(acceptance.__file__), Path(monthly.__file__))
    write_json(ROOT/f'breakout_overextension_{mode}_nov25.json', dict(
        cases=result, policies=POLICIES, own_atr=EXTENSION_ATR, peer_atr=PEER_PRESSURE_ATR,
        source_sha256={p.name:file_hash(p) for p in source_files}, live_activation_supported=False,
        limitations=[
            'The rule was motivated by the observed November 23 BTC and ETH stops. Those trades are not the test cohort.',
            'Only the original crypto breakout family is modified. Stock positions continue competing for shared capital.',
            'Every feature uses only closed bars before the entry. The own range excludes the breakout bar.',
            'Exact cash modes price historical OKX depth. The longer sample uses an existing candle-price proxy.',
        ]))


if __name__ == '__main__':
    main()
