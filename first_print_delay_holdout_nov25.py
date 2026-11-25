"""Validate the fixed 15-second delay on the separate November 22 crypto cohort."""

from pathlib import Path
from unittest.mock import patch

import cash_loss_pause_audit_nov24 as archive
import first_print_confirmation_nov25 as confirmation
from okx_unbankable_tp1_nov21 import read_json
from okx_xperp_multiday_replay_nov20 import ROOT, file_hash
from stock_liquid_quote_routes_nov21 import write_json


def main():
    source = ROOT / 'settled_cash_gate_holdout_nov25.json'
    results = read_json(source)['cases']['results']
    controls = {lag: result['cases']['baseline'] for lag, result in results.items()}
    policy = 'delay_15_seconds'

    def compare(material, *_args, **_kwargs):
        return confirmation.compare(*material, controls)

    with patch.object(archive, 'compare', compare), patch.dict(
        confirmation.POLICIES, {policy: confirmation.POLICIES[policy]}, clear=True
    ):
        scores, evidence = archive.fresh_crypto()
    sources = (source, Path(__file__), Path(confirmation.__file__), Path(archive.__file__))
    write_json(ROOT / 'first_print_delay_holdout_nov25.json', dict(
        start=archive.START, end=archive.STOP, policy=policy, scores=scores, evidence=evidence,
        source_sha256={path.name: file_hash(path) for path in sources}, live_activation_supported=False,
        limitations=['Same 15-second delay and frozen original brackets. Only crypto is included.',
                     'This twelve-hour simulation uses public prices, not private broker cash.']))


if __name__ == '__main__':
    main()
