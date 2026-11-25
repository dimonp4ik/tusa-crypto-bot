"""Check the outcome filter on the separate 22 November executable cohort."""

from pathlib import Path
from unittest.mock import patch

import cash_loss_pause_audit_nov24 as archive
from okx_unbankable_tp1_nov21 import read_json
from okx_xperp_cash_tick_nov19 import CLOSE_LAGS
from okx_xperp_multiday_replay_nov20 import ROOT, file_hash
from settled_cash_gate_nov25 import compare
from stock_liquid_quote_routes_nov21 import write_json


def main():
    source = archive.FRESH / 'fresh_result.json'
    original = read_json(source)['cases']['nov22_crypto_12h']
    control = {str(lag): original[f'60s_{lag}ms_release_900s']['baseline'] for lag in CLOSE_LAGS}

    def evaluate(data, *_args, **_kwargs):
        rows, specs, ticks, books, _, _, references = data
        return compare(rows, specs, ticks, books, references, control)

    with patch.object(archive, 'compare', evaluate):
        cases, evidence = archive.fresh_crypto()
    write_json(ROOT/'settled_cash_gate_holdout_nov25.json', dict(cases=cases, evidence=evidence,
        source_sha256={path.name:file_hash(path) for path in (source, Path(archive.__file__), Path(__file__))},
        live_activation_supported=False))


if __name__ == '__main__':
    main()
