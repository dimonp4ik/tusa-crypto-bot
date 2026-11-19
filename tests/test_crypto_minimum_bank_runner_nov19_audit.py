from copy import deepcopy
from crypto_minimum_bank_runner_nov19_audit import admitted_records


def row(symbol='A',start=0,end=9000,known=True):
    return dict(symbol=symbol,entry_time=start,exit_time=end,direction='LONG',risk_pct=.01,
        net_r=1,sequential_outcome='TARGET',known=known,unknown_reason='stale_runner_time_exit')


def test_unknown_rejected_by_open_symbol_never_changes_any_future_admission():
    records=[row(),row(start=900,known=False),row(symbol='B',start=11000,end=12000)]
    accepted,unknown=admitted_records(records)
    assert not unknown and [r['symbol'] for r in accepted]==['A','B']
    mutated=deepcopy(records)
    mutated[1].update(exit_time=1e20,net_r=-10000,sequential_outcome='SL')
    again,unknown=admitted_records(mutated)
    assert not unknown and again==accepted


def test_accepted_unknown_refuses_cash_scoring_even_if_legacy_payload_says_profit():
    accepted,unknown=admitted_records([row(known=False)])
    assert accepted is None and unknown==[('A',0,'stale_runner_time_exit')]
