from crypto_causal_runner_grid_nov16_audit import policies, portfolio
import pandas as pd


def test_exit_grid_is_bounded_and_trailing_is_after_2r_before_a_further_target():
    rules = list(policies())
    assert len(rules) == 32
    assert sum(rule['trail_distance_r'] is not None for rule in rules) == 16
    assert all(rule['target_r'] > 2 for rule in rules if rule['trail_distance_r'] is not None)


def test_live_pause_proxy_does_not_call_a_negative_timed_exit_an_sl_hit():
    records = []
    for i, (kind, value) in enumerate((('SL', -1.), ('SL', -1.), ('EXPIRED', -.2), ('TP_RUNNER', 2.))):
        entry = 36000 + i * 3600
        records.append(dict(symbol=f'TEST-{i}', direction='LONG', entry_time=entry,
            exit_time=entry + 600, risk_pct=.01, net_r=value, sequential_outcome=kind))
    frame = pd.DataFrame(records)
    assert len(portfolio(frame, operational_pause=False)) == 3
    assert len(portfolio(frame, operational_pause=True)) == 4
