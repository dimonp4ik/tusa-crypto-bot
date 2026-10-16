import pandas as pd

import crypto_profit_first_regime_target_router_oct17_audit as audit


def test_rules_are_causal_and_return_boolean_masks():
    frame = pd.DataFrame({
        "direction": [1, -1], "btc_ret24": [.02, .02],
        "btc_ret168": [.05, -.05], "atr14_pct": [.009, .005],
        **{f"family_{name}": [1, 0] for name in
           ("trend", "pullback", "breakout", "compression_release",
            "breadth", "relative_leader")},
    })
    for rule in audit.rules().values():
        mask = rule(frame)
        assert list(mask.index) == list(frame.index)
        assert mask.dtype == bool


def test_development_gate_ignores_win_rate():
    value = {"return_pct": 10.0, "dd_pct": 12.0,
             "trades_per_calendar_day": 3.1, "wr": 24.0}
    assert audit.development_ok(value)
