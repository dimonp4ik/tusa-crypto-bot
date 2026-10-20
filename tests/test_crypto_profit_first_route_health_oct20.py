import pandas as pd

import crypto_profit_first_route_health_oct20_audit as audit


def test_health_gate_never_uses_exit_at_or_after_entry():
    rows = pd.DataFrame([
        {"health_key": "trend|pullback", "symbol": "BTCUSDT",
         "entry_time": 10, "exit_time": 30, "signed_price_return": -.02},
        {"health_key": "trend|pullback", "symbol": "ETHUSDT",
         "entry_time": 20, "exit_time": 21, "signed_price_return": .01},
    ])
    result = audit.health_gate(rows, lookback=10, minimum_mean=0)
    assert result.route_health_observations.tolist() == [0, 0]


def test_health_gate_uses_only_strictly_prior_exit():
    rows = pd.DataFrame([
        {"health_key": "trend|pullback", "symbol": f"S{i}",
         "entry_time": i * 10, "exit_time": i * 10 + 1,
         "signed_price_return": -.01}
        for i in range(6)
    ])
    result = audit.health_gate(rows, lookback=10, minimum_mean=0)
    assert result.entry_time.tolist() == [0, 10, 20, 30, 40]
