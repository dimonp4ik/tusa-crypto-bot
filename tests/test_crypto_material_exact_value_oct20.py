import math

import pandas as pd

import crypto_material_exact_value_oct20_audit as audit


def test_fraction_pricing_follows_one_exact_exit_path():
    frame = pd.DataFrame({
        "runner_outcome": ["TP2", "BE", "SL"],
        "net_r": [2.0, 0.0, -1.0],
    })
    result = audit.with_fraction(frame, 0.5)
    assert result.net_r.tolist() == [1.25, 0.25, -1.0]
    assert result.runner_net_r.tolist() == [2.0, 0.0, -1.0]


def test_repricing_preserves_stop_and_reprices_be_and_runner():
    original = [{
        "runner_outcome": "TP2", "runner_net_r": 2.0,
        "net_r": 1.0, "risk_pct": 0.01, "outcome": "TP",
    }, {
        "runner_outcome": "BE", "runner_net_r": 0.0,
        "net_r": 0.1, "risk_pct": 0.02, "outcome": "TP",
    }, {
        "runner_outcome": "SL", "runner_net_r": -1.0,
        "net_r": -1.0, "risk_pct": 0.015, "outcome": "SL",
    }]
    priced = audit.reprice_rows(original, 0.3)
    assert math.isclose(priced[0]["net_r"], 1.55)
    assert math.isclose(priced[1]["net_r"], 0.15)
    assert math.isclose(priced[2]["net_r"], -1.0)
    assert math.isclose(priced[0]["signed_price_return"], 0.0155)
    assert original[0]["net_r"] == 1.0
