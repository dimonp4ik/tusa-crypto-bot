import pandas as pd

import crypto_long_history_five_year_regime_router_oct20_audit as audit


def experts():
    return pd.DataFrame({
        "variant": ["a"], "direction": ["LONG"],
        "worst_year_mean": [.01], "mean_return": [.02],
    })


def test_confirmation_requires_both_years():
    frame = pd.DataFrame({
        "year": [2023], "variant": ["a"], "direction": ["LONG"],
        "signed_price_return": [.01],
    })
    found = audit.confirmed_experts(
        frame, ("variant", "direction"), experts(), 1, 0)
    assert found.empty


def test_confirmation_accepts_positive_2023_and_2024():
    frame = pd.DataFrame({
        "year": [2023, 2024], "variant": ["a", "a"],
        "direction": ["LONG", "LONG"],
        "signed_price_return": [.01, .02],
    })
    found = audit.confirmed_experts(
        frame, ("variant", "direction"), experts(), 1, 0)
    assert len(found) == 1
    assert found.iloc[0].confirmation_worst_mean == .01
