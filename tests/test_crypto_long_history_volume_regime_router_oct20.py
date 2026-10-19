import pandas as pd

import crypto_long_history_volume_regime_router_oct20_audit as audit


def test_stable_expert_requires_every_development_year():
    frame = pd.DataFrame({
        "year": [2020, 2021], "variant": ["a", "a"],
        "direction": ["LONG", "LONG"],
        "signed_price_return": [.01, .01],
    })
    found = audit.stable_experts(
        frame, ("variant", "direction"), 1, 0)
    assert found.empty


def test_stable_expert_accepts_positive_all_year_bucket():
    frame = pd.DataFrame({
        "year": [2020, 2021, 2022], "variant": ["a"] * 3,
        "direction": ["LONG"] * 3,
        "signed_price_return": [.01, .02, .03],
    })
    found = audit.stable_experts(
        frame, ("variant", "direction"), 1, 0)
    assert len(found) == 1
    assert found.iloc[0].worst_year_mean == .01


def test_hidden_years_are_not_expert_development_years():
    assert set(audit.base.DEVELOPMENT_YEARS).isdisjoint(
        audit.base.HIDDEN_YEARS)
