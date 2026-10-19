import pandas as pd

import crypto_long_history_funding_style_router_oct20_audit as audit


def test_funding_join_is_strictly_before_entry():
    frame = pd.DataFrame({
        "symbol": ["BTCUSDT", "BTCUSDT", "ETHUSDT", "ETHUSDT"],
        "entry_time": [1000, 2000, 1000, 2000],
        "direction": ["LONG", "SHORT", "LONG", "SHORT"],
    })
    rates = list(range(100))
    funding = {
        "BTCUSDT": {
            "time": [100 + 10 * value for value in rates],
            "rate": [.0001 + value * .000001 for value in rates],
        },
        "ETHUSDT": {
            "time": [100 + 10 * value for value in rates],
            "rate": [-.0001 - value * .000001 for value in rates],
        },
    }
    found = audit.attach_funding(frame, funding)
    assert (found.funding_time < found.entry_time).all()
    assert int(found.iloc[0].funding_time) == 990


def test_stable_expert_requires_every_requested_year():
    frame = pd.DataFrame({
        "year": [2020, 2021], "variant": ["a", "a"],
        "direction": ["LONG", "LONG"],
        "signed_price_return": [.01, .02],
    })
    found = audit.stable_experts(
        frame, ("variant", "direction"), (2020, 2021, 2022), 1, 0)
    assert found.empty


def test_confirmation_requires_both_confirmation_years():
    development = pd.DataFrame({
        "variant": ["a"], "direction": ["LONG"],
        "worst_year_mean": [.01], "mean_return": [.02],
        "worst_year_win_rate": [.6], "observations": [10],
    })
    frame = pd.DataFrame({
        "year": [2023], "variant": ["a"], "direction": ["LONG"],
        "signed_price_return": [.01],
    })
    found = audit.confirmed_experts(
        frame, ("variant", "direction"), development, 1, 0)
    assert found.empty


def test_hidden_years_are_disjoint_from_learning_years():
    assert set(audit.base.HIDDEN_YEARS).isdisjoint(
        set(audit.base.DEVELOPMENT_YEARS) | set(audit.base.CONFIRMATION_YEARS))
