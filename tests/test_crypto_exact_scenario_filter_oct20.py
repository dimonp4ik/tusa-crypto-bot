import pandas as pd

import crypto_exact_scenario_filter_oct20_audit as audit


def test_stable_buckets_require_positive_expectancy_in_both_periods():
    selection = pd.DataFrame({
        "family": ["good", "good", "unstable", "unstable"],
        "net_r": [1.0, 0.0, 1.0, 1.0],
    })
    confirmation = pd.DataFrame({
        "family": ["good", "good", "unstable", "unstable"],
        "net_r": [1.0, 0.0, -1.0, -1.0],
    })

    result = audit.stable_buckets(
        selection, confirmation, ("family",), min_count=2,
        min_mean_r=.25)

    assert result == {("good",)}


def test_apply_buckets_handles_multiple_fields():
    frame = pd.DataFrame({
        "family": ["breakout", "pullback"],
        "direction": ["LONG", "SHORT"],
    })

    result = audit.apply_buckets(
        frame, ("family", "direction"), {("pullback", "SHORT")})

    assert result.index.tolist() == [1]
