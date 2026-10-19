import pandas as pd

import crypto_long_history_monthly_style_router_oct20_audit as audit


def row(exit_time, value):
    return {"exit_time": exit_time, "signed_price_return": value}


def test_trailing_scores_exclude_unfinished_and_cutoff_exits():
    cutoff = pd.Timestamp("2025-02-01", tz="UTC")
    library = {
        "a": [
            row(pd.Timestamp("2025-01-20", tz="UTC").timestamp(), .01),
            row(cutoff.timestamp(), -1.0),
            row(pd.Timestamp("2025-02-02", tz="UTC").timestamp(), -1.0),
        ]
    }
    scores = audit.trailing_scores(library, cutoff, 1, 1, 0, "mean")
    assert scores == [(.01, 1, "a")]


def test_router_stages_are_disjoint():
    assert set(audit.base.DEVELOPMENT_YEARS).isdisjoint(
        audit.base.CONFIRMATION_YEARS)
    assert set(audit.base.CONFIRMATION_YEARS).isdisjoint(
        audit.base.HIDDEN_YEARS)
