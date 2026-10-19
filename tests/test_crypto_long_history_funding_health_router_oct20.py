import pandas as pd

import crypto_long_history_funding_health_router_oct20_audit as audit


def test_trailing_scores_exclude_exit_at_decision_cutoff():
    cutoff = pd.Timestamp("2025-02-01", tz="UTC")
    frame = pd.DataFrame({
        "expert_id": ["a", "a"],
        "exit_time": [cutoff.timestamp() - 1, cutoff.timestamp()],
        "signed_price_return": [.01, -1.0],
    })
    found = audit.trailing_scores(frame, cutoff, 1, 1, 0)
    assert len(found) == 1
    assert found.iloc[0]["count"] == 1
    assert found.iloc[0].mean_return == .01


def test_monthly_candidates_never_use_future_entry_for_month():
    frame = pd.DataFrame({
        "expert_id": ["a", "a"],
        "exit_time": [pd.Timestamp("2024-12-31", tz="UTC").timestamp()] * 2,
        "entry_time": [pd.Timestamp("2025-01-15", tz="UTC").timestamp(),
                       pd.Timestamp("2025-02-01", tz="UTC").timestamp()],
        "signed_price_return": [.01, .01],
    })
    found, _ = audit.monthly_candidates(
        frame, "2025-01", "2025-01", lookback_months=1,
        top_k=1, min_count=1, min_mean=0)
    assert len(found) == 1
    assert pd.Timestamp(found.iloc[0].entry_time, unit="s", tz="UTC").month == 1


def test_hidden_years_are_not_policy_selection_years():
    assert set(audit.base.HIDDEN_YEARS).isdisjoint(
        audit.base.CONFIRMATION_YEARS)
