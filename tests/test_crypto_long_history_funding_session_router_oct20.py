import pandas as pd

import crypto_long_history_funding_session_router_oct20_audit as audit


def test_calendar_buckets_are_deterministic_utc():
    times = [
        pd.Timestamp("2025-01-06 00:00", tz="UTC").timestamp(),
        pd.Timestamp("2025-01-06 08:00", tz="UTC").timestamp(),
        pd.Timestamp("2025-01-06 16:00", tz="UTC").timestamp(),
    ]
    found = audit.add_calendar(pd.DataFrame({"entry_time": times}))
    assert found.session.tolist() == ["asia", "europe", "us"]
    assert found.hour_block.tolist() == [0, 2, 5]
    assert found.weekday.tolist() == [0, 0, 0]


def test_hidden_years_are_disjoint_from_learning_years():
    assert set(audit.base.HIDDEN_YEARS).isdisjoint(
        set(audit.base.DEVELOPMENT_YEARS) | set(audit.base.CONFIRMATION_YEARS))
