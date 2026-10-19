import pandas as pd

import crypto_xperp_native_expanding_oct20_audit as audit


def test_training_rows_excludes_calibration_and_test_months():
    frame = pd.DataFrame({
        "month": ["2026-04", "2026-05", "2026-06", "2026-07"],
        "exit_time": [pd.Timestamp(value, tz="UTC").timestamp() for value in (
            "2026-04-10", "2026-05-10", "2026-06-10", "2026-07-10")],
        "value": [1, 2, 3, 4],
    })

    result = audit.training_rows(frame, "2026-06")

    assert result.month.tolist() == ["2026-04", "2026-05"]


def test_training_rows_uses_lexicographic_iso_month_order():
    frame = pd.DataFrame({
        "month": ["2025-12", "2026-01", "2026-02"],
        "exit_time": [pd.Timestamp(value, tz="UTC").timestamp() for value in (
            "2025-12-10", "2026-01-10", "2026-02-10")],
        "value": [1, 2, 3],
    })

    result = audit.training_rows(frame, "2026-01")

    assert result.month.tolist() == ["2025-12"]


def test_training_rows_excludes_trade_that_exits_after_cutoff():
    cutoff = pd.Timestamp("2026-06-01", tz="UTC").timestamp()
    frame = pd.DataFrame({
        "month": ["2026-05", "2026-05"],
        "exit_time": [cutoff - 1, cutoff + 1],
    })
    result = audit.training_rows(frame, "2026-06")
    assert result.exit_time.tolist() == [cutoff - 1]
