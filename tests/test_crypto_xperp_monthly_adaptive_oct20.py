import crypto_xperp_monthly_adaptive_oct20_audit as audit


def test_previous_month_crosses_year_boundary():
    assert audit.previous_month("2026-01") == "2025-12"


def test_policy_key_prioritizes_frequency_before_backtest_return():
    sparse = {
        "hourly_cap": 1,
        "metrics": {
            "trades_per_calendar_day": 2.9,
            "dd_pct": 1.0,
            "negative_months": 0,
            "return_pct": 100.0,
        },
    }
    frequent = {
        "hourly_cap": 2,
        "metrics": {
            "trades_per_calendar_day": 3.0,
            "dd_pct": 9.0,
            "negative_months": 0,
            "return_pct": 1.0,
        },
    }

    assert audit.policy_key(frequent) > audit.policy_key(sparse)


def test_policy_key_prefers_nonnegative_calibration_month():
    loss = {
        "hourly_cap": 1,
        "metrics": {
            "trades_per_calendar_day": 3.0,
            "dd_pct": 2.0,
            "negative_months": 1,
            "return_pct": -1.0,
        },
    }
    profit = {
        "hourly_cap": 2,
        "metrics": {
            "trades_per_calendar_day": 3.0,
            "dd_pct": 8.0,
            "negative_months": 0,
            "return_pct": .1,
        },
    }

    assert audit.policy_key(profit) > audit.policy_key(loss)
