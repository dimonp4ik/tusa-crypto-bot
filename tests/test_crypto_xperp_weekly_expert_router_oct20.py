import pandas as pd

import crypto_xperp_weekly_expert_router_oct20_audit as audit


def test_causal_parts_require_labels_closed_before_each_boundary():
    start = pd.Timestamp("2026-06-01", tz="UTC")
    end = start + pd.Timedelta(days=7)
    calibration_start = start - pd.Timedelta(days=audit.CALIBRATION_DAYS)
    frame = pd.DataFrame([
        {"entry_time": (calibration_start - pd.Timedelta(days=2)).timestamp(),
         "exit_time": (calibration_start - pd.Timedelta(days=1)).timestamp()},
        {"entry_time": (calibration_start + pd.Timedelta(days=1)).timestamp(),
         "exit_time": (start - pd.Timedelta(hours=1)).timestamp()},
        {"entry_time": (start - pd.Timedelta(hours=1)).timestamp(),
         "exit_time": (start + pd.Timedelta(hours=1)).timestamp()},
        {"entry_time": (start + pd.Timedelta(days=1)).timestamp(),
         "exit_time": (start + pd.Timedelta(days=2)).timestamp()},
    ])

    train, calibration, test = audit.causal_parts(frame, start, end)

    assert train.index.tolist() == [0]
    assert calibration.index.tolist() == [1]
    assert test.index.tolist() == [3]


def test_flat_rule_requires_profitable_frequent_low_dd_calibration():
    item = {"policy": {"metrics": {
        "return_pct": 1.0, "negative_months": 0,
        "dd_pct": 5.0, "trades_per_calendar_day": 3.0,
    }}}

    assert audit.should_trade(item)
    item["policy"]["metrics"]["return_pct"] = 0.0
    assert not audit.should_trade(item)
