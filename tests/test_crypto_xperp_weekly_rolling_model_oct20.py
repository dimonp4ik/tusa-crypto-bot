import json
from pathlib import Path

import pandas as pd

import crypto_xperp_weekly_rolling_model_oct20_audit as audit


def test_causal_parts_require_closed_training_and_calibration_outcomes():
    start = pd.Timestamp("2026-06-01", tz="UTC")
    end = start + pd.Timedelta(days=7)
    calibration_start = start - pd.Timedelta(days=audit.CALIBRATION_DAYS)
    frame = pd.DataFrame({
        "entry_time": [
            (calibration_start - pd.Timedelta(days=5)).timestamp(),
            (calibration_start - pd.Timedelta(hours=1)).timestamp(),
            (calibration_start + pd.Timedelta(days=1)).timestamp(),
            (start + pd.Timedelta(days=1)).timestamp(),
        ],
        "exit_time": [
            (calibration_start - pd.Timedelta(seconds=1)).timestamp(),
            (calibration_start + pd.Timedelta(hours=1)).timestamp(),
            (start - pd.Timedelta(seconds=1)).timestamp(),
            (end + pd.Timedelta(days=1)).timestamp(),
        ],
    })
    train, calibration, test = audit.causal_parts(frame, start, end, 21)
    assert len(train) == 1
    assert len(calibration) == 1
    assert len(test) == 1


def test_report_is_rejected_and_sources_match():
    report = json.loads(Path(audit.OUT).read_text(encoding="utf-8"))
    assert report["status"] == "REJECTED_OBJECTIVE"
    assert report["ready_for_live_orders"] is False
    assert report["objective_pass"] is False
    for source, expected in report["sources"].items():
        assert audit.digest(Path(source)) == expected
