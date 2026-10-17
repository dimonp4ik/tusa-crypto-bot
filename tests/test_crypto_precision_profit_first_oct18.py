import pandas as pd
import pytest

import crypto_precision_profit_first_oct18_audit as audit


def _far_rows():
    return pd.DataFrame([
        {"symbol": "BTCUSDT", "entry_time": 1.0, "direction": "LONG",
         "family": "breakout", "outcome": "TP2", "exit_time": 10.0},
        {"symbol": "ETHUSDT", "entry_time": 2.0, "direction": "SHORT",
         "family": "range_reversion", "outcome": "SL", "exit_time": 11.0},
    ])


def test_direct_target_paths_join_by_signal_identity(tmp_path, monkeypatch):
    far = tmp_path / "far.csv"
    near = tmp_path / "near.csv"
    _far_rows().to_csv(far, index=False)
    pd.DataFrame([
        {"symbol": "ETHUSDT", "entry_time": 2.0, "direction": "SHORT",
         "family": "range_reversion", "outcome": "TP2", "exit_time": 4.0},
        {"symbol": "BTCUSDT", "entry_time": 1.0, "direction": "LONG",
         "family": "breakout", "outcome": "SL", "exit_time": 3.0},
    ]).to_csv(near, index=False)
    monkeypatch.setattr(audit, "FRESH_FAR", far)
    monkeypatch.setattr(audit, "FRESH_NEAR", near)

    joined = audit.direct_frame().set_index("symbol")

    assert joined.loc["BTCUSDT", "near_outcome"] == "SL"
    assert joined.loc["ETHUSDT", "near_outcome"] == "TP2"


def test_direct_target_paths_reject_duplicate_signal(tmp_path, monkeypatch):
    far = tmp_path / "far.csv"
    near = tmp_path / "near.csv"
    rows = _far_rows()
    rows.to_csv(far, index=False)
    pd.concat([rows, rows.iloc[[0]]], ignore_index=True).to_csv(near, index=False)
    monkeypatch.setattr(audit, "FRESH_FAR", far)
    monkeypatch.setattr(audit, "FRESH_NEAR", near)

    with pytest.raises(ValueError, match="duplicate signal identities"):
        audit.direct_frame()
