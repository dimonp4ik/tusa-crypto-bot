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


def test_canonical_history_assigns_overlap_to_later_period():
    rows = _far_rows()
    earlier = rows.iloc[[0]].assign(period="2021h2")
    later = rows.iloc[[0]].assign(period="2022h1")
    unique = rows.iloc[[1]].assign(period="2021h2")

    result = audit.canonical_history(pd.concat(
        [later, unique, earlier], ignore_index=True))

    overlap = result[result.symbol.eq("BTCUSDT")]
    assert overlap.period.tolist() == ["2022h1"]
    assert result.attrs["duplicate_rows_removed"] == 1


def test_rejected_exact_replay_invalidates_optimistic_report():
    report = {
        "status": "PROFIT_FIRST_5PCT_SHADOW_CANDIDATE",
        "ready_for_live_orders": False,
        "objective_checks": {"legacy_gate": True},
    }
    exact = {
        "status": "REJECTED_OBJECTIVE",
        "direct_xperp": {"margin_3pct": {"return_pct": 0.4838}},
    }

    result = audit.invalidate_with_exact_replay(report, exact)

    assert result["status"] == "INVALIDATED_EXIT_PATH"
    assert result["legacy_status_before_exact_replay"] == (
        "PROFIT_FIRST_5PCT_SHADOW_CANDIDATE")
    assert not result["objective_checks"]["exact_runtime_exit_path_validated"]
    assert result["invalidation"]["exact_direct_xperp"] == exact["direct_xperp"]
