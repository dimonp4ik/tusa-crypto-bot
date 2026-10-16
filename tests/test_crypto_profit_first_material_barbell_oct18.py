import crypto_profit_first_material_barbell_oct18_audit as audit
import pandas as pd
import pytest


def test_material_price_move_ranges_are_enforced():
    assert audit.legacy.RISK_PCT_MIN == .01
    assert audit.legacy.RISK_PCT_MAX == .02
    assert .5 * audit.legacy.RISK_PCT_MIN == .005
    assert 2.0 * audit.legacy.RISK_PCT_MAX == .04


def test_profit_first_grid_keeps_most_size_for_runner():
    assert min(audit.FRACTIONS) == .10
    assert .30 in audit.FRACTIONS


def test_development_gate_does_not_require_win_rate():
    value = {"all": {"n": 200, "return_pct": 20.0, "dd_pct": 8.0,
                     "trades_per_calendar_day": 3.1, "wr": 40.0},
             "periods": {"x": {"return_pct": 1.0}}}
    assert audit.development_ok(value)


def test_sparse_material_module_can_enter_supplement_funnel():
    value = {"all": {"n": 200, "return_pct": 20.0, "dd_pct": 8.0,
                     "trades_per_calendar_day": 1.0, "wr": 40.0},
             "periods": {"x": {"return_pct": 1.0}}}
    assert audit.development_ok(value)


def test_objective_requires_every_month_at_thirty_percent():
    value = {"all": {"monthly_return_pct": {"a": 31.0, "b": 29.0},
                     "dd_pct": 5.0, "trades_per_calendar_day": 3.0},
             "periods": {}}
    assert not audit.objective_ok(value)


def test_direct_targets_join_by_signal_identity(tmp_path):
    keys = {"symbol": ["BTCUSDT"], "entry_time": [1.0],
            "direction": ["LONG"], "family": ["breakout"]}
    far = pd.DataFrame({**keys, "row_id": [99], "outcome": ["TP1"],
                        "exit_time": [3.0]})
    near = pd.DataFrame({**keys, "row_id": [7], "outcome": ["SL"],
                         "exit_time": [2.0]})
    far_path, near_path = tmp_path / "far.csv", tmp_path / "near.csv"
    far.to_csv(far_path, index=False)
    near.to_csv(near_path, index=False)
    joined = audit.join_direct_outcomes(far_path, near_path)
    assert joined.loc[0, "near_outcome"] == "SL"


def test_direct_targets_reject_missing_signal(tmp_path):
    far = pd.DataFrame({"symbol": ["BTCUSDT"], "entry_time": [1.0],
                        "direction": ["LONG"], "family": ["breakout"],
                        "outcome": ["TP1"], "exit_time": [3.0]})
    near = far.assign(entry_time=2.0)
    far_path, near_path = tmp_path / "far.csv", tmp_path / "near.csv"
    far.to_csv(far_path, index=False)
    near.to_csv(near_path, index=False)
    with pytest.raises(ValueError):
        audit.join_direct_outcomes(far_path, near_path)
