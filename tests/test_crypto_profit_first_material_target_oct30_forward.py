import json

import pandas as pd

import crypto_profit_first_material_target_oct30_forward as audit


def report():
    return json.loads(audit.OUT.read_text(encoding="utf-8"))


def test_oct30_snapshot_preserves_the_full_previous_prefix():
    parity = report()["integrity"]["previous_snapshot_prefix"]
    assert parity["passes"] is True
    assert parity["old_rows"] == parity["refresh_prefix_rows"] == 750
    assert parity["matched_rows"] == 750
    assert parity["missing_from_refresh"] == 0
    assert parity["unexpected_in_refresh_prefix"] == 0
    assert parity["maximum_signed_return_delta"] == 0.0


def test_saved_forward_rows_reproduce_the_reported_metrics():
    rows = pd.read_pickle(audit.CUMULATIVE_ROWS_OUT)
    saved = report()["cumulative_unseen_after_original_oct16_cutoff"]
    for margin in (.03, .05):
        key = f"margin_{int(100 * margin)}pct"
        observed = audit.metrics(rows, margin)
        for field in (
                "n", "wr", "return_pct", "dd_pct", "max_loss_streak",
                "negative_months", "trades_per_calendar_day",
                "projected_monthly_return_pct", "literal_2r_runner_hit_rate_pct",
                "near_target_hit_rate_pct"):
            assert observed[field] == saved[key][field]
        assert observed["economic_gate"] == saved[key]["economic_gate"]


def test_frozen_crypto_route_passes_the_predeclared_forward_gate():
    value = report()
    forward = value["cumulative_unseen_after_original_oct16_cutoff"]["margin_3pct"]
    assert forward["n"] == 69
    assert forward["wr"] == 78.26
    assert forward["return_pct"] == 15.6213
    assert forward["dd_pct"] == 1.0381
    assert forward["trades_per_calendar_day"] == 4.929
    assert forward["projected_monthly_return_pct"] == 36.4838
    assert forward["economic_gate"]["passes"] is True
    assert value["activation_gate"] == {
        "required_new_completed_trades": 60,
        "observed_new_completed_trades": 69,
        "required_calendar_days": 14,
        "observed_calendar_days": 14,
        "met": True,
    }
    assert value["status"] == "ACTIVATION_GATE_PASSED"
    assert value["ready_for_live_orders"] is False


def test_october_meets_the_requested_three_percent_return_gate():
    october = report()["current_october"]["margin_3pct"]
    assert october["n"] == 156
    assert october["wr"] == 74.36
    assert october["return_pct"] == 31.0841
    assert october["dd_pct"] == 1.7646
    assert october["economic_gate"]["passes"] is True
