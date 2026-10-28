import json

import pandas as pd

import crypto_profit_first_material_target_oct28_renewed_forward as audit


def report():
    return json.loads(audit.OUT.read_text(encoding="utf-8"))


def test_renewed_snapshot_preserves_every_previously_accepted_row():
    parity = report()["integrity"]["previous_snapshot_prefix"]
    assert parity["passes"] is True
    assert parity["old_rows"] == parity["refresh_prefix_rows"] == 749
    assert parity["matched_rows"] == 749
    assert parity["missing_from_refresh"] == 0
    assert parity["unexpected_in_refresh_prefix"] == 0
    assert parity["maximum_signed_return_delta"] == 0.0


def test_saved_cumulative_rows_reproduce_the_forward_metrics():
    rows = pd.read_pickle(audit.CUMULATIVE_ROWS_OUT)
    saved = report()["cumulative_unseen_after_original_oct16_cutoff"]
    for margin in (.03, .05):
        key = f"margin_{int(100 * margin)}pct"
        reproduced = audit.metrics(rows, margin)
        for field in (
                "n", "wr", "return_pct", "dd_pct", "max_loss_streak",
                "negative_months", "trades_per_calendar_day",
                "projected_monthly_return_pct", "literal_2r_runner_hit_rate_pct",
                "near_target_hit_rate_pct"):
            assert reproduced[field] == saved[key][field]
        assert reproduced["economic_gate"] == saved[key]["economic_gate"]


def test_october_now_meets_the_three_percent_economic_objective():
    october = report()["current_october"]["margin_3pct"]
    assert october["n"] == 142
    assert october["wr"] == 74.65
    assert october["return_pct"] == 30.25
    assert october["dd_pct"] == 1.7646
    assert october["trades_per_calendar_day"] >= 3
    assert october["economic_gate"]["passes"] is True


def test_activation_still_waits_for_both_frozen_sample_requirements():
    value = report()
    gate = value["activation_gate"]
    assert gate == {
        "required_new_completed_trades": 60,
        "observed_new_completed_trades": 55,
        "required_calendar_days": 14,
        "observed_calendar_days": 12,
        "met": False,
    }
    assert value["status"] == "FORWARD_GATE_PASSED_AWAIT_MORE_TIME"
    assert value["ready_for_live_orders"] is False
