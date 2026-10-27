import json

import pandas as pd

import crypto_profit_first_material_target_oct28_forward as audit


def report():
    return json.loads(audit.OUT.read_text(encoding="utf-8"))


def test_previous_snapshot_is_an_exact_prefix():
    parity = report()["integrity"]["previous_snapshot_prefix"]
    assert parity["passes"] is True
    assert parity["old_rows"] == parity["refresh_prefix_rows"]
    assert parity["old_rows"] == parity["matched_rows"]
    assert parity["missing_from_refresh"] == 0
    assert parity["unexpected_in_refresh_prefix"] == 0
    assert parity["maximum_signed_return_delta"] == 0.0


def test_saved_cumulative_rows_reproduce_money_and_economic_metrics():
    value = report()["cumulative_unseen_after_original_oct16_cutoff"]
    rows = pd.read_pickle(audit.CUMULATIVE_ROWS_OUT)
    for margin in (.03, .05):
        key = f"margin_{int(margin*100)}pct"
        observed = audit.metrics(rows, margin)
        for field in (
                "n", "wr", "return_pct", "dd_pct", "max_loss_streak",
                "negative_months", "trades_per_calendar_day",
                "projected_monthly_return_pct",
                "literal_2r_runner_hit_rate_pct", "near_target_hit_rate_pct"):
            assert observed[field] == value[key][field]
        assert observed["economic_gate"] == value[key]["economic_gate"]


def test_new_slice_and_activation_are_reproduced_from_saved_rows():
    value = report()
    fresh = value["newly_revealed_after_oct27_0418_snapshot"]["margin_3pct"]
    rows = pd.read_pickle(audit.NEW_ROWS_OUT)
    assert fresh["n"] == len(rows)
    cumulative = value["cumulative_unseen_after_original_oct16_cutoff"]["margin_3pct"]
    gate = value["activation_gate"]
    expected = bool(
        value["forward_gate_pass"]
        and cumulative["n"] >= gate["required_new_completed_trades"]
        and cumulative["observed_calendar_days"] >= gate["required_calendar_days"])
    assert gate["met"] is expected
    assert value["ready_for_live_orders"] is False


def test_october_three_percent_route_keeps_the_economic_gate():
    value = report()["current_october"]["margin_3pct"]
    assert value["projected_monthly_return_pct"] >= 30
    assert value["economic_gate"]["passes"] is True

