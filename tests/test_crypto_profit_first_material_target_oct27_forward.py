import json

import pandas as pd

import crypto_profit_first_material_target_oct27_forward as audit


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


def test_new_slice_is_profitable_but_activation_waits_for_sample_size():
    value = report()
    fresh = value["newly_revealed_after_oct26_snapshot"]["margin_3pct"]
    assert fresh["n"] == 4
    assert fresh["return_pct"] > 0
    assert fresh["dd_pct"] == 0
    assert value["activation_gate"]["observed_new_completed_trades"] < 60
    assert value["activation_gate"]["observed_calendar_days"] < 14
    assert value["activation_gate"]["met"] is False
    assert value["ready_for_live_orders"] is False


def test_october_three_percent_route_passes_projection_not_activation():
    value = report()["current_october"]["margin_3pct"]
    assert value["return_pct"] < 30
    assert value["projected_monthly_return_pct"] >= 30
    assert value["economic_gate"]["passes"] is True
