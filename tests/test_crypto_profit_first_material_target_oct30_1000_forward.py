import json

import pandas as pd

import crypto_profit_first_material_target_oct30_1000_forward as audit


def report():
    return json.loads(audit.OUT.read_text(encoding="utf-8"))


def test_refresh_preserves_every_overlapping_candle_and_trade():
    integrity = report()["integrity"]
    candles = integrity["candle_overlap"]
    assert candles["passes"] is True
    assert candles["symbols"] == 12
    assert candles["overlap_bars"] == 184275
    assert candles["value_mismatches"] == 0
    assert all(value["new_bars_after_old_end"] == 30
               for value in candles["per_symbol"].values())

    parity = integrity["previous_snapshot_prefix"]
    assert parity["passes"] is True
    assert parity["old_rows"] == parity["refresh_prefix_rows"] == 764
    assert parity["matched_rows"] == 764
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


def test_frozen_route_still_passes_but_latest_partial_day_is_negative():
    value = report()
    forward = value["cumulative_unseen_after_original_oct16_cutoff"]["margin_3pct"]
    assert forward["n"] == 73
    assert forward["wr"] == 76.71
    assert forward["return_pct"] == 14.8011
    assert forward["dd_pct"] == 1.0381
    assert forward["trades_per_calendar_day"] == 5.214
    assert forward["projected_monthly_return_pct"] == 34.4175
    assert forward["economic_gate"]["passes"] is True

    latest = value["newly_revealed_after_oct30_0030_snapshot"]["margin_3pct"]
    assert latest["n"] == 4
    assert latest["wr"] == 50.0
    assert latest["return_pct"] == -0.7094
    assert latest["economic_gate"]["passes"] is False
    assert value["status"] == "ACTIVATION_GATE_PASSED"
    assert value["ready_for_live_orders"] is False


def test_october_remains_above_the_requested_return_gate():
    october = report()["current_october"]["margin_3pct"]
    assert october["n"] == 160
    assert october["wr"] == 73.75
    assert october["return_pct"] == 30.1542
    assert october["dd_pct"] == 1.7646
    assert october["economic_gate"]["passes"] is True
