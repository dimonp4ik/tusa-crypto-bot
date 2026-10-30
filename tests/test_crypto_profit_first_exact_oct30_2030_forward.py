import json

import crypto_profit_first_exact_oct30_2030_forward as audit


def report():
    return json.loads(audit.OUT.read_text(encoding="utf-8"))


def test_exact_refresh_preserves_candles_and_every_old_accepted_trade():
    value = report()["integrity"]
    candles = value["raw_candle_overlap"]
    prefix = value["exact_accepted_prefix"]
    assert candles["passes"] is True
    assert candles["symbols"] == 12
    assert candles["overlap_bars"] == 184385
    assert candles["value_mismatches"] == 0
    assert prefix["passes"] is True
    assert prefix["old_rows"] == prefix["new_prefix_rows"] == 825
    assert prefix["maximum_net_r_delta"] == 0
    assert prefix["maximum_exit_time_delta_seconds"] == 0


def test_new_exact_slice_uses_live_exit_order():
    value = report()
    fresh = value["newly_accepted_after_morning_snapshot"]["margin_3pct"]
    assert value["new_accepted_rows"] == 11
    assert fresh["outcome_counts"] == {"LOCK": 7, "SL": 2, "TP_RUNNER": 2}
    assert fresh["wr"] == 81.82
    assert fresh["literal_2r_before_be_rate_pct"] == 18.1818
    assert fresh["tp1_or_better_rate_pct"] == 81.8182


def test_five_percent_margin_passes_fresh_economic_gate_but_three_does_not():
    unseen = report()["current"]["unseen_after_oct16"]
    three = unseen["margin_3pct"]
    five = unseen["margin_5pct"]
    assert three["n"] == five["n"] == 93
    assert three["projected_monthly_return_pct"] == 20.1705
    assert three["economic_gate"]["passes"] is False
    assert five["projected_monthly_return_pct"] == 35.5206
    assert five["economic_gate"]["passes"] is True
    assert five["dd_pct"] == 2.037


def test_report_stays_shadow_only_despite_five_percent_forward_pass():
    value = report()
    assert value["status"] == "EXACT_FORWARD_GATE_NOT_PASSED"
    assert value["ready_for_live_orders"] is False
    assert value["orders_possible"] is False
    assert value["policy"]["orders"] == "market only"
    assert value["policy"]["position_fraction"] == "client setting"
