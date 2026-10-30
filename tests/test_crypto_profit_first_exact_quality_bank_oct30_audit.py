import json

import pandas as pd
import pytest

import crypto_profit_first_exact_quality_bank_oct30_audit as audit


def report():
    return json.loads(audit.OUT.read_text(encoding="utf-8"))


def test_defensive_bank_applies_only_below_the_frozen_quality_cutoff():
    frame = pd.DataFrame({
        "model_probability": [.34, .35],
        "a0p5_l0_runner_r": [0.0, 2.0],
        "a0p5_l0_banked": [True, True],
        "risk_pct": [.01, .01],
    })
    value = audit.apply_policy(
        frame, threshold=.345, defensive_fraction=1.0)

    assert value.first_fraction.tolist() == [.05 + .95, .05]
    assert value.net_r.tolist() == pytest.approx([.5, 1.925])
    assert value.signed_price_return.tolist() == pytest.approx([.005, .01925])


def test_quality_bank_improves_historical_holdouts_but_fails_fresh_exact_data():
    value = report()
    assert value["candidate_count"] == 25
    assert value["development_pass_count"] == 5
    selected = value["selected_before_holdouts"]
    assert selected["probability_below"] == .345
    assert selected["defensive_first_fraction"] == 1.0
    assert value["historical_holdout_pass"] is True
    assert value["fresh_exact_forward_pass"] is False

    baseline = value["baseline"]["direct_xperp"]
    candidate = value["direct_xperp"]
    assert baseline["october"]["return_pct"] == 15.7126
    assert candidate["october"]["return_pct"] == 13.4266
    assert baseline["after_oct16"]["n"] == candidate["after_oct16"]["n"] == 93
    assert baseline["after_oct16"]["return_pct"] == 9.6223
    assert candidate["after_oct16"]["return_pct"] == 8.9697


def test_failed_quality_bank_cannot_reach_runtime():
    value = report()
    assert value["status"] == "REJECTED_FRESH_EXACT_FORWARD"
    assert value["robust_improvement"] is False
    assert value["runtime_change_allowed"] is False
    assert value["ready_for_live_orders"] is False
    assert value["orders_possible"] is False
