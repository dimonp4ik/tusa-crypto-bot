import json

import pandas as pd
import pytest

import crypto_profit_first_exact_tp1_volume_bank_oct30_audit as audit


def report():
    return json.loads(audit.OUT.read_text(encoding="utf-8"))


def test_extra_market_close_uses_observed_hit_candle_close_only_while_open():
    frame = pd.DataFrame({
        "a0p5_l0_runner_r": [0.0, 2.0, 2.0, -1.0],
        "a0p5_l0_banked": [True, True, True, False],
        "runner_open_after_hit_close": [True, True, False, False],
        "hit_volume_ratio": [1.0, 2.0, 1.0, float("nan")],
        "hit_close_r": [.4, .4, .4, float("nan")],
        "risk_pct": [.01, .01, .01, .01],
    })
    value = audit.apply_policy(
        frame, volume_threshold=1.5, extra_close_fraction=.75)

    assert value.tp1_extra_close_fraction.tolist() == [.75, 0.0, 0.0, 0.0]
    assert value.net_r.tolist() == pytest.approx([
        .05 * .5 + .75 * .4 + .20 * 0.0,
        .05 * .5 + .95 * 2.0,
        .05 * .5 + .95 * 2.0,
        -1.0,
    ])


def test_selected_policy_dominates_sealed_history_and_exact_forward():
    value = report()
    selected = value["selected_before_holdouts"]
    assert value["candidate_count"] == 20
    assert value["development_pass_count"] == 12
    assert selected["volume_ratio_below"] == 1.5
    assert selected["extra_close_fraction_at_hit_bar_close"] == .75
    assert selected["total_closed_after_weak_tp1_bar"] == .80
    assert value["historical_holdout_pass"] is True
    assert value["fresh_exact_forward_pass"] is True
    assert value["robust_improvement"] is True

    baseline = value["baseline"]["direct_xperp"]
    candidate = value["direct_xperp"]
    assert baseline["october"]["margin_3pct"]["return_pct"] == 15.7126
    assert candidate["october"]["margin_3pct"]["return_pct"] == 16.052
    assert candidate["october"]["margin_3pct"]["dd_pct"] == 2.497
    assert baseline["after_oct16"]["margin_3pct"]["return_pct"] == 9.6223
    assert candidate["after_oct16"]["margin_3pct"]["return_pct"] == 10.7763
    assert candidate["after_oct16"]["margin_5pct"][
        "projected_monthly_return_pct"] == 40.3817


def test_candidate_remains_shadow_only_until_genuinely_new_forward():
    value = report()
    assert value["status"] == (
        "EXACT_TP1_VOLUME_BANK_SHADOW_CHALLENGER_AWAIT_NEW_FORWARD")
    assert value["runtime_change_allowed"] is False
    assert value["ready_for_live_orders"] is False
    assert value["orders_possible"] is False
    assert value["protocol"]["orders"] == "market only"
    assert value["protocol"]["position_fraction"] == "client setting"
