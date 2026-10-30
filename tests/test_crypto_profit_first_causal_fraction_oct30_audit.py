import json

import pandas as pd

import crypto_profit_first_causal_fraction_oct30_audit as audit


def report():
    return json.loads(audit.OUT.read_text(encoding="utf-8"))


def test_fixed_fraction_exactly_reproduces_the_frozen_historical_rows():
    frozen = pd.read_pickle(audit.HISTORICAL_ROWS)
    reproduced = audit.fixed_fraction(frozen)
    assert len(reproduced) == len(frozen) == 3221
    assert (reproduced.net_r - frozen.net_r).abs().max() == 0.0
    assert (
        reproduced.signed_price_return - frozen.signed_price_return
    ).abs().max() == 0.0


def test_controller_only_observes_strictly_completed_prior_outcomes():
    rows = pd.DataFrame([
        {"entry_time": 0, "exit_time": 100, "symbol": "A",
         "direction": "LONG", "barbell_outcome": "TP_RUNNER",
         "net_r": 1.925, "risk_pct": .01},
        {"entry_time": 100, "exit_time": 102, "symbol": "B",
         "direction": "LONG", "barbell_outcome": "TP_BE",
         "net_r": .025, "risk_pct": .01},
        {"entry_time": 101, "exit_time": 103, "symbol": "C",
         "direction": "LONG", "barbell_outcome": "TP_BE",
         "net_r": .025, "risk_pct": .01},
    ])
    observed = audit.causal_fraction(
        rows, window=1, threshold=.5, defensive_fraction=.5)
    assert observed.runner_history_n.tolist() == [0, 0, 1]
    assert observed.runner_history_hit_rate.iloc[:2].isna().all()
    assert observed.runner_history_hit_rate.tolist()[2] == 1.0
    assert observed.attrs["strictly_prior_exit_violations"] == 0


def test_adaptive_bank_fraction_is_rejected_before_holdouts():
    value = report()
    assert value["candidate_count"] == 27
    assert value["selected_before_holdouts"] is None
    assert value["status"] == "REJECTED_DEVELOPMENT"
    assert value["ready_for_live_orders"] is False
    assert value["orders_possible"] is False
    assert all(
        item["strictly_prior_exit_violations"] == 0
        for item in value["top_development"]
    )
    baseline = value["direct_xperp"]["baseline"]["after_oct16"]
    assert baseline["n"] == 69
    assert baseline["return_pct"] == 15.6213
    assert baseline["projected_monthly_return_pct"] == 36.4838
