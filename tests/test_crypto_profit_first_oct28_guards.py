import json

import crypto_profit_first_stop_guard_oct28_audit as stop_guard
import crypto_profit_first_third_target_oct28_audit as third


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_third_target_grid_cannot_improve_both_development_stages():
    value = load(third.OUT)
    assert value["grid"]["candidates"] == 10
    assert value["selected_before_holdouts"] is None
    assert value["direct_xperp"]["baseline"]["n"] == 749
    assert value["status"] == "REJECTED_DEVELOPMENT"
    assert value["ready_for_live_orders"] is False


def test_higher_quality_threshold_kills_profit_and_frequency():
    value = load(stop_guard.OUT)
    assert value["grid"]["candidate_thresholds"] == 7
    assert value["selected_before_holdouts"] is None
    assert value["direct_xperp"]["baseline"]["n"] == 749
    assert value["status"] == "REJECTED_DEVELOPMENT"
    assert value["orders_possible"] is False
