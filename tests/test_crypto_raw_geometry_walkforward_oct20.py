import crypto_raw_geometry_walkforward_oct20_audit as audit


def values(high, low, close=None):
    size = len(high)
    return {
        "time": [1000 + 900 * index for index in range(size)],
        "open": [100.0] * size,
        "high": high,
        "low": low,
        "close": close or [100.0] * size,
    }


def test_geometry_outcome_is_stop_first_on_ambiguous_bar():
    value, outcome, _ = audit.geometry_outcome(
        values([101.1], [99.4]), 0, 1, .01, .005, 1)

    assert outcome == "SL"
    assert round(value, 6) == -.005


def test_geometry_outcome_supports_half_percent_literal_target():
    value, outcome, _ = audit.geometry_outcome(
        values([100.6], [99.9]), 0, 1, .005, .005, 1)

    assert outcome == "TP"
    assert value == .005


def test_rank_key_prefers_admissible_then_worst_month():
    weak = {
        "admissible": False, "development_floor_pct": 20,
        "selection": {"dd_pct": 1, "return_pct": 100},
        "confirmation": {"dd_pct": 1, "return_pct": 100},
    }
    robust = {
        "admissible": True, "development_floor_pct": 1,
        "selection": {"dd_pct": 5, "return_pct": 10},
        "confirmation": {"dd_pct": 5, "return_pct": 10},
    }

    assert audit.rank_key(robust) > audit.rank_key(weak)
