import pandas as pd
import pytest

import crypto_profit_first_exact_post_tp2_trail_oct31_audit as audit


def candles(rows):
    return {
        "time": [900 * i for i in range(len(rows))],
        "open": [row[0] for row in rows],
        "high": [row[1] for row in rows],
        "low": [row[2] for row in rows],
        "close": [row[3] for row in rows],
    }


def test_post_tp2_trail_uses_completed_extreme_on_next_bar_only():
    frame = candles([
        (102.0, 102.2, 101.8, 102.1),  # TP2 activation bar
        (102.1, 103.0, 102.1, 102.9),  # raises next bar's stop to 2.5R
        (102.4, 102.6, 102.3, 102.4),  # gaps through 2.5R -> fills 2.4R
    ])
    value, complete, _ = audit.post_tp2_trail_exit_r(
        frame, 0, direction="LONG", entry=100.0, stop=99.0,
        distance_r=.5, floor_r=2.0, horizon=2)
    assert complete
    assert value == pytest.approx(2.4)


def test_current_bar_high_cannot_move_stop_before_current_bar_low():
    frame = candles([
        (102.0, 102.0, 101.9, 102.0),
        (102.1, 104.0, 101.9, 103.8),
    ])
    value, complete, _ = audit.post_tp2_trail_exit_r(
        frame, 0, direction="LONG", entry=100.0, stop=99.0,
        distance_r=.5, floor_r=2.0, horizon=1)
    assert complete
    assert value == pytest.approx(2.0)


def test_policy_banks_most_at_tp2_and_only_trails_small_tail():
    key = audit.policy_key(.5, 2.0, 12)
    frame = pd.DataFrame([{
        "period": "2024", "sequential_outcome": "TP_RUNNER",
        "net_r": 1.925, "risk_pct": .01,
        f"{key}_r": 3.0, f"{key}_complete": True,
    }])
    result = audit.apply_policy(
        frame, tail_fraction=.2, distance_r=.5,
        floor_r=2.0, horizon=12, direct=True)
    expected = .05 * .5 + .95 * (.8 * 2.0 + .2 * 3.0)
    assert result.iloc[0].net_r == pytest.approx(expected)
    assert result.iloc[0].post_tp2_tail_fraction == pytest.approx(.2)
