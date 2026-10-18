import pandas as pd

import crypto_exact_2r_profit_first_oct19_audit as audit


def test_eligible_requires_literal_two_r_move_between_half_and_two_percent():
    frame = pd.DataFrame({"risk_pct": [.0024, .0025, .01, .0101]})

    assert audit.eligible(frame).index.tolist() == [1, 2]


def test_admissible_has_no_win_rate_gate():
    value = {
        "n": 100,
        "dd_pct": 10.0,
        "trades_per_calendar_day": 3.0,
        "negative_months": 0,
        "wr": 34.0,
    }

    assert audit.admissible(value)


def test_stop_streak_uses_exit_order():
    rows = [
        {"exit_time": 3, "entry_time": 1, "symbol": "A", "net_r": 2},
        {"exit_time": 1, "entry_time": 0, "symbol": "B", "net_r": -1},
        {"exit_time": 2, "entry_time": 0, "symbol": "C", "net_r": -1},
        {"exit_time": 4, "entry_time": 2, "symbol": "D", "net_r": -1},
    ]

    assert audit.maximum_stop_streak(rows) == 2
