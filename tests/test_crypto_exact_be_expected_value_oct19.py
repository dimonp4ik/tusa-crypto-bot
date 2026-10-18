import pandas as pd

import crypto_exact_be_expected_value_oct19_audit as audit


def test_eligible_requires_literal_runner_move_between_half_and_two_percent():
    frame = pd.DataFrame({"risk_pct": [.0024, .0025, .01, .0101]})

    selected = audit.eligible(frame)

    assert selected.index.tolist() == [1, 2]


def test_month_floor_requires_every_named_month():
    value = {"monthly_return_pct": {"2026-07": 31.0, "2026-08": 28.0}}

    assert audit.month_floor(value, ("2026-07", "2026-08")) == 28.0
    assert audit.month_floor(value, ("2026-07", "2026-09")) == -999.0


def test_admissible_uses_profit_rails_without_a_win_rate_gate():
    value = {
        "n": 100,
        "dd_pct": 10.0,
        "trades_per_calendar_day": 3.0,
        "negative_months": 0,
        "wr": 20.0,
    }

    assert audit.admissible(value)
    value["negative_months"] = 1
    assert not audit.admissible(value)
