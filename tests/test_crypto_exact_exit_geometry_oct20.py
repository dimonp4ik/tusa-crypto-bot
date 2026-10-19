import pandas as pd

import crypto_exact_exit_geometry_oct20_audit as audit


def test_eligible_uses_literal_target_price_move():
    frame = pd.DataFrame({"risk_pct": [.0024, .0025, .01, .0101]})

    selected = audit.eligible(frame, 2.0)

    assert selected.index.tolist() == [1, 2]


def test_development_admissible_has_no_win_rate_gate():
    value = {
        "n": 100, "dd_pct": 10.0, "trades_per_calendar_day": 3.0,
        "wr": 1.0,
    }

    assert audit.development_admissible(value)
