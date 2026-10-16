import pandas as pd

import crypto_profit_first_expanded_literal_oct17_audit as audit


def test_feature_columns_can_exclude_symbol_identity():
    frame = pd.DataFrame({"symbol": ["BTCUSDT", "ETHUSDT"]})
    transferable = audit.feature_columns(frame, False)
    identified = audit.feature_columns(frame, True)
    assert not any(name.startswith("symbol_") for name in transferable)
    assert {"symbol_BTCUSDT", "symbol_ETHUSDT"}.issubset(identified)


def test_selection_is_profit_first_without_win_rate_gate():
    value = {"return_pct": 12.0, "dd_pct": 9.0,
             "trades_per_calendar_day": 3.1, "wr": 34.0}
    assert audit.selection_ok(value)


def test_objective_requires_every_month_to_reach_target():
    value = {"months": 12, "months_ge_30pct": 11, "dd_pct": 8.0,
             "trades_per_calendar_day": 4.0}
    assert not audit.objective_ok(value)
