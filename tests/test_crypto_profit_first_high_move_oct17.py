import crypto_profit_first_high_move_oct17_audit as audit


def test_selection_does_not_gate_on_win_rate():
    value = {"return_pct": 15.0, "dd_pct": 12.0,
             "trades_per_calendar_day": 3.2, "wr": 24.0}
    assert audit.selection_ok(value)


def test_objective_requires_thirty_percent_in_every_month():
    value = {"months": 12, "months_ge_30pct": 12,
             "dd_pct": 9.0, "trades_per_calendar_day": 3.0}
    assert audit.objective_ok(value)
    value["months_ge_30pct"] = 11
    assert not audit.objective_ok(value)
