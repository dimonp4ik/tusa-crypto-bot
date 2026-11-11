import pytest

from shadow_regime_health import filter_rows


def row(start, end, value, direction="SHORT"):
    return dict(symbol="TEST", direction=direction, btc_regime="bull",
                entry_time=start, exit_time=end, signed_price_return=value)


def apply(rows, **kwargs):
    return filter_rows(rows, fields=("direction", "btc_regime"), days=1,
                       minimum=1, **kwargs)[0]


def test_equal_time_exit_is_unavailable_and_future_values_cannot_gate_entry():
    rows = [row(0, 10, -1), row(10, 20, 999), row(11, 30, 1)]
    assert [r["entry_time"] for r in apply(rows)] == [0, 10]
    rows[1]["signed_price_return"] = -999
    assert [r["entry_time"] for r in apply(rows)] == [0, 10]


def test_rejected_shadow_winner_can_restore_group_health():
    rows = [row(0, 1, -1), row(2, 3, 2), row(4, 5, 0)]
    assert [r["entry_time"] for r in apply(rows)] == [0, 4]


def test_direction_is_separate_and_old_history_expires():
    rows = [row(0, 1, -1), row(2, 3, 0, "LONG"), row(86402, 86403, 0)]
    assert len(apply(rows)) == 3


def test_malformed_outcome_fails_instead_of_silently_changing_sample():
    with pytest.raises(ValueError, match="invalid shadow outcome"):
        apply([row(2, 1, -1)])
