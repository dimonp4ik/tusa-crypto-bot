import numpy as np

import crypto_long_history_hourly_cross_section_oct20_audit as audit


def tiny_candles():
    times = np.arange(0, 26 * 300, 300, dtype=np.int64)
    result = {}
    for offset, symbol in enumerate(audit.SYMBOLS):
        close = 100 + offset + np.arange(len(times), dtype=float) * (offset + 1) / 100
        result[symbol] = {
            "time": times, "open": close.copy(), "high": close + .1,
            "low": close - .1, "close": close, "volume": np.ones(len(times)),
        }
    return result


def test_hourly_feature_uses_closed_bar_and_next_open():
    frame = audit.feature_rows(tiny_candles(), 1, "reversal")
    assert not frame.empty
    assert (frame.entry_time % 3600 == 0).all()
    for row in frame.itertuples():
        raw = tiny_candles()[row.symbol]
        assert raw["time"][row.entry_index] == row.entry_time
        signal_index = row.entry_index - 1
        expected = raw["close"][signal_index] / raw["close"][signal_index - 12] - 1
        assert np.isclose(row.signal_return, expected)


def test_hidden_years_are_disjoint_from_selection_years():
    assert set(audit.DEVELOPMENT_YEARS).isdisjoint(audit.CONFIRMATION_YEARS)
    assert set(audit.DEVELOPMENT_YEARS).isdisjoint(audit.HIDDEN_YEARS)
    assert set(audit.CONFIRMATION_YEARS).isdisjoint(audit.HIDDEN_YEARS)


def test_label_adapts_string_direction_to_numeric_replay(monkeypatch):
    frame = audit.feature_rows(tiny_candles(), 1, "reversal")

    def fake_add_outcomes(found, candles, target, stop, hold):
        assert set(found.direction.unique()).issubset({-1, 1})
        result = found.copy()
        result["signed_return"] = 0.0
        result["exit_time"] = result.entry_time + 300
        result["outcome_code"] = 1
        return result

    monkeypatch.setattr(audit.replay, "add_outcomes", fake_add_outcomes)
    labeled = audit.label(frame, tiny_candles(), (.5, .5, 12))
    assert set(labeled.direction.unique()).issubset({"LONG", "SHORT"})
