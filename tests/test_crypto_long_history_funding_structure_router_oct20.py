import numpy as np
import pandas as pd

import crypto_long_history_funding_structure_router_oct20_audit as audit


def candles():
    size = 400
    close = np.full(size, 100.0)
    return {"BTCUSDT": {
        "time": np.arange(size) * 300,
        "open": close.copy(), "high": np.full(size, 101.0),
        "low": np.full(size, 99.0), "close": close,
        "volume": np.ones(size),
    }}


def test_structure_uses_only_bars_before_entry():
    source = candles()
    frame = pd.DataFrame({
        "symbol": ["BTCUSDT"], "entry_index": [350],
        "direction_n": [1], "target_pct": [.5], "stop_pct": [.5],
    })
    first = audit.add_structure(frame, source)
    source["BTCUSDT"]["high"][350:] = 1000
    source["BTCUSDT"]["low"][350:] = 1
    second = audit.add_structure(frame, source)
    assert first.iloc[0].room_to_24h_level_pct == second.iloc[0].room_to_24h_level_pct
    assert first.iloc[0].backing_24h_distance_pct == second.iloc[0].backing_24h_distance_pct


def test_room_compares_prior_level_with_literal_target():
    frame = pd.DataFrame({
        "symbol": ["BTCUSDT"], "entry_index": [350],
        "direction_n": [1], "target_pct": [2.0], "stop_pct": [.5],
    })
    found = audit.add_structure(frame, candles())
    assert found.iloc[0].room_state == "blocked"


def test_short_history_does_not_wrap_future_close_for_stretch():
    source = candles()
    frame = pd.DataFrame({
        "symbol": ["BTCUSDT"], "entry_index": [30],
        "direction_n": [1], "target_pct": [.5], "stop_pct": [.5],
    })
    assert audit.add_structure(frame, source).empty


def test_hidden_years_are_disjoint_from_learning_years():
    assert set(audit.base.HIDDEN_YEARS).isdisjoint(
        set(audit.base.DEVELOPMENT_YEARS) | set(audit.base.CONFIRMATION_YEARS))
