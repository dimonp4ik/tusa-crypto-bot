import numpy as np
import pandas as pd

from crypto_5m_sweep_delayed_confirmation_oct21_audit import (
    confirmation_entries,
)


def candle_fixture():
    size = 32
    open_ = np.full(size, 100.0)
    high = np.full(size, 101.0)
    low = np.full(size, 99.0)
    close = np.full(size, 100.0)
    # Signal at 20 is a low sweep/reclaim.  Bar 21 confirms upward and the
    # market entry must therefore be bar 22, never bar 21.
    low[20] = 97.0
    close[20] = 99.5
    open_[21] = 99.5
    close[21] = 100.5
    high[21] = 100.7
    low[21] = 99.4
    times = np.arange(size, dtype=np.int64) * 300
    candle = {"open": open_, "high": high, "low": low, "close": close,
              "volume": np.ones(size), "time": times}
    entries = pd.DataFrame({
        "symbol": ["BTCUSDT"], "lookback": [12], "entry_index": [21],
        "entry_time": [times[21]], "direction": np.array([1], dtype=np.int8),
        "sweep_depth_atr": [.5], "wick_fraction": [.7],
        "volume_ratio": [2.0], "prior_move_atr": [.5],
        "close_back_atr": [.2], "score": [.7], "year": [2020],
        "month": [202001],
    })
    return candle, entries


def test_confirmation_enters_only_after_confirmation_bar():
    candle, entries = candle_fixture()
    result = confirmation_entries(
        entries, {"BTCUSDT": candle}, "body_progress_hold")

    assert result.entry_index.tolist() == [22]
    assert result.confirmation_index.tolist() == [21]
    assert result.entry_time.tolist() == [candle["time"][22]]


def test_confirmation_does_not_read_bars_after_entry():
    candle, entries = candle_fixture()
    original = confirmation_entries(
        entries, {"BTCUSDT": candle}, "body_progress_hold")
    changed = {key: value.copy() for key, value in candle.items()}
    for key in ("open", "high", "low", "close", "volume"):
        changed[key][23:] *= 10
    revised = confirmation_entries(
        entries, {"BTCUSDT": changed}, "body_progress_hold")

    pd.testing.assert_frame_equal(original, revised)
