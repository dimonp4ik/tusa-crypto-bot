import numpy as np

import crypto_multiday_material_trend_oct21_audit as audit


def candles(size=5000):
    times = np.arange(size, dtype=np.int64) * 300
    close = 100 * np.exp(np.arange(size, dtype=float) * .00002)
    volume = 100 + np.arange(size, dtype=float) % 17
    return {
        "time": times, "open": close.copy(), "high": close * 1.001,
        "low": close * .999, "close": close, "volume": volume,
    }


def test_features_enter_after_closed_hourly_signal():
    source = candles()
    frame = audit.symbol_features("BTCUSDT", source)

    assert not frame.empty
    assert (frame.entry_time % 3600 == 0).all()
    row = frame.iloc[0]
    assert row.entry_index > 0
    assert source["time"][int(row.entry_index)] == row.entry_time


def test_future_prices_do_not_change_prior_feature():
    source = candles()
    original = audit.symbol_features("BTCUSDT", source)
    cutoff = int(source["time"][4500])
    changed = {key: value.copy() for key, value in source.items()}
    for key in ("open", "high", "low", "close", "volume"):
        changed[key][4500:] *= 5
    revised = audit.symbol_features("BTCUSDT", changed)
    columns = [column for column in original.columns if column != "month"]
    left = original[original.entry_time < cutoff][columns].reset_index(drop=True)
    right = revised[revised.entry_time < cutoff][columns].reset_index(drop=True)

    assert left.equals(right)


def test_three_daily_economic_threshold_is_exact():
    assert round(audit.required_move(3) or 0, 6) == .973137


def test_spec_library_has_distinct_market_scenarios():
    rules = audit.specs()

    assert len(rules) == 24
    assert {rule["family"] for rule in rules} == {
        "trend_persistence", "channel_expansion", "pullback_reclaim"}
