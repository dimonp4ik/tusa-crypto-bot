import numpy as np

from crypto_5m_vwap_regime_oct21_audit import (
    add_economic_fields,
    entries_for_rule,
    feature_frame,
    rule_specs,
)


def candle_fixture(size=400):
    close = 100 + np.sin(np.arange(size) / 17)
    return {
        "open": close - .05,
        "high": close + .2,
        "low": close - .2,
        "close": close,
        "volume": 10 + np.arange(size) % 7,
        "time": np.arange(size, dtype=np.int64) * 300,
    }


def test_vwap_features_do_not_read_future_bars():
    candle = candle_fixture()
    original = feature_frame("BTCUSDT", candle)
    changed = {key: value.copy() for key, value in candle.items()}
    for key in ("open", "high", "low", "close", "volume"):
        changed[key][350:] *= 5
    revised = feature_frame("BTCUSDT", changed)

    columns = ["atr", "vwap_96", "vwap_288", "efficiency_96",
               "volume_ratio"]
    np.testing.assert_allclose(
        original.loc[:349, columns].to_numpy(float),
        revised.loc[:349, columns].to_numpy(float), equal_nan=True)


def test_rolling_vwap_includes_only_current_and_prior_window():
    candle = candle_fixture()
    frame = feature_frame("BTCUSDT", candle)
    index = 320
    typical = ((candle["high"] + candle["low"] + candle["close"]) / 3)[
        index - 95:index + 1]
    volume = candle["volume"][index - 95:index + 1]
    expected = float(np.sum(typical * volume) / np.sum(volume))

    assert np.isclose(frame.loc[index, "vwap_96"], expected)


def test_flat_history_produces_no_invalid_entries():
    size = 400
    candle = {
        "open": np.full(size, 100.0),
        "high": np.full(size, 100.0),
        "low": np.full(size, 100.0),
        "close": np.full(size, 100.0),
        "volume": np.ones(size),
        "time": np.arange(size, dtype=np.int64) * 300,
    }
    features = {"BTCUSDT": feature_frame("BTCUSDT", candle)}

    assert all(entries_for_rule(features, rule).empty for rule in rule_specs())


def test_economic_gate_matches_three_daily_requirement():
    result = add_economic_fields({
        "mean_signed_move_pct": 0.9731372335085217,
        "trades_per_calendar_day": 3.0,
    }, .03)

    assert np.isclose(result["projected_30d_return_pct"], 30.0, atol=1e-5)
    assert result["economic_30pct_gate_pass"]
