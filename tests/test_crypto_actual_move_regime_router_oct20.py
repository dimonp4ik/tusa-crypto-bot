import numpy as np

from crypto_actual_move_regime_router_oct20_audit import btc_context


def test_btc_context_uses_only_prior_closed_bars():
    size = 1300
    close = np.linspace(100.0, 200.0, size)
    candle = {
        "time": np.arange(size, dtype=np.int64) * 300,
        "high": close + 1,
        "low": close - 1,
        "close": close,
    }

    original = btc_context(candle)
    changed = dict(candle)
    changed["close"] = close.copy()
    changed["close"][1200] = 10000.0
    changed["high"] = changed["close"] + 1
    changed["low"] = changed["close"] - 1
    revised = btc_context(changed)

    # The current candle is unknown at its own open, so changing it cannot
    # change the context used by an entry at the same timestamp.
    for field in ("btc_trend", "btc_impulse", "btc_impulse_value"):
        assert original.loc[1200, field] == revised.loc[1200, field]
