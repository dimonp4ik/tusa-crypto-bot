import pandas as pd

import crypto_profit_first_regime_direction_oct20_audit as audit


def test_alignment_semantics():
    assert audit.is_aligned("bull_pullback", "LONG")
    assert audit.is_aligned("bear_rally", "SHORT")
    assert not audit.is_aligned("bull", "SHORT")
    assert not audit.is_aligned("bear", "LONG")


def test_trend_scope_keeps_countertrend_range_reversion():
    frame = pd.DataFrame([
        {"btc_regime": "bull", "direction": "SHORT", "family": "breakout",
         "btc_fast_slow_atr": 1.0, "btc_slow_long_atr": 1.0},
        {"btc_regime": "bull", "direction": "SHORT", "family": "range_reversion",
         "btc_fast_slow_atr": 1.0, "btc_slow_long_atr": 1.0},
    ])
    result = audit.apply_guard(frame, "trend_families", .5)
    assert result.family.tolist() == ["range_reversion"]
