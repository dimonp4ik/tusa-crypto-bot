import math
from unittest.mock import patch

import main
from src import crypto_precision_profit_first_shadow as router


CONTEXT = {
    "btc_regime": "bull",
    "btc_fast_slow_atr": 1.0,
    "btc_slow_long_atr": 2.0,
    "btc_eff20": .2,
    "btc_eff96": .3,
    "btc_return20_atr": 1.5,
    "btc_return96_atr": 3.0,
    "btc_z20": .8,
    "btc_atr_pct": .01,
}
SIGNAL = {
    "family": "breakout", "direction": "LONG", "atr": 1.0,
    "eff_ratio": .4, "z": 2.5, "abs_z": 2.5,
}


def test_feature_row_matches_frozen_schema_and_direction_transforms():
    values = router._feature_values(
        "BTCUSDT", SIGNAL, CONTEXT, 1_800.0, 100.0)

    assert list(values) == router._MODEL["features"]
    assert values["risk_pct"] == .02
    assert values["dir_z"] == 2.5
    assert values["btc_trend_dir_atr"] == 1.0
    assert values["btc_return_dir_atr"] == 1.5
    assert values["symbol=BTCUSDT"] == 1.0
    assert values["direction=SHORT"] == 0.0
    assert math.isclose(values["hour_sin"], math.sin(2 * math.pi * .5 / 24))


def test_shadow_setup_has_market_barbell_and_no_position_size():
    candles = {"close": [99.0]}
    with patch.object(router, "btc_context", return_value=CONTEXT), \
         patch.object(router, "family_signals", return_value=[SIGNAL]), \
         patch.object(router, "score_named", return_value=.8):
        rows = router.shadow_setups(
            candles, candles, entry_time=1_800.0,
            market_price=100.0, symbol="BTCUSDT")

    assert len(rows) == 1
    setup = rows[0]
    assert setup["entry_source"] == "MARKET"
    assert setup["entry"] == 100.0
    assert setup["sl"] == 98.0
    assert setup["tp"] == 100.5
    assert setup["runner_target_r"] == 2.0
    assert setup["near_close_fraction"] == .10
    assert setup["hold_bars"] == 48
    assert setup["_precision_profit_first_shadow"] is True
    assert not any("size" in key or "margin" in key for key in setup)


def test_shadow_setup_fails_closed_below_frozen_quality():
    candles = {"close": [99.0]}
    with patch.object(router, "btc_context", return_value=CONTEXT), \
         patch.object(router, "family_signals", return_value=[SIGNAL]), \
         patch.object(router, "score_named",
                      return_value=router.QUALITY_THRESHOLD - 1e-9):
        assert router.shadow_setups(
            candles, candles, entry_time=1_800.0,
            market_price=100.0, symbol="BTCUSDT") == []


def test_analysis_persists_exact_15m_exit_contract_even_in_live_mode():
    setup = {
        **SIGNAL, **CONTEXT,
        "entry": 100.0, "sl": 98.0, "tp": 100.5,
        "target_r": .25, "runner_target_r": 2.0,
        "near_close_fraction": .10, "hold_bars": 48,
        "first_target_move_pct": .5, "runner_move_pct": 4.0,
        "model_quality": .8, "utc_session": "00_05",
        "signal_bar_ts": 900.0,
        "module": "shadow_precision_profit_first_025_2r",
        "_shadow_experiment": True,
        "_precision_profit_first_shadow": True,
    }
    with patch.object(main, "DEPLOYMENT_MODE", "live"):
        analysis = main._venue_analysis("BTCUSDT", setup)

    assert analysis["_shadow_only"] is True
    assert analysis["source"] == "venue_shadow_precision_profit_first"
    assert analysis["fixed_target_r"] == .25
    assert analysis["fixed_runner_target_r"] == 2.0
    assert analysis["strategy_tp1_close_frac"] == .10
    assert analysis["strategy_runner_mode"] == "fixed_be"
    assert analysis["strategy_bar_seconds"] == 900
    assert analysis["strategy_entry_bar_ts"] == 1_800.0
    assert analysis["strategy_stop_on_close"] is False
    assert analysis["strategy_max_bars"] == 48
