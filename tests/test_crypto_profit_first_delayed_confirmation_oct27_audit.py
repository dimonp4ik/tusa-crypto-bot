import pandas as pd

import crypto_profit_first_delayed_confirmation_oct27_audit as audit


def test_confirmation_values_use_closed_bar_and_original_structure():
    candle = {
        "open": [99.0, 100.0, 102.0],
        "high": [101.0, 103.0, 103.0],
        "low": [98.0, 99.0, 101.0],
        "close": [100.0, 102.0, 102.5],
    }
    row = pd.Series({"direction": "LONG", "entry": 100.0, "sl": 98.0})
    values = audit._confirmation_values(row, candle, 1)
    assert values["confirmation_body_r"] == 1.0
    assert values["confirmation_progress_r"] == 1.0
    assert values["confirmation_held_stop"] is True
    assert values["confirmation_broke_signal_extreme"] is True
    assert values["confirmation_sweep_reclaim"] is True


def test_delayed_resolver_uses_stop_first_on_ambiguous_bar():
    candle = {
        "time": [0, 900],
        "open": [100.0, 100.0],
        "high": [102.0, 100.0],
        "low": [97.0, 100.0],
        "close": [101.0, 100.0],
    }
    result = audit._resolve(
        candle, 0, "LONG", entry=100.0, stop=98.0, target_r=.5)
    assert result is not None
    assert result.outcome == "SL"
    assert result.price == 98.0


def test_cluster_size_counts_distinct_same_direction_symbols():
    frame = pd.DataFrame([
        {"period": "x", "entry_time": 1, "direction": "LONG",
         "symbol": "BTCUSDT"},
        {"period": "x", "entry_time": 1, "direction": "LONG",
         "symbol": "ETHUSDT"},
        {"period": "x", "entry_time": 1, "direction": "LONG",
         "symbol": "ETHUSDT"},
        {"period": "x", "entry_time": 1, "direction": "SHORT",
         "symbol": "XRPUSDT"},
    ])
    result = audit.add_cluster_size(frame)
    assert result.loc[result.direction.eq("LONG"), "cluster_symbols"].eq(2).all()
    assert result.loc[result.direction.eq("SHORT"), "cluster_symbols"].eq(1).all()


def test_variant_delays_only_requested_family_and_reapplies_risk_gate():
    common = {
        "period": "x", "entry_time": 100, "direction": "LONG",
        "cluster_symbols": 1, "confirmation_valid": True,
        "confirmation_held_stop": True, "confirmation_body_r": .2,
        "confirmation_progress_r": .2, "confirmation_close_location": .8,
        "confirmation_broke_signal_extreme": False,
        "confirmation_sweep_reclaim": False,
        "entry": 100.0, "risk_pct": .015, "near_outcome": "SL",
        "near_exit_time": 120, "outcome": "SL", "exit_time": 120,
        "delayed_entry_time": 115, "delayed_entry": 101.0,
        "delayed_risk_pct": .015, "delayed_near_outcome": "TP2",
        "delayed_near_exit_time": 130, "delayed_far_outcome": "TP2",
        "delayed_far_exit_time": 145,
    }
    frame = pd.DataFrame([
        {**common, "family": "trend_pullback", "symbol": "BTCUSDT"},
        {**common, "family": "breakout", "symbol": "ETHUSDT"},
    ])
    result = audit.apply_variant(frame, "trend_pullback", "progress_hold")
    delayed = result[result.symbol.eq("BTCUSDT")].iloc[0]
    unchanged = result[result.symbol.eq("ETHUSDT")].iloc[0]
    assert delayed.confirmation_delayed
    assert delayed.entry_time == 115
    assert delayed.entry == 101.0
    assert delayed.outcome == "TP2"
    assert not unchanged.confirmation_delayed
    assert unchanged.entry_time == 100
    assert unchanged.outcome == "SL"
