import os
from unittest.mock import patch

import pytest

os.environ["BOT_STARTUP_ENABLED"] = "0"

import main


def setup(max_bars=2):
    return {
        "id": 7,
        "symbol": "BTCUSDT",
        "direction": "LONG",
        "entry_price": 100.0,
        "sl": 98.0,
        "tp1": 101.0,
        "tp2": 104.0,
        "atr": 1.0,
        "strategy_tp1_close_frac": .05,
        "strategy_runner_mode": "fixed_be",
        "strategy_bar_seconds": 900,
        "strategy_entry_bar_ts": 1800.0,
        "strategy_stop_on_close": 0,
        "strategy_max_bars": max_bars,
        "strategy_expire_at": None,
    }


def candles(rows):
    return {
        "time": [1800 + 900 * i for i in range(len(rows))],
        "open": [row[0] for row in rows],
        "high": [row[1] for row in rows],
        "low": [row[2] for row in rows],
        "close": [row[3] for row in rows],
        "volume": [1.0] * len(rows),
        "confirmed": [1] * len(rows),
    }


def test_fixed_be_activates_on_bar_after_tp1():
    frame = candles([
        (100.0, 101.2, 99.5, 100.5),
        (100.4, 100.5, 99.9, 100.1),
    ])
    value = main._resolve_unsent_venue_setup(
        setup(), frame, now=3600)
    assert value[:3] == ("TP1", 1, 0)
    assert value[3] == pytest.approx(.025)


def test_ambiguous_tp1_bar_is_stop_first_before_runner_activation():
    frame = candles([(100.0, 101.2, 97.9, 100.5)])
    value = main._resolve_unsent_venue_setup(
        setup(max_bars=1), frame, now=2700)
    assert value[:3] == ("SL", 0, 0)
    assert value[3] == pytest.approx(-1.0)


def test_direct_tp2_uses_frozen_barbell_fraction():
    frame = candles([(100.0, 104.1, 99.5, 103.5)])
    value = main._resolve_unsent_venue_setup(
        setup(max_bars=1), frame, now=2700)
    assert value[:3] == ("TP2", 1, 1)
    assert value[3] == pytest.approx(1.925)


def test_unresolved_horizon_waits_for_all_closed_bars():
    frame = candles([(100.0, 100.5, 99.5, 100.2)])
    assert main._resolve_unsent_venue_setup(
        setup(max_bars=2), frame, now=2700) is None


def test_tp1_runner_remains_pending_until_next_bar_or_horizon():
    frame = candles([(100.0, 101.2, 99.5, 100.5)])
    assert main._resolve_unsent_venue_setup(
        setup(max_bars=2), frame, now=2700) is None


def test_tracker_uses_persisted_timeframe_and_records_exact_r():
    row = setup(max_bars=1)
    frame = candles([(100.0, 104.1, 99.5, 103.5)])
    with patch.object(main, "get_unresolved_venue_setups",
                      return_value=[row]), \
         patch.object(main.time, "time", return_value=2700), \
         patch.object(main, "get_klines_xperp", return_value=frame) as fetch, \
         patch.object(main, "mark_setup_resolved") as resolved:
        main._track_blocked_venue_setups()
    assert fetch.call_args.kwargs["interval"] == "15m"
    resolved.assert_called_once()
    assert resolved.call_args.args[:4] == (7, "TP2", 1, 1)
    assert resolved.call_args.kwargs["net_r"] == pytest.approx(1.925)
