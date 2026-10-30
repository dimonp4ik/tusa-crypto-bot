from pathlib import Path
import tempfile
from unittest.mock import patch

import pytest

import main
from src import db
from src.tp1_volume_bank import candidate_r, completed_observation


def candles(hit_volume=120.0, confirmed=True):
    return {
        "time": [900 * index for index in range(21)],
        "open": [100.0] * 21,
        "high": [101.0] * 21,
        "low": [99.0] * 21,
        "close": [100.0] * 20 + [100.4],
        "volume": [100.0] * 20 + [hit_volume],
        "confirmed": [True] * 20 + [confirmed],
    }


def test_observation_waits_for_close_and_uses_only_prior_volume():
    forming = completed_observation(
        candles(confirmed=False), activation_ts=20 * 900,
        bar_seconds=900, now=20 * 900 + 100)
    assert forming is None

    weak = completed_observation(
        candles(), activation_ts=20 * 900,
        bar_seconds=900, now=21 * 900)
    assert weak["volume_ratio"] == pytest.approx(1.2)
    assert weak["close"] == 100.4
    assert weak["extra_close_fraction"] == .75

    strong = completed_observation(
        candles(hit_volume=200.0), activation_ts=20 * 900,
        bar_seconds=900, now=21 * 900)
    assert strong["volume_ratio"] == 2.0
    assert strong["extra_close_fraction"] == 0.0


def test_three_leg_candidate_r_keeps_twenty_percent_runner():
    assert candidate_r(
        tp1_fraction=.05, tp1_r=.5,
        extra_fraction=.75, extra_r=.4, runner_r=2.0,
    ) == pytest.approx(.725)


def test_parallel_r_does_not_replace_baseline_accounting():
    signal = {
        "direction": "LONG",
        "strategy_tp1_volume_threshold": 1.5,
        "strategy_tp1_close_frac": .05,
        "tp1_extra_close_frac": .75,
        "tp1_extra_exit_price": 100.4,
    }
    value = main._tp1_volume_parallel_r(
        signal, realized_r=1.925, reached_tp1=True,
        entry=100.0, sl=99.0, tp1=100.5)
    assert value == pytest.approx(.725)
    assert main._tp1_volume_parallel_r(
        signal, realized_r=-1.0, reached_tp1=False,
        entry=100.0, sl=99.0, tp1=100.5) == -1.0


def test_database_persists_baseline_and_challenger_separately():
    with tempfile.TemporaryDirectory() as tmp:
        path = str(Path(tmp) / "signals.db")
        with patch.object(db, "DB_PATH", path):
            db.init_db()
            signal_id = db.log_signal({
                "symbol": "TEST", "direction": "LONG",
                "current_price": 100.0, "_shadow_only": True,
                "strategy_tp1_close_frac": .05,
                "strategy_runner_mode": "fixed_be",
                "fixed_runner_target_r": 2.0,
                "strategy_bar_seconds": 900,
                "strategy_tp1_volume_threshold": 1.5,
                "strategy_tp1_extra_close_frac": .75,
            }, 100.5, 102.0, 99.0)
            assert db.record_tp1_volume_observation(
                signal_id, volume_ratio=1.2, extra_fraction=.75,
                exit_price=100.4, observed_at=18000.0)
            assert not db.record_tp1_volume_observation(
                signal_id, volume_ratio=2.0, extra_fraction=0.0,
                exit_price=100.8, observed_at=18900.0)
            db.update_signal_status(
                signal_id, "TP2_HIT", 102.0, realized_r=1.925,
                tp1_volume_candidate_r=.725)
            row = db.get_signal_by_id(signal_id)
            assert row["strategy_tp1_volume_threshold"] == 1.5
            assert row["strategy_tp1_extra_close_frac"] == .75
            assert row["tp1_hit_volume_ratio"] == 1.2
            assert row["tp1_extra_close_frac"] == .75
            assert row["tp1_extra_exit_price"] == 100.4
            assert row["realized_r"] == 1.925
            assert row["tp1_volume_candidate_r"] == .725
            stats = db.get_tp1_volume_forward_stats(
                days=30, margin_pct=3.0, leverage=10.0)
            assert stats["n"] == 1
            assert stats["weak_volume_banks"] == 1
            assert stats["baseline_return_pct"] == pytest.approx(.578)
            assert stats["candidate_return_pct"] == pytest.approx(.218)


def test_monitor_records_closed_tp1_bar_then_keeps_baseline_and_candidate():
    start = 2_100_000_000.0
    activation = start + 20 * 900
    frame = {
        "time": [start + index * 900 for index in range(22)],
        "open": [100.0] * 22,
        "high": [100.2] * 20 + [101.0, 102.1],
        "low": [99.5] * 20 + [100.1, 100.2],
        "close": [100.0] * 20 + [100.4, 102.0],
        "volume": [100.0] * 20 + [120.0, 150.0],
        "confirmed": [True] * 22,
    }
    signal = {
        "id": 91, "symbol": "BTCUSDT", "direction": "LONG",
        "entry_price": 100.0, "sl": 99.0, "tp1": 100.5, "tp2": 102.0,
        "atr": .5, "opened_at": start, "status": "TP1_PARTIAL",
        "tp1_hit_at": activation + 100,
        "runner_activation_bar_ts": activation,
        "strategy_tp1_close_frac": .05,
        "strategy_runner_mode": "fixed_be",
        "strategy_runner_target_r": 2.0,
        "strategy_bar_seconds": 900,
        "strategy_entry_bar_ts": start,
        "strategy_stop_on_close": 0,
        "strategy_max_bars": 48,
        "strategy_tp1_volume_threshold": 1.5,
        "strategy_tp1_extra_close_frac": .75,
        "tp1_hit_volume_ratio": None,
    }
    with patch.object(main, "get_open_signals", return_value=[signal]), \
         patch.object(main.time, "time", return_value=start + 22 * 900), \
         patch.object(main, "get_klines_xperp", return_value=frame), \
         patch.object(main, "record_tp1_volume_observation", return_value=True) as record, \
         patch.object(main, "update_signal_status") as update, \
         patch.object(main, "send_signal_update"), \
         patch.object(main.autotrader, "mirror_transition"):
        main._check_open_signals()

    record.assert_called_once_with(
        91, volume_ratio=pytest.approx(1.2), extra_fraction=.75,
        exit_price=100.4, observed_at=activation + 900)
    assert update.call_args.args[1:3] == ("TP2_HIT", 102.0)
    assert update.call_args.kwargs["realized_r"] == pytest.approx(1.925)
    assert update.call_args.kwargs["tp1_volume_candidate_r"] == pytest.approx(.725)
