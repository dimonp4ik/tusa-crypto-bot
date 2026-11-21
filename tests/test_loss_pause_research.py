from datetime import datetime, timezone
import threading
from unittest.mock import patch

import main
from src import db


def test_paused_observation_runs_outside_the_position_monitor_lock():
    monitor_lock = threading.Lock()

    def observe(**kwargs):
        assert kwargs == {"early": True, "observation_only": True}
        assert monitor_lock.acquire(blocking=False)
        monitor_lock.release()

    with (patch.object(main, "_portfolio_state_lock", monitor_lock),
          patch.object(main, "_check_open_signals"),
          patch.object(main, "_venue_daily_loss_pause", return_value=True),
          patch.object(main, "_run_venue_strategy_scan", side_effect=observe) as scan):
        main.run_scan(early=True)
    scan.assert_called_once()


def test_paused_scan_persists_observations_without_publishing_or_claiming_the_live_bar(monkeypatch):
    now = datetime(2026, 11, 21, 22, 0, 12, tzinfo=timezone.utc)
    bucket = int(now.timestamp()) - 12
    candles = dict(time=[bucket - 900, bucket], open=[100., 101.], close=[101., 101.2],
                   high=[101., 101.3], low=[99., 100.9], volume=[30., 1.], confirmed=[True, False])
    monkeypatch.setattr(main, "TRADE_WEEKENDS", True)
    monkeypatch.setattr(main, "TRADING_HOURS_START", 0)
    monkeypatch.setattr(main, "TRADING_HOURS_END", 24)
    monkeypatch.setattr(main, "REGIME_FILTER_MODE", "on")
    monkeypatch.setattr(main, "get_open_signals", lambda: [])
    monkeypatch.setattr(main, "get_active_symbol_blocks", lambda: [])
    with (patch.object(main, "datetime", wraps=datetime) as dt,
          patch.object(main.time, "time", return_value=now.timestamp()),
          patch.object(main, "_venue_daily_loss_pause", return_value=True) as pause,
          patch.object(main, "get_bot_state", return_value=None),
          patch.object(main, "get_xperp_instruments", return_value={"BTC": "BTC-X"}),
          patch.object(main, "_venue_scan_universe", return_value=["BTCUSDT"]),
          patch.object(main, "get_klines_xperp", return_value=candles),
          patch.object(main, "_venue_signal_on_cooldown", return_value=False),
          patch.object(main, "profit_first_material_shadow_setups",
                       return_value=[dict(direction="LONG")]),
          patch.object(main, "prequalified_2r_shadow_setups", return_value=[]) as second_route,
          patch.object(main, "_venue_analysis",
                       side_effect=lambda *_: {"symbol": "BTCUSDT", "direction": "LONG", "source": "normal"}),
          patch.object(main, "log_setup_candidate_once", return_value=1) as claim,
          patch.object(main, "_publish_venue_candidates", return_value=0) as publish,
          patch.object(main, "_record_candidate_decision_books") as books,
          patch.object(main, "_persist_crypto_scan_audit") as audit,
          patch.object(main, "set_bot_state") as state):
        dt.now.return_value = now
        main._run_venue_strategy_scan(early=True)
        publish.assert_not_called()
        second_route.assert_not_called()
        state.assert_not_called()
        assert claim.call_args.kwargs == {"block_reason": "daily_loss_pause"}
        assert claim.call_args.args[0]["source"] == db.LOSS_PAUSE_RESEARCH_SOURCE
        assert claim.call_args.args[0]["_shadow_only"] is True
        assert books.call_args.args[0][0][1] == 1
        saved = audit.call_args.args[0]
        assert saved["result"] == "daily_loss_pause"
        assert saved["published"] == saved["publish_candidates"] == 0
        assert saved["pause_observed"] == saved["strategy"]["symbols_evaluated"] == 1
        pause.return_value = False
        main._run_venue_strategy_scan(early=True)
        publish.assert_called_once()
        assert claim.call_args.args[0]["source"] == "normal"
        state.assert_called_once_with("crypto_completed_early_bar", str(bucket))


def test_pause_research_cannot_claim_the_normal_setup_and_unpriced_rows_stay_unpriced(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_PATH", str(tmp_path / "pause.sqlite"))
    db.init_db()
    setup = dict(symbol="BTCUSDT", direction="LONG", current_price=100., signal_bar_ts=900)
    shadow = dict(setup, source=db.LOSS_PAUSE_RESEARCH_SOURCE)
    observed = db.log_setup_candidate_once(shadow, block_reason="daily_loss_pause")
    assert observed
    assert db.log_setup_candidate_once(shadow, block_reason="daily_loss_pause") is None
    assert db.log_setup_candidate_once(dict(setup, source="venue_shadow_profit_first_2r")) != observed
    db.mark_setup_resolved(observed, "UNVERIFIED_INTRABAR", 0, 0)
    assert db.loss_pause_research_summary(0) == {
        "observed": 1, "with_book": 0, "priced": 0, "unpriced": 1, "total_r": 0.0,
    }


def test_loss_pause_outcomes_require_an_unambiguous_first_bar_and_bracket_order():
    setup = dict(source=db.LOSS_PAUSE_RESEARCH_SOURCE, direction="LONG", entry_price=100.,
                 sl=99., tp1=100.5, tp2=102., strategy_bar_seconds=900, strategy_entry_bar_ts=900,
                 strategy_max_bars=3, strategy_runner_mode="fixed_be")
    base = dict(time=[900, 1800], open=[100., 100.], close=[100., 101.],
                high=[100.2, 102.1], low=[99.5, 99.3], volume=[1., 1.])
    assert main._resolve_unsent_venue_setup(setup, base, now=2800)[0] == "TP2"
    assert main._resolve_unsent_venue_setup(setup, dict(base, high=[100.6, 102.1]), now=2800) == (
        "UNVERIFIED_INTRABAR", 0, 0, None)
    assert main._resolve_unsent_venue_setup(setup, dict(base, low=[99.5, 98.9]), now=2800) == (
        "UNVERIFIED_INTRABAR", 0, 0, None)
    assert main._resolve_unsent_venue_setup(setup, dict(base, time=[900, 2700]), now=3800) == (
        "UNVERIFIED_GAP", 0, 0, None)
