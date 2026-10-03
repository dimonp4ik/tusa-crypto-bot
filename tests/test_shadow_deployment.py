import os
import subprocess
import sys
import unittest
from unittest.mock import patch

os.environ["BOT_STARTUP_ENABLED"] = "0"

import config
import main
from src import telegram_notifier as notifier


class ShadowDeploymentTests(unittest.TestCase):
    def test_real_orders_fail_closed(self):
        self.assertEqual(config.DEPLOYMENT_MODE, "shadow")
        self.assertFalse(config.AUTOTRADE_ENABLED)
        # The legacy strategy is not a flag any more - it was deleted on
        # 2026-09-16, so there is nothing left to switch back on.
        self.assertFalse(hasattr(config, "LEGACY_STRATEGY_ENABLED"))
        self.assertEqual(config.REGIME_FILTER_MODE, "paper")

    def test_scan_dispatches_only_frozen_router(self):
        calls = []
        with patch.object(main, "_check_open_signals",
                          side_effect=lambda: calls.append("reconcile")), \
             patch.object(main, "_run_venue_strategy_scan",
                          side_effect=lambda: calls.append("scan")) as scan:
            main.run_scan()
        scan.assert_called_once_with()
        self.assertEqual(calls, ["reconcile", "scan"])

    def test_scan_fails_closed_when_book_cannot_be_reconciled(self):
        with patch.object(main, "_check_open_signals",
                          side_effect=RuntimeError("feed unavailable")), \
             patch.object(main, "_run_venue_strategy_scan") as scan:
            self.assertIsNone(main.run_scan())
        scan.assert_not_called()

    def test_telegram_paper_signal_is_explicit_and_untradeable(self):
        analysis = {
            "symbol": "BTCUSDT", "direction": "LONG", "decision": "LONG",
            "current_price": 100.0, "atr": 1.0, "fixed_stop_atr": 2.0,
            "fixed_target_r": 0.5, "signals": ["Venue module frozen_test"],
            "_shadow_only": True,
        }
        sent = []
        with patch.object(notifier, "_send_message", side_effect=lambda text: sent.append(text) or True),              patch("src.db.log_signal", return_value=42):
            self.assertTrue(notifier.send_signal(analysis))
        self.assertEqual(analysis["_signal_id"], 42)
        self.assertIn("ОРДЕР НЕ ОТКРЫТ", sent[0])
        self.assertNotIn("Плечо", sent[0])

    def test_telegram_barbell_signal_shows_both_market_exit_levels(self):
        analysis = {
            "symbol": "BTCUSDT", "direction": "SHORT", "decision": "SHORT",
            "current_price": 100.0, "atr": 1.0, "fixed_stop_atr": 2.0,
            "fixed_target_r": .25, "fixed_runner_target_r": 4.0,
            "strategy_tp1_close_frac": .70,
            "signals": ["Venue module shadow_5m_oi_trend_pullback"],
            "_shadow_only": True,
        }
        sent = []
        with patch.object(notifier, "_send_message",
                          side_effect=lambda text: sent.append(text) or True), \
             patch("src.db.log_signal", return_value=43):
            self.assertTrue(notifier.send_signal(analysis))
        self.assertIn("TP1, закрыть 70%", sent[0])
        self.assertIn("TP2, остаток 30%", sent[0])
        self.assertIn("`4.00R`", sent[0])

    def test_menu_has_no_autotrading_button(self):
        labels = str(main._USER_KB) + str(main._ADMIN_KB) + str(main._KB_PEOPLE)
        self.assertNotIn("Автотрейдинг", labels)
        self.assertIn("Paper-сделки", labels)

    def test_every_button_has_a_handler(self):
        """A button with no branch does nothing when pressed."""
        import re
        from pathlib import Path
        src = Path("main.py").read_text(encoding="utf-8")
        buttons = sorted(set(re.findall(r'"callback_data":\s*"([^"{]+)"', src)))
        self.assertTrue(buttons)
        for cb in buttons:
            handled = (re.search(r'==\s*"%s"' % re.escape(cb), src)
                       or re.search(r'data in \([^)]*"%s"' % re.escape(cb), src)
                       or any(re.search(r'startswith\(\s*\(?"%s' % re.escape(cb[:k]), src)
                              for k in range(4, len(cb) + 1)))
            self.assertTrue(handled, "no handler for button %s" % cb)

    def test_publish_path_never_opens_real_positions(self):
        """No autotrader call may reappear in the scan without a deliberate change."""
        import inspect
        src = inspect.getsource(main._run_venue_strategy_scan)
        code = chr(10).join(ln.split("#", 1)[0] for ln in src.splitlines())
        self.assertNotIn("open_positions_for_signal", code)

    def test_experimental_venue_analysis_is_always_shadow_only(self):
        setup = {
            "direction": "SHORT", "entry": 100.0, "atr": 2.0,
            "target_r": .25, "utc_session": "18_23",
            "btc_regime": "bull_pullback", "eff_ratio": .2,
            "signal_bar_ts": 123.0,
            "module": "shadow_pullback_short_bull_pullback_evening",
            "_shadow_experiment": True,
        }
        with patch.object(main, "DEPLOYMENT_MODE", "live"):
            analysis = main._venue_analysis("AAVEUSDT", setup)
        self.assertTrue(analysis["_shadow_only"])
        self.assertEqual(analysis["source"], "venue_shadow_experiment")

    def test_live_scan_source_keeps_shadow_experiment_behind_mode_gate(self):
        import inspect
        src = inspect.getsource(main._run_venue_strategy_scan)
        self.assertIn('DEPLOYMENT_MODE == "shadow"', src)
        self.assertIn("shadow_experimental_setups", src)

    def test_five_minute_oi_analysis_is_permanently_shadow_only(self):
        setup = {
            "direction": "SHORT", "entry": 100.0, "atr": 2.0,
            "target_r": .25, "utc_session": "18",
            "btc_regime": "oi_confirmed_trend", "eff_ratio": .2,
            "signal_bar_ts": 123.0, "oi_change_168h": -.05,
            "module": "shadow_5m_oi_trend_pullback",
            "_shadow_experiment": True, "_five_minute_oi_shadow": True,
        }
        with patch.object(main, "DEPLOYMENT_MODE", "live"):
            analysis = main._venue_analysis("AAVEUSDT", setup)
        self.assertTrue(analysis["_shadow_only"])
        self.assertTrue(analysis["_five_minute_oi_shadow"])
        self.assertEqual(analysis["source"], "venue_shadow_5m_oi")
        self.assertEqual(analysis["fixed_runner_target_r"], 4.0)
        self.assertEqual(analysis["strategy_tp1_close_frac"], .70)
        self.assertEqual(analysis["strategy_runner_mode"], "fixed_be")
        self.assertEqual(analysis["strategy_bar_seconds"], 300)
        self.assertEqual(analysis["strategy_entry_bar_ts"], 423.0)
        self.assertFalse(analysis["strategy_stop_on_close"])
        self.assertEqual(analysis["strategy_max_bars"], 72)

    def test_five_minute_fixed_runner_uses_5m_bars_and_next_bar_breakeven(self):
        start = 2_000_000_100.0
        frame = {
            "time": [start, start + 300],
            "open": [99.8, 99.6], "high": [99.9, 100.0],
            "low": [99.4, 99.0], "close": [99.6, 100.0],
            "confirmed": [True, False],
        }
        signal = {
            "id": 71, "symbol": "BTCUSDT", "direction": "SHORT",
            "entry_price": 100.0, "sl": 102.0, "tp1": 99.5, "tp2": 92.0,
            "atr": 1.0, "opened_at": start - 60, "tp1_hit_at": start + 30,
            "status": "TP1_PARTIAL", "strategy_tp1_close_frac": .70,
            "strategy_runner_mode": "fixed_be", "strategy_bar_seconds": 300,
            "strategy_entry_bar_ts": start,
            "runner_activation_bar_ts": start,
            "strategy_stop_on_close": 0, "strategy_max_bars": 72,
        }
        with patch.object(main, "get_open_signals", return_value=[signal]), \
             patch.object(main.time, "time", return_value=start + 360), \
             patch.object(main, "get_klines_xperp", return_value=frame) as candles, \
             patch.object(main, "update_signal_status") as update, \
             patch.object(main, "send_signal_update"), \
             patch.object(main.autotrader, "mirror_transition"):
            main._check_open_signals()
        self.assertEqual(candles.call_args.kwargs["interval"], "5m")
        self.assertEqual(update.call_args.args[1], "BREAKEVEN")
        self.assertAlmostEqual(update.call_args.kwargs["realized_r"], .175)

    def test_five_minute_stop_is_wick_touch_even_when_forming_close_recovers(self):
        start = 2_000_001_000.0
        frame = {
            "time": [start], "open": [100.0], "high": [100.5],
            "low": [97.8], "close": [100.0], "confirmed": [False],
        }
        signal = {
            "id": 72, "symbol": "BTCUSDT", "direction": "LONG",
            "entry_price": 100.0, "sl": 98.0, "tp1": 101.0, "tp2": 108.0,
            "atr": 1.0, "opened_at": start, "status": "OPEN",
            "strategy_tp1_close_frac": .70, "strategy_runner_mode": "fixed_be",
            "strategy_bar_seconds": 300, "strategy_entry_bar_ts": start,
            "strategy_stop_on_close": 0, "strategy_max_bars": 72,
        }
        with patch.object(main, "get_open_signals", return_value=[signal]), \
             patch.object(main.time, "time", return_value=start + 60), \
             patch.object(main, "get_klines_xperp", return_value=frame), \
             patch.object(main, "_deep_feed_breached", return_value=True), \
             patch.object(main, "set_sl_xperp_only"), \
             patch.object(main, "update_signal_status") as update, \
             patch.object(main, "send_signal_update"), \
             patch.object(main.autotrader, "mirror_transition"):
            main._check_open_signals()
        self.assertEqual(update.call_args.args[1], "SL_HIT")
        self.assertEqual(update.call_args.kwargs["realized_r"], -1.0)

    def test_crypto_five_minute_target_after_72_bars_is_ignored(self):
        start = 2_000_010_000.0
        times = [start + 300 * i for i in range(73)]
        frame = {
            "time": times, "open": [100.0] * 73, "high": [100.4] * 72 + [109.0],
            "low": [99.0] * 73, "close": [100.0] * 73,
            "confirmed": [True] * 72 + [False],
        }
        signal = {
            "id": 73, "symbol": "BTCUSDT", "direction": "LONG",
            "entry_price": 100.0, "sl": 98.0, "tp1": 101.0, "tp2": 108.0,
            "atr": 1.0, "opened_at": start, "status": "OPEN",
            "strategy_tp1_close_frac": .70, "strategy_runner_mode": "fixed_be",
            "strategy_bar_seconds": 300, "strategy_entry_bar_ts": start,
            "strategy_stop_on_close": 0, "strategy_max_bars": 72,
        }
        with patch.object(main, "get_open_signals", return_value=[signal]), \
             patch.object(main.time, "time", return_value=start + 72 * 300 + 60), \
             patch.object(main, "get_klines_xperp", return_value=frame), \
             patch.object(main, "update_signal_status") as update, \
             patch.object(main, "send_signal_update"), \
             patch.object(main.autotrader, "mirror_transition"):
            main._check_open_signals()
        self.assertEqual(update.call_args.args[1], "EXPIRED")

    def test_five_minute_oi_shadow_obeys_two_per_hour_cap(self):
        candidates = [({
            "symbol": "XUSDT", "direction": "SHORT",
            "_five_minute_oi_shadow": True,
        }, 7)]
        with patch.object(main, "get_recent_signals", return_value=[
                {"opened_at": main.time.time() - 60},
                {"opened_at": main.time.time() - 120},
             ]), patch.object(main, "send_signal") as send, \
             patch.object(main, "mark_setup_blocked") as blocked:
            self.assertEqual(main._publish_venue_candidates(candidates), 0)
        send.assert_not_called()
        blocked.assert_called_once_with(7, "hour_cap")

    def test_hour_cap_applies_to_whole_audited_portfolio(self):
        candidates = [({
            "symbol": "XUSDT", "direction": "LONG", "source": "venue_regime",
        }, 8)]
        with patch.object(main, "get_recent_signals", return_value=[
                {"opened_at": main.time.time() - 60},
                {"opened_at": main.time.time() - 120},
             ]), patch.object(main, "send_signal") as send, \
             patch.object(main, "mark_setup_blocked") as blocked:
            self.assertEqual(main._publish_venue_candidates(candidates), 0)
        send.assert_not_called()
        blocked.assert_called_once_with(8, "hour_cap")

    def test_production_has_priority_over_five_minute_supplement(self):
        candidates = [
            ({"symbol": "AUSDT", "direction": "SHORT",
              "source": "venue_shadow_5m_oi", "_five_minute_oi_shadow": True}, 9),
            ({"symbol": "ZUSDT", "direction": "LONG",
              "source": "venue_regime"}, 10),
        ]
        with patch.object(main, "get_recent_signals", return_value=[
                {"opened_at": main.time.time() - 60},
             ]), patch.object(main, "send_signal", return_value=True) as send, \
             patch.object(main, "mark_setup_sent"), \
             patch.object(main, "mark_setup_blocked") as blocked, \
             patch.object(main, "_cache_signal"):
            self.assertEqual(main._publish_venue_candidates(candidates), 1)
        self.assertEqual(send.call_args.args[0]["source"], "venue_regime")
        blocked.assert_called_once_with(9, "hour_cap")

    def test_scan_preserves_five_minute_bar_time_and_lazily_fetches_oi(self):
        fifteen = {"time": [900.0], "close": [100.0]}
        five = {"time": [300.0], "close": [99.0]}
        setup = {
            "direction": "SHORT", "entry": 99.0, "atr": 2.0,
            "target_r": .25, "utc_session": "00",
            "btc_regime": "oi_confirmed_trend", "eff_ratio": .2,
            "signal_bar_ts": 300.0, "oi_change_168h": -.05,
            "module": "shadow_5m_oi_trend_pullback",
            "_shadow_experiment": True, "_five_minute_oi_shadow": True,
        }

        def klines(_symbol, **kwargs):
            return five if kwargs.get("interval") == "5m" else fifteen

        with patch.object(main, "ROBUST_SYMBOLS", {"BTCUSDT"}), \
             patch.object(main, "DEPLOYMENT_MODE", "shadow"), \
             patch.object(main, "REGIME_FILTER_MODE", "paper"), \
             patch.object(main, "TRADE_WEEKENDS", True), \
             patch.object(main, "TRADING_HOURS_START", 0), \
             patch.object(main, "TRADING_HOURS_END", 24), \
             patch.object(main, "_venue_daily_loss_pause", return_value=False), \
             patch.object(main, "get_xperp_instruments",
                          return_value={"BTC": "BTC-XPERP"}), \
             patch.object(main, "get_klines_xperp", side_effect=klines), \
             patch.object(main, "get_xperp_price", return_value=99.0), \
             patch.object(main, "get_open_signals", return_value=[]), \
             patch.object(main, "get_active_symbol_blocks", return_value=[]), \
             patch.object(main, "venue_setups", return_value=[]), \
             patch.object(main, "shadow_experimental_setups", return_value=[]), \
             patch.object(main, "five_minute_oi_candidate_context",
                          return_value={"direction": "SHORT"}), \
             patch.object(main, "get_xperp_open_interest_history",
                          return_value={"time": [1], "oi": [1]}) as oi, \
             patch.object(main, "five_minute_oi_shadow_setup",
                          return_value=setup), \
             patch.object(main, "log_setup_candidate_once", return_value=11), \
             patch.object(main, "_venue_signal_on_cooldown", return_value=False), \
             patch.object(main, "_publish_venue_candidates", return_value=1) as publish:
            main._run_venue_strategy_scan()

        oi.assert_called_once_with("BTCUSDT", limit=170)
        analysis = publish.call_args.args[0][0][0]
        self.assertEqual(analysis["signal_bar_ts"], 300.0)
        self.assertEqual(analysis["source"], "venue_shadow_5m_oi")

    def test_scan_does_not_fetch_oi_without_price_candidate(self):
        with patch.object(main, "DEPLOYMENT_MODE", "shadow"), \
             patch.object(main, "five_minute_oi_candidate_context",
                          return_value=None), \
             patch.object(main, "get_xperp_open_interest_history") as oi:
            result = main._five_minute_oi_setup(
                "BTCUSDT", {"time": [300]}, {"time": [300]}, 99.0)
        self.assertIsNone(result)
        oi.assert_not_called()

    def test_blocked_symbols_are_skipped(self):
        """The admin block list has to reach the scan, or it is a placebo."""
        import inspect
        src = inspect.getsource(main._run_venue_strategy_scan)
        self.assertIn("get_active_symbol_blocks", src)
        self.assertIn("symbol in blocked", src)

    def test_one_scan_obeys_tighter_hourly_portfolio_cap(self):
        self.assertLessEqual(config.MAX_SAME_DIRECTION_POSITIONS, 3)
        candidates = [
            ({"symbol": f"S{i}USDT", "direction": "LONG"}, i + 1)
            for i in range(config.VENUE_MAX_SIGNALS_PER_SCAN + 2)
        ]
        with patch.object(main, "send_signal", return_value=True) as send, \
             patch.object(main, "get_recent_signals", return_value=[]), \
             patch.object(main, "mark_setup_sent") as sent, \
             patch.object(main, "mark_setup_blocked") as blocked, \
             patch.object(main, "_cache_signal"):
            published = main._publish_venue_candidates(candidates)
        self.assertEqual(published, 2)
        self.assertEqual(send.call_count, 2)
        self.assertEqual(sent.call_count, 2)
        self.assertEqual(blocked.call_count,
                         config.VENUE_MAX_SIGNALS_PER_SCAN)
        self.assertTrue(all(call.args[1] == "hour_cap"
                            for call in blocked.call_args_list))

    def test_stale_environment_cannot_raise_direction_cap(self):
        env = {**os.environ, "MAX_SAME_DIRECTION_POSITIONS": "99"}
        value = subprocess.check_output(
            [sys.executable, "-c",
             "import config; print(config.MAX_SAME_DIRECTION_POSITIONS)"],
            env=env, text=True,
        ).strip()
        self.assertEqual(value, "3")

    def test_venue_cooldown_is_per_symbol_and_direction(self):
        now = 2_000_000.0
        with patch.dict(main._signal_cache, {
                "BTCUSDT|LONG": ("LONG", now - 60)}, clear=True):
            self.assertTrue(main._venue_signal_on_cooldown("BTCUSDT", "LONG", now))
            self.assertFalse(main._venue_signal_on_cooldown("BTCUSDT", "SHORT", now))

    def test_existing_direction_exposure_blocks_whole_new_batch(self):
        candidates = [({"symbol": "XUSDT", "direction": "LONG"}, 1)]
        with patch.object(main, "send_signal", return_value=True) as send, \
             patch.object(main, "get_recent_signals", return_value=[]), \
             patch.object(main, "mark_setup_blocked") as blocked:
            published = main._publish_venue_candidates(
                candidates, {"LONG": config.MAX_SAME_DIRECTION_POSITIONS})
        self.assertEqual(published, 0)
        send.assert_not_called()
        blocked.assert_called_once_with(1, "dir_cap")

    def test_daily_loss_pause_uses_only_closed_signal_streak(self):
        with patch.object(main, "get_today_sl_streak",
                          return_value=config.VENUE_LOSS_PAUSE_STREAK):
            self.assertTrue(main._venue_daily_loss_pause())


if __name__ == "__main__":
    unittest.main()
