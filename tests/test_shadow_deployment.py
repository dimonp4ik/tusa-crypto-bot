import os
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
        with patch.object(main, "_run_venue_strategy_scan") as scan:
            main.run_scan()
        scan.assert_called_once_with()

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

    def test_blocked_symbols_are_skipped(self):
        """The admin block list has to reach the scan, or it is a placebo."""
        import inspect
        src = inspect.getsource(main._run_venue_strategy_scan)
        self.assertIn("get_active_symbol_blocks", src)
        self.assertIn("symbol in blocked", src)

    def test_one_scan_cannot_publish_an_unbounded_signal_batch(self):
        candidates = [
            ({"symbol": f"S{i}USDT", "direction": "LONG"}, i + 1)
            for i in range(config.VENUE_MAX_SIGNALS_PER_SCAN + 2)
        ]
        with patch.object(main, "send_signal", return_value=True) as send, \
             patch.object(main, "mark_setup_sent") as sent, \
             patch.object(main, "mark_setup_blocked") as blocked, \
             patch.object(main, "_cache_signal"):
            published = main._publish_venue_candidates(candidates)
        self.assertEqual(published, config.VENUE_MAX_SIGNALS_PER_SCAN)
        self.assertEqual(send.call_count, config.VENUE_MAX_SIGNALS_PER_SCAN)
        self.assertEqual(sent.call_count, config.VENUE_MAX_SIGNALS_PER_SCAN)
        self.assertEqual(blocked.call_count, 2)
        self.assertTrue(all(call.args[1] == "scan_cap"
                            for call in blocked.call_args_list))


if __name__ == "__main__":
    unittest.main()
