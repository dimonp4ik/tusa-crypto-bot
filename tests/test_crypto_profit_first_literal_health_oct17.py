import unittest

import crypto_profit_first_literal_health_oct17_audit as audit


def row(entry, exit_, outcome, *, symbol="BTC", direction="LONG"):
    return {"entry_time": entry, "exit_time": exit_,
            "signed_price_return": outcome, "symbol": symbol,
            "direction": direction, "source_priority": 0, "year": 2024}


class LiteralHealthGateAuditTest(unittest.TestCase):
    def test_outcome_is_not_visible_until_after_exit(self):
        rows = [row(0, 100, -.01),
                row(50, 60, .02, direction="SHORT"),
                row(101, 120, .02)]
        accepted = audit.causal_gate(
            rows, key_mode="direction", window=20, min_history=1,
            min_mean=0, min_hit_rate=.30)
        self.assertEqual([item["entry_time"] for item in accepted], [0, 50])

    def test_disabled_signal_still_updates_shadow_history(self):
        rows = [row(0, 1, -.01), row(2, 3, .02), row(4, 5, .02)]
        accepted = audit.causal_gate(
            rows, key_mode="direction", window=2, min_history=1,
            min_mean=0, min_hit_rate=.30)
        self.assertEqual([item["entry_time"] for item in accepted], [0, 4])

    def test_key_modes(self):
        value = row(0, 1, .02, symbol="ETH", direction="SHORT")
        self.assertEqual(audit.health_key(value, "direction"), ("SHORT",))
        self.assertEqual(audit.health_key(value, "symbol"), ("ETH",))
        self.assertEqual(audit.health_key(value, "symbol_direction"),
                         ("ETH", "SHORT"))


if __name__ == "__main__":
    unittest.main()
