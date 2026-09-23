import unittest

from crypto_forward_monitor import portfolio_gate_with_pending


class CryptoForwardMonitorParityTests(unittest.TestCase):
    @staticmethod
    def _pending(symbol, entry_time, direction="LONG"):
        return {"symbol": symbol, "direction": direction,
                "entry_time": entry_time}

    @staticmethod
    def _completed(symbol, entry_time, exit_time, direction="LONG"):
        return {"symbol": symbol, "direction": direction,
                "entry_time": entry_time, "exit_time": exit_time,
                "outcome": "TP2", "net_r": 0.5}

    def test_pending_positions_consume_direction_capacity(self):
        pending = [self._pending(symbol, 900) for symbol in ("A", "B", "C")]
        completed = [self._completed("D", 1800, 2700)]
        accepted_completed, accepted_pending = portfolio_gate_with_pending(
            completed, pending, direction_cap=3)
        self.assertEqual(accepted_completed, [])
        self.assertEqual([row["symbol"] for row in accepted_pending],
                         ["A", "B", "C"])

    def test_pending_position_blocks_same_symbol_retry(self):
        pending = [self._pending("A", 900), self._pending("A", 1800)]
        _, accepted_pending = portfolio_gate_with_pending(
            [], pending, direction_cap=3)
        self.assertEqual([row["entry_time"] for row in accepted_pending], [900])


if __name__ == "__main__":
    unittest.main()
