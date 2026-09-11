import unittest

from entry_rate_cap_lab import metrics, replay


def trade(symbol, direction, entry, exit_, net_r=1.0, outcome="TRAIL"):
    return {
        "symbol": symbol,
        "direction": direction,
        "entry_time": float(entry),
        "exit_time": float(exit_),
        "net_r": float(net_r),
        "outcome": outcome,
    }


class EntryRateCapTests(unittest.TestCase):
    def test_rate_window_uses_only_prior_entry_times(self):
        rows = [
            trade("A", "LONG", 0, 10, -1, "SL"),
            trade("B", "LONG", 100, 200),
            trade("C", "LONG", 3600, 3700),
        ]
        selected, blocked = replay(
            rows,
            same_direction_positions=5,
            loss_streak=0,
            rate_count=1,
            rate_window_hours=1,
        )
        self.assertEqual([row["symbol"] for row in selected], ["A", "C"])
        self.assertEqual(blocked["entry_rate_cap"], 1)

    def test_loss_streak_waits_for_exit(self):
        rows = [
            trade("A", "LONG", 0, 100, -1, "SL"),
            trade("B", "SHORT", 50, 60),
            trade("C", "SHORT", 101, 110),
        ]
        selected, blocked = replay(
            rows,
            same_direction_positions=5,
            loss_streak=1,
        )
        self.assertEqual([row["symbol"] for row in selected], ["A", "B"])
        self.assertEqual(blocked["loss_streak"], 1)

    def test_metrics_orders_drawdown_by_exit(self):
        rows = [
            trade("A", "LONG", 0, 30, -2, "SL"),
            trade("B", "SHORT", 1, 10, 1),
            trade("C", "LONG", 2, 20, 1),
        ]
        result = metrics(rows)
        self.assertEqual(result["net_r"], 0)
        self.assertEqual(result["max_drawdown_r"], 2)


if __name__ == "__main__":
    unittest.main()
