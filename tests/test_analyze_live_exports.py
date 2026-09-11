import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import analyze_live_exports


class AnalyzeLiveExportsTests(unittest.TestCase):
    def test_cost_model_gate_join_and_drawdown(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            signals = root / "signals.csv"
            setups = root / "setups.csv"
            output = root / "result.json"

            signal_fields = [
                "id", "symbol", "direction", "entry_price", "sl", "opened_at",
                "closed_at", "status", "realized_r", "size_mult", "confidence",
                "session", "entry_source", "trend_1h", "trend_4h", "exit_price",
            ]
            with signals.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=signal_fields)
                writer.writeheader()
                writer.writerow({
                    "id": 1, "symbol": "A", "direction": "LONG", "entry_price": 100,
                    "sl": 99, "opened_at": 1, "closed_at": 2, "status": "TP1_TRAIL",
                    "realized_r": 1, "confidence": "HIGH", "session": "OPEN",
                    "entry_source": "FVG", "trend_1h": "bullish", "trend_4h": "bullish",
                    "exit_price": 101,
                })
                writer.writerow({
                    "id": 2, "symbol": "B", "direction": "LONG", "entry_price": 100,
                    "sl": 99, "opened_at": 3, "closed_at": 4, "status": "SL_HIT",
                    "realized_r": -1, "confidence": "LOW", "session": "OFF",
                    "entry_source": "OB", "trend_1h": "neutral", "trend_4h": "neutral",
                    "exit_price": 99,
                })

            setup_fields = ["signal_id", "decision", "trend", "source", "net_r"]
            with setups.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=setup_fields)
                writer.writeheader()
                writer.writerow({"signal_id": 1, "decision": "LONG", "trend": "bull", "source": "live", "net_r": 1})
                writer.writerow({"signal_id": 2, "decision": "NO TRADE", "trend": "range", "source": "live", "net_r": -1})

            argv = [
                "analyze_live_exports.py", str(signals), str(setups),
                "--round-trip-bps", "12", "--stress-bps", "22", "--out", str(output),
            ]
            with patch.object(sys, "argv", argv):
                analyze_live_exports.main()

            result = json.loads(output.read_text(encoding="utf-8"))
            self.assertAlmostEqual(result["metrics"]["net_r"], -0.24)
            self.assertAlmostEqual(result["metrics"]["max_drawdown_r"], 1.12)
            self.assertAlmostEqual(result["stress_net_r"], -0.44)
            self.assertEqual(result["gate_groups"]["accepted"]["n"], 1)
            self.assertEqual(result["gate_groups"]["rejected"]["n"], 1)
            self.assertEqual(result["gate_groups"]["accepted_and_aligned"]["n"], 1)
            self.assertEqual(result["join_integrity"]["missing_setup_links"], [])


if __name__ == "__main__":
    unittest.main()
