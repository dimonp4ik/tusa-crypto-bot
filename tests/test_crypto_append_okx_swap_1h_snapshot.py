import hashlib
import pickle
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import crypto_append_okx_swap_1h_snapshot as snapshot


def row(timestamp, value):
    return [str(timestamp * 1000), *[str(value)] * 7, "1"]


class AppendOkxSnapshotTests(unittest.TestCase):
    def source(self, root: Path):
        candle = {"time": [0, 3600], **{
            field: [1.0, 2.0] for field in snapshot.FIELDS}}
        raw = pickle.dumps(candle, protocol=pickle.HIGHEST_PROTOCOL)
        path = root / "BTCUSDT.pkl"
        path.write_bytes(raw)
        return {
            "instrument": "BTC-USDT-SWAP", "path": str(path),
            "sha256": hashlib.sha256(raw).hexdigest(),
        }

    def test_appends_only_contiguous_confirmed_candles(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            item = self.source(root)
            output = root / "new"
            output.mkdir()
            with patch.object(snapshot, "fetch_latest", return_value=[
                    row(7200, 3), row(3600, 2)]):
                _, meta = snapshot.append_symbol(
                    "BTCUSDT", item, root / "manifest.json", output,
                    10_000_000)
            candle = pickle.loads((output / "BTCUSDT.pkl").read_bytes())
            self.assertEqual(candle["time"], [0, 3600, 7200])
            self.assertEqual(meta["appended_bars"], 1)
            self.assertEqual(meta["overlap_bars_verified"], 1)

    def test_rejects_revision_of_confirmed_overlap(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            item = self.source(root)
            output = root / "new"
            output.mkdir()
            with patch.object(snapshot, "fetch_latest", return_value=[
                    row(7200, 3), row(3600, 9)]):
                with self.assertRaisesRegex(ValueError, "revision"):
                    snapshot.append_symbol(
                        "BTCUSDT", item, root / "manifest.json", output,
                        10_000_000)

    def test_rejects_gap_before_first_new_candle(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            item = self.source(root)
            output = root / "new"
            output.mkdir()
            with patch.object(snapshot, "fetch_latest", return_value=[
                    row(10800, 4), row(3600, 2)]):
                with self.assertRaisesRegex(ValueError, "append gap"):
                    snapshot.append_symbol(
                        "BTCUSDT", item, root / "manifest.json", output,
                        20_000_000)


if __name__ == "__main__":
    unittest.main()
