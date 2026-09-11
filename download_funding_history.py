"""Download public realized X-Perp funding rates for frozen venue windows."""
import argparse
import hashlib
import json
import pickle
import time
from pathlib import Path

import requests


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", required=True)
    args = parser.parse_args()
    symbols = args.symbols.split(",")
    folder = Path("reports/audit_2026_09_08")
    instruments = json.loads(
        (folder / "public_xperp_instruments.json").read_text(encoding="utf-8")
    )["data"]
    mapping = {
        item["instId"].split("-")[0] + "USDT": item["instId"]
        for item in instruments
        if "_UM_XPERP-" in item.get("instId", "") and item.get("state") == "live"
    }
    result = {"status": "PUBLIC_FUNDING_DATA", "symbols": {}, "source_sha256": None}
    for symbol in symbols:
        instrument = mapping.get(symbol)
        if not instrument:
            raise ValueError(f"No X-Perp instrument for {symbol}")
        candle_path = folder / "xperp_cache" / f"{symbol}_15min_18000.pkl"
        candles = pickle.loads(candle_path.read_bytes())
        start_ms = min(candles["time"]) * 1000
        after = None
        rates = {}
        for _ in range(10):
            params = {"instId": instrument, "limit": "400"}
            if after is not None:
                params["after"] = str(after)
            response = requests.get(
                "https://www.okx.com/api/v5/public/funding-rate-history",
                params=params, timeout=25,
            )
            response.raise_for_status()
            body = response.json()
            if body.get("code") != "0":
                raise RuntimeError(str(body))
            batch = body.get("data", [])
            if not batch:
                break
            for row in batch:
                funding_time = int(row["fundingTime"])
                if funding_time >= start_ms:
                    rates[funding_time] = row
            oldest = min(int(row["fundingTime"]) for row in batch)
            if oldest <= start_ms:
                break
            if after is not None and oldest >= after:
                raise RuntimeError(f"Funding pagination stalled for {symbol}")
            after = oldest
            time.sleep(0.15)
        ordered = [rates[key] for key in sorted(rates)]
        if not ordered:
            raise ValueError(f"No funding history for {symbol}")
        coverage_start = int(ordered[0]["fundingTime"])
        result["symbols"][symbol] = {
            "instrument": instrument,
            "venue_start_ms": start_ms,
            "coverage_start_ms": coverage_start,
            "truncated_before_coverage": coverage_start > start_ms + 8 * 3600 * 1000,
            "rates": ordered,
        }
        print(json.dumps({"symbol": symbol, "rates": len(ordered),
                          "start": ordered[0]["fundingTime"],
                          "end": ordered[-1]["fundingTime"]}), flush=True)
    result["source_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (folder / "xperp_funding_history.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
