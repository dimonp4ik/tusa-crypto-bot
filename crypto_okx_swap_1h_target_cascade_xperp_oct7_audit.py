"""No-selection EU X-Perp transfer of the frozen OKX 1H target cascade."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

import crypto_okx_swap_1h_target_cascade_oct7_audit as cascade
import crypto_okx_swap_1h_xperp_transfer_oct7_audit as transfer
import crypto_5m_regime_router_literal_oct6_audit as portfolio


ROOT = Path("reports/audit_2026_09_08")
ROWS = ROOT / "crypto_okx_swap_1h_target_cascade_oct7.pkl"
REPORT = ROOT / "crypto_okx_swap_1h_target_cascade_oct7.json"
OUT = ROOT / "crypto_okx_swap_1h_target_cascade_xperp_oct7.json"
ROWS_OUT = ROOT / "crypto_okx_swap_1h_target_cascade_xperp_oct7.pkl"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def direct_rows(entries: pd.DataFrame, candles: dict) -> pd.DataFrame:
    parts = []
    for target, target_rows in entries.groupby("selected_target_pct", sort=True):
        target = float(target)
        hold = cascade.HOLD[target] * 12
        mapped = transfer.map_entries(target_rows, candles, hold)
        for symbol, part in mapped.groupby("symbol", sort=False):
            candle = candles[symbol]
            arrays = [np.asarray(candle[key], dtype=float)
                      for key in ("open", "high", "low", "close")]
            values, exits, hits, reasons = transfer.resolve_fixed(
                *arrays, np.asarray(candle["time"], dtype=np.int64),
                part.entry_index.to_numpy(np.int64),
                part.direction.to_numpy(np.int8), target, 1.0, hold)
            found = part.copy()
            found["signed_return"], found["exit_time"] = values, exits
            found["target_hit"], found["exit_reason"] = hits.astype(bool), reasons
            found["quality"] = 1.0
            parts.append(found)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def main() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    if report.get("status") not in ("CASCADE_RESEARCH_CANDIDATE", "REJECTED_LATER"):
        raise ValueError("cascade did not freeze before direct transfer")
    entries = pd.read_pickle(ROWS)
    candles, sources = transfer.load_xperp()
    start = max(candle["time"][0] for candle in candles.values())
    end = min(candle["time"][-1] for candle in candles.values()) + 300
    entries = entries[entries.entry_time.between(start, end, inclusive="left")].copy()
    direct = direct_rows(entries, candles)
    direct["year"] = pd.to_datetime(direct.entry_time, unit="s", utc=True).dt.year
    result = cascade.evaluate(direct, tuple(sorted(direct.year.unique())))
    by_target = {}
    for target, part in direct.groupby("selected_target_pct", sort=True):
        accepted = portfolio.portfolio_rows(part, tuple(sorted(part.year.unique())), .5)
        by_target[str(float(target))] = {
            "available": len(part), "accepted_alone": len(accepted),
            "literal_hits_alone": sum(bool(row["target_hit"]) for row in accepted),
            "literal_hit_rate_alone": round(
                100 * sum(bool(row["target_hit"]) for row in accepted) / len(accepted), 4)
                if accepted else 0.0,
        }
    output = {
        "status": ("DIRECT_XPERP_CASCADE_CANDIDATE"
                   if result["return_pct"] > 0 and result["dd_pct"] <= 5
                   and result["literal_target_hit_rate"] >= 65
                   and result["trades_per_calendar_day"] >= 1.8
                   else "REJECTED_DIRECT_TRANSFER"),
        "protocol": {
            "selection": "none on X-Perp", "venue": "EU X-Perp",
            "entry": "same frozen global OKX hourly timestamps",
            "targets_pct": list(cascade.TARGETS), "stop_pct": 1.0,
            "hold_hours": cascade.HOLD,
            "money": "3% current equity margin, fixed 10x gross",
        },
        "available_entry_rows": len(entries), "mapped_rows": len(direct),
        "result": result, "by_target": by_target,
        "coverage": {"start": start, "end_exclusive": end},
        "sources": {**sources, str(ROWS): digest(ROWS), str(REPORT): digest(REPORT),
                    str(Path(__file__)): digest(Path(__file__))},
    }
    direct.to_pickle(ROWS_OUT)
    OUT.write_text(json.dumps(output, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(output, indent=2), flush=True)


if __name__ == "__main__":
    main()
