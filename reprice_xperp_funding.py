"""Reprice frozen X-Perp trades with public realized funding-rate history."""
import csv
import hashlib
import json
import math
from pathlib import Path

from filter_lab import metrics


def funding_impact(direction, rate, risk_pct, weight=1.0):
    if direction not in ("LONG", "SHORT"):
        raise ValueError("Invalid direction")
    values = (float(rate), float(risk_pct), float(weight))
    if not all(math.isfinite(value) for value in values) or values[1] <= 0 or values[2] <= 0:
        raise ValueError("Invalid funding inputs")
    sign = 1 if direction == "LONG" else -1
    return -sign * values[0] / values[1] * values[2]


def main():
    folder = Path("reports/audit_2026_09_08")
    trade_path = folder / "venue_expand_xperp.csv"
    funding_path = folder / "xperp_funding_history.json"
    funding = json.loads(funding_path.read_text(encoding="utf-8"))
    with trade_path.open(newline="", encoding="utf-8") as handle:
        source = list(csv.DictReader(handle))
    repriced, excluded, boundary_events = [], 0, 0
    for row in source:
        info = funding["symbols"][row["symbol"]]
        entry_ms = int(float(row["entry_time"]) * 1000)
        exit_ms = int(float(row["exit_time"]) * 1000)
        if entry_ms < int(info["coverage_start_ms"]):
            excluded += 1
            continue
        risk_pct = abs(float(row["entry"])-float(row["sl"]))/float(row["entry"])
        weight = float(row["size_mult"])
        if not math.isfinite(risk_pct) or risk_pct <= 0:
            raise ValueError("Invalid trade risk")
        funding_r = 0.0
        pessimistic_boundary_r = 0.0
        events = 0
        for event in info["rates"]:
            event_ms = int(event["fundingTime"])
            rate = float(event.get("realizedRate") or event["fundingRate"])
            if not math.isfinite(rate):
                raise ValueError("Nonfinite funding rate")
            impact = funding_impact(row["direction"], rate, risk_pct, weight)
            if entry_ms < event_ms < exit_ms:
                funding_r += impact
                events += 1
            elif event_ms in (entry_ms, exit_ms):
                boundary_events += 1
                pessimistic_boundary_r += min(0.0, impact)
        adjusted = float(row["net_r"]) + funding_r
        repriced.append({
            "symbol": row["symbol"], "direction": row["direction"],
            "entry_time": float(row["entry_time"]), "exit_time": float(row["exit_time"]),
            "outcome": row["outcome"], "risk_pct": risk_pct,
            "size_mult": weight, "net_r": adjusted,
            "funding_r": funding_r, "funding_events": events,
            "pessimistic_net_r": adjusted + pessimistic_boundary_r,
        })
    pessimistic = [{**row, "net_r": row["pessimistic_net_r"]} for row in repriced]
    report = {
        "status": "RESEARCH_ONLY",
        "trade_sha256": hashlib.sha256(trade_path.read_bytes()).hexdigest(),
        "funding_sha256": hashlib.sha256(funding_path.read_bytes()).hexdigest(),
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "source_trades": len(source), "eligible_trades": len(repriced),
        "excluded_before_funding_coverage": excluded,
        "boundary_events": boundary_events,
        "strict_interior_events": sum(row["funding_events"] for row in repriced),
        "funding_r": sum(row["funding_r"] for row in repriced),
        "adjusted": metrics(repriced),
        "pessimistic_boundary": metrics(pessimistic),
        "limitations": [
            "Only trades fully inside public funding coverage are included.",
            "Funding uses entry-notional approximation; actual mark notional is unavailable here.",
            "Events exactly at entry/exit are excluded in adjusted and only negative impacts included in the pessimistic boundary case.",
            "Modeled OHLC trades, not account bills or fills.",
        ],
    }
    (folder / "venue_expand_xperp_funding.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report))


if __name__ == "__main__":
    main()
