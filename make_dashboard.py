#!/usr/bin/env python3
"""
Create a standalone HTML dashboard from backtest.py --export-trades CSV.

Example:
  python make_dashboard.py work_top20_allfeatures.csv --out dashboard.html --start 1000 --risk 0.01 --use-risk-mult
"""

from __future__ import annotations

import argparse
import csv
import html
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


WIN_OUTCOMES = {"TP1", "TP2", "TRAIL"}


def _float(value, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _month(ts) -> str:
    t = _float(ts, 0.0)
    if t <= 0:
        return "unknown"
    return datetime.fromtimestamp(t, tz=timezone.utc).strftime("%Y-%m")


def _fmt(value: float, digits: int = 2) -> str:
    return f"{value:,.{digits}f}"


def _pct(value: float) -> str:
    return f"{value:.1f}%"


def load_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    rows.sort(key=lambda r: (_float(r.get("entry_time")), r.get("symbol", "")))
    return rows


def summarize(rows: list[dict]) -> dict:
    trades = len(rows)
    wins = sum(1 for r in rows if r.get("outcome") in WIN_OUTCOMES)
    net_r = sum(_float(r.get("net_r")) for r in rows)
    gross_r = sum(_float(r.get("gross_r")) for r in rows)
    outcomes = Counter(r.get("outcome", "UNKNOWN") for r in rows)
    return {
        "trades": trades,
        "wins": wins,
        "win_rate": wins / trades * 100.0 if trades else 0.0,
        "net_r": net_r,
        "gross_r": gross_r,
        "net_rpt": net_r / trades if trades else 0.0,
        "outcomes": outcomes,
    }


def equity_curve(rows: list[dict], start: float, risk: float, use_risk_mult: bool) -> tuple[list[float], float, float]:
    equity = start
    peak = start
    max_dd = 0.0
    curve = [equity]
    for row in rows:
        mult = _float(row.get("risk_mult"), 1.0) if use_risk_mult else 1.0
        equity += equity * risk * mult * _float(row.get("net_r"))
        peak = max(peak, equity)
        if peak > 0:
            max_dd = min(max_dd, (equity - peak) / peak * 100.0)
        curve.append(equity)
    return curve, equity, max_dd


def grouped_table(rows: list[dict], key: str) -> list[dict]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        value = _month(row.get("entry_time")) if key == "month" else (row.get(key) or "unknown")
        groups[value].append(row)

    out = []
    for name, items in groups.items():
        s = summarize(items)
        out.append({
            "name": name,
            "trades": s["trades"],
            "win_rate": s["win_rate"],
            "net_r": s["net_r"],
            "avg_q": sum(_float(r.get("quality_score")) for r in items) / len(items) if items else 0.0,
            "avg_mult": sum(_float(r.get("risk_mult"), 1.0) for r in items) / len(items) if items else 1.0,
        })
    return sorted(out, key=lambda x: x["name"])


def svg_curve(points: list[float], width: int = 900, height: int = 240) -> str:
    if len(points) < 2:
        return ""
    lo, hi = min(points), max(points)
    if math.isclose(lo, hi):
        hi = lo + 1.0
    step = width / (len(points) - 1)
    coords = []
    for i, value in enumerate(points):
        x = i * step
        y = height - ((value - lo) / (hi - lo) * (height - 20) + 10)
        coords.append(f"{x:.1f},{y:.1f}")
    return (
        f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="Equity curve">'
        f'<rect width="{width}" height="{height}" fill="#101418"/>'
        f'<polyline points="{" ".join(coords)}" fill="none" stroke="#4cc9f0" stroke-width="3"/>'
        f'<text x="12" y="24" fill="#cbd5e1" font-size="14">High {hi:.2f}</text>'
        f'<text x="12" y="{height - 12}" fill="#cbd5e1" font-size="14">Low {lo:.2f}</text>'
        "</svg>"
    )


def render_table(rows: list[dict], title: str) -> str:
    body = []
    for row in rows:
        body.append(
            "<tr>"
            f"<td>{html.escape(str(row['name']))}</td>"
            f"<td>{row['trades']}</td>"
            f"<td>{_pct(row['win_rate'])}</td>"
            f"<td>{row['net_r']:+.2f}R</td>"
            f"<td>{row['avg_q']:.1f}</td>"
            f"<td>x{row['avg_mult']:.2f}</td>"
            "</tr>"
        )
    return (
        f"<h2>{html.escape(title)}</h2>"
        "<table><thead><tr><th>Name</th><th>Trades</th><th>Win rate</th>"
        "<th>Net R</th><th>Avg Q</th><th>Avg risk</th></tr></thead>"
        f"<tbody>{''.join(body)}</tbody></table>"
    )


def build_html(rows: list[dict], source: Path, start: float, risk: float, use_risk_mult: bool) -> str:
    s = summarize(rows)
    curve, final_equity, max_dd = equity_curve(rows, start, risk, use_risk_mult)
    outcomes = " ".join(f"{html.escape(k)}: {v}" for k, v in sorted(s["outcomes"].items()))
    month_table = render_table(grouped_table(rows, "month"), "By Month")
    pack_table = render_table(grouped_table(rows, "adaptive_pack"), "By Adaptive Pack")
    generated = datetime.now().strftime("%Y-%m-%d %H:%M")

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>TUSA Backtest Dashboard</title>
  <style>
    body {{ margin: 0; font-family: Arial, sans-serif; background: #0b0f14; color: #e5e7eb; }}
    main {{ max-width: 1120px; margin: 0 auto; padding: 28px; }}
    h1, h2 {{ margin: 0 0 14px; }}
    h2 {{ margin-top: 28px; font-size: 20px; }}
    .meta {{ color: #94a3b8; margin-bottom: 20px; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(155px, 1fr)); gap: 10px; }}
    .metric {{ border: 1px solid #223042; padding: 14px; border-radius: 8px; background: #111827; }}
    .metric b {{ display: block; font-size: 22px; margin-top: 6px; }}
    table {{ width: 100%; border-collapse: collapse; margin-top: 8px; background: #111827; }}
    th, td {{ border-bottom: 1px solid #263244; padding: 9px 10px; text-align: right; }}
    th:first-child, td:first-child {{ text-align: left; }}
    th {{ color: #93c5fd; font-weight: 600; }}
    svg {{ width: 100%; height: auto; border: 1px solid #223042; border-radius: 8px; margin-top: 12px; }}
    .note {{ color: #94a3b8; margin-top: 12px; line-height: 1.45; }}
  </style>
</head>
<body>
<main>
  <h1>TUSA Backtest Dashboard</h1>
  <div class="meta">Source: {html.escape(str(source))} · Generated: {generated}</div>
  <div class="grid">
    <div class="metric">Trades<b>{s['trades']}</b></div>
    <div class="metric">Win rate<b>{_pct(s['win_rate'])}</b></div>
    <div class="metric">Net R<b>{s['net_r']:+.2f}R</b></div>
    <div class="metric">R / trade<b>{s['net_rpt']:+.3f}</b></div>
    <div class="metric">Final equity<b>{_fmt(final_equity)}</b></div>
    <div class="metric">Max DD<b>{_pct(max_dd)}</b></div>
  </div>
  {svg_curve(curve)}
  <div class="note">Outcomes: {outcomes}. Equity uses start={_fmt(start)}, risk={risk * 100:.2f}% and risk_mult={'on' if use_risk_mult else 'off'}.</div>
  {month_table}
  {pack_table}
</main>
</body>
</html>"""


def main() -> int:
    parser = argparse.ArgumentParser(description="Build HTML dashboard from backtest CSV.")
    parser.add_argument("csv_path", type=Path)
    parser.add_argument("--out", type=Path, default=Path("backtest_dashboard.html"))
    parser.add_argument("--start", type=float, default=1000.0)
    parser.add_argument("--risk", type=float, default=0.01)
    parser.add_argument("--use-risk-mult", action="store_true")
    args = parser.parse_args()

    rows = load_rows(args.csv_path)
    args.out.write_text(build_html(rows, args.csv_path, args.start, args.risk, args.use_risk_mult), encoding="utf-8")
    print(f"Dashboard written: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
