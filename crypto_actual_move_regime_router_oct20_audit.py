"""Cross-era BTC-regime router for exact 0.5%-2% crypto move entries."""
from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

import crypto_actual_move_health_gate_oct6_audit as source
import crypto_5m_tree_filter_audit as candle_source
from crypto_runtime_shadow_portfolio_oct3_audit import money, runtime_gate


ROOT = Path("reports/audit_2026_09_08")
OUT = ROOT / "crypto_actual_move_regime_router_oct20.json"
ROWS_OUT = ROOT / "crypto_actual_move_regime_router_oct20.pkl"
FIELD_SETS = (
    ("module", "direction", "btc_trend"),
    ("module", "direction", "btc_impulse"),
    ("module", "direction", "btc_trend", "btc_impulse"),
    ("module", "direction", "btc_trend", "vol_regime"),
    ("module", "direction", "btc_impulse", "vol_regime"),
    ("module", "direction", "btc_trend", "session"),
    ("module", "direction", "btc_trend", "btc_impulse", "vol_regime"),
    ("module", "direction", "btc_trend", "btc_impulse", "session"),
)
MIN_COUNTS = (10, 20, 40)
MIN_MEANS = (0.0, .0001, .00025)
MIN_WIN_RATES = (.50, .55, .60)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def btc_context(candle: dict) -> pd.DataFrame:
    """Build features from bars fully closed before each market entry."""
    frame = pd.DataFrame({
        "entry_time": np.asarray(candle["time"], dtype=np.int64),
        "high": np.asarray(candle["high"], dtype=float),
        "low": np.asarray(candle["low"], dtype=float),
        "close": np.asarray(candle["close"], dtype=float),
    })
    previous_close = frame.close.shift(1)
    fast = previous_close.rolling(288).mean()
    slow = previous_close.rolling(1152).mean()
    trend = np.full(len(frame), "transition", dtype=object)
    trend[(previous_close > fast) & (fast > slow)] = "bull"
    trend[(previous_close < fast) & (fast < slow)] = "bear"
    impulse_value = previous_close / previous_close.shift(12) - 1
    impulse = np.full(len(frame), "flat", dtype=object)
    impulse[impulse_value > .005] = "up"
    impulse[impulse_value < -.005] = "down"
    prior = frame.close.shift(2)
    tr = pd.concat(((frame.high.shift(1) - frame.low.shift(1)),
                    (frame.high.shift(1) - prior).abs(),
                    (frame.low.shift(1) - prior).abs()), axis=1).max(axis=1)
    atr_pct = tr.rolling(288).mean() / previous_close
    trailing_median = atr_pct.shift(1).rolling(8640, min_periods=2016).median()
    volatility = np.where(atr_pct > trailing_median, "high", "low")
    stamp = pd.to_datetime(frame.entry_time, unit="s", utc=True)
    return pd.DataFrame({
        "entry_time": frame.entry_time, "btc_trend": trend,
        "btc_impulse": impulse, "btc_impulse_value": impulse_value,
        "vol_regime": volatility,
        "session": (stamp.dt.hour // 6).astype(int),
    })


def attach_context(rows: list[dict], btc: dict) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    context = btc_context(btc)
    found = frame.merge(context, on="entry_time", how="left", validate="many_to_one")
    return found.dropna(subset=["btc_trend", "btc_impulse", "vol_regime"])


def evaluate(rows: list[dict], year: int, margin: float) -> dict:
    accepted = runtime_gate([row for row in rows if int(row["year"]) == year])
    result = money(accepted, margin)
    result["literal_target_hit_rate"] = round(
        100 * sum(bool(row["literal_target_hit"]) for row in accepted)
        / len(accepted), 4) if accepted else 0.0
    return result


def floor_month(periods: dict[str, dict]) -> float:
    values = [month for period in periods.values()
              for month in period.get("monthly_return_pct", {}).values()]
    return min(map(float, values), default=-999.0)


def main() -> None:
    rows = source.source_rows()
    candles, candle_sources = candle_source.load()
    frame = attach_context(rows, candles["BTCUSDT"])
    development = frame[frame.year.isin((2022, 2023, 2024))]
    candidates = []
    for fields, minimum, mean_min, win_min in itertools.product(
            FIELD_SETS, MIN_COUNTS, MIN_MEANS, MIN_WIN_RATES):
        grouped = development.groupby([*fields, "year"], observed=True).agg(
            n=("signed_price_return", "size"),
            mean=("signed_price_return", "mean"),
            win=("signed_price_return", lambda values: (values > 0).mean()),
        ).reset_index()
        allowed = []
        for key, values in grouped.groupby(list(fields), observed=True):
            if (set(values.year) == {2022, 2023, 2024}
                    and values.n.min() >= minimum
                    and values["mean"].min() >= mean_min
                    and values.win.min() >= win_min):
                allowed.append(key if isinstance(key, tuple) else (key,))
        if not allowed:
            continue
        allowed_set = set(allowed)
        mask = frame.loc[:, fields].apply(tuple, axis=1).isin(allowed_set)
        selected = frame[mask].to_dict("records")
        periods = {str(year): evaluate(selected, year, .03)
                   for year in (2022, 2023, 2024)}
        admissible = all(
            value["n"] >= 500 and value["return_pct"] > 0
            and value["dd_pct"] <= 8
            and value["trades_per_calendar_day"] >= 2.5
            for value in periods.values())
        candidates.append({
            "fields": list(fields), "minimum_each_year": minimum,
            "minimum_mean_price_return": mean_min,
            "minimum_win_rate": win_min, "allowed_buckets": allowed,
            "development": periods,
            "development_floor_month_pct": round(floor_month(periods), 4),
            "development_worst_return_pct": min(
                value["return_pct"] for value in periods.values()),
            "development_max_dd_pct": max(
                value["dd_pct"] for value in periods.values()),
            "admissible": admissible, "_rows": selected,
        })
    candidates.sort(key=lambda item: (
        item["admissible"], item["development_floor_month_pct"],
        item["development_worst_return_pct"],
        -item["development_max_dd_pct"]), reverse=True)
    selected = candidates[0] if candidates else None
    holdouts = None
    if selected:
        holdouts = {
            f"margin_{int(margin * 100)}pct": {
                str(year): evaluate(selected["_rows"], year, margin)
                for year in (2025, 2026)
            } for margin in (.03, .05)
        }
        pd.DataFrame(selected["_rows"]).to_pickle(ROWS_OUT)
    passed = bool(
        selected and selected["admissible"] and holdouts
        and all(value["return_pct"] > 0 and value["dd_pct"] <= 10
                and value["negative_months"] == 0
                and value["trades_per_calendar_day"] >= 3
                for value in holdouts["margin_3pct"].values()))

    def public(item):
        return ({key: value for key, value in item.items() if key != "_rows"}
                if item else None)

    report = {
        "status": ("REGIME_ROUTER_AWAITING_DIRECT_VENUE_FORWARD" if passed
                   else "REJECTED_OBJECTIVE"),
        "ready_for_live_orders": False,
        "protocol": {
            "policy_selection": [2022, 2023, 2024],
            "holdouts": [2025, 2026],
            "features": "BTC bars closed before the next-open market entry",
            "entry": "next 5m market open", "orders": "market only",
            "money": "client 3% or 5% current-equity margin; fixed 10x",
        },
        "source_rows": len(frame), "candidate_count": len(candidates),
        "admissible_count": sum(item["admissible"] for item in candidates),
        "selected_before_holdouts": public(selected),
        "top20_before_holdouts": [public(item) for item in candidates[:20]],
        "holdouts": holdouts, "objective_pass": passed,
        "sources": {**candle_sources,
                    str(Path(__file__).resolve()): digest(Path(__file__).resolve())},
        "limitations": [
            "Component definitions came from earlier finite searches.",
            "The regime grid adds selection risk.",
            "Binance spot history still requires direct X-Perp transfer.",
            "Gross results exclude fees at user request.",
        ],
    }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({
        "status": report["status"], "candidate_count": len(candidates),
        "admissible_count": report["admissible_count"],
        "selected": public(selected), "holdouts": holdouts,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
