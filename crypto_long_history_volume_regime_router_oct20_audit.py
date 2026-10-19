"""Causal volume/market-regime router over long-history cross-section rules.

The diverse rule library is selected on 2020-2022 only.  Regime experts must
show positive price expectancy in each of those years independently.  Router
policy is selected on 2023-2024 and revealed once on 2025-2026.
"""
from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

import crypto_long_history_hourly_cross_section_oct20_audit as base
import crypto_long_history_monthly_style_router_oct20_audit as monthly
import crypto_raw_hourly_direction_oct19_audit as portfolio


OUT = base.ROOT / "crypto_long_history_volume_regime_router_oct20.json"
ROWS_OUT = base.ROOT / "crypto_long_history_volume_regime_router_oct20.pkl"
FIELD_SETS = (
    ("variant", "direction", "btc_trend", "volume_regime"),
    ("variant", "direction", "breadth_alignment", "volume_regime"),
    ("variant", "direction", "btc_eff_regime", "dispersion_regime"),
    ("variant", "direction", "btc_trend_alignment", "btc_eff_regime",
     "volume_regime"),
    ("variant", "direction", "breadth_alignment", "dispersion_regime",
     "volume_regime"),
    ("variant", "direction", "btc_trend_alignment", "breadth_alignment"),
)
MIN_COUNTS = (20, 50)
MIN_YEAR_MEANS = (0.0, .00025)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rolling_sum(values: np.ndarray, window: int) -> np.ndarray:
    return pd.Series(values).rolling(window).sum().to_numpy(float)


def market_context(candles: dict[str, dict]) -> tuple[pd.DataFrame, dict]:
    """Compute hourly context from bars closed immediately before entry."""
    pieces = []
    for symbol, raw in candles.items():
        times = np.asarray(raw["time"], dtype=np.int64)
        close = np.asarray(raw["close"], dtype=float)
        volume = np.asarray(raw["volume"], dtype=float)
        indices = np.arange(288, len(times) - 1, dtype=np.int64)
        mask = ((times[indices] + 300) % 3600 == 0)
        mask &= times[indices + 1] - times[indices] == 300
        indices = indices[mask]
        absolute_change = np.abs(np.diff(close, prepend=close[0]))
        path = _rolling_sum(absolute_change, 288)
        efficiency = np.divide(
            np.abs(close[indices] - close[indices - 288]), path[indices],
            out=np.full(len(indices), np.nan), where=path[indices] > 0)
        hourly_volume = _rolling_sum(volume, 12)[indices]
        baseline = pd.Series(hourly_volume).rolling(168).median().shift(1)
        volume_ratio = np.divide(
            hourly_volume, baseline.to_numpy(float),
            out=np.full(len(indices), np.nan),
            where=baseline.to_numpy(float) > 0)
        ret1 = close[indices] / close[indices - 12] - 1
        ret4 = close[indices] / close[indices - 48] - 1
        ret24 = close[indices] / close[indices - 288] - 1
        hourly_sigma = pd.Series(ret1).rolling(168).std().shift(1).to_numpy(float)
        trend_z = np.divide(
            ret24, hourly_sigma * np.sqrt(24),
            out=np.full(len(indices), np.nan), where=hourly_sigma > 0)
        pieces.append(pd.DataFrame({
            "entry_time": times[indices] + 300, "symbol": symbol,
            "ret4": ret4, "ret24": ret24,
            "volume_ratio": volume_ratio, "efficiency24": efficiency,
            "trend_z24": trend_z,
        }))
    long = pd.concat(pieces, ignore_index=True).dropna()
    ret4 = long.pivot(index="entry_time", columns="symbol", values="ret4")
    volume = long.pivot(
        index="entry_time", columns="symbol", values="volume_ratio")
    context = pd.DataFrame(index=ret4.index)
    context["market_universe"] = ret4.notna().sum(axis=1)
    context["breadth_positive"] = (ret4 > 0).sum(axis=1) / context.market_universe
    context["dispersion"] = ret4.std(axis=1)
    context["volume_ratio"] = volume.median(axis=1)
    btc = long[long.symbol.eq("BTCUSDT")].set_index("entry_time")
    context = context.join(btc[["trend_z24", "efficiency24"]], how="left")
    context = context[context.market_universe.eq(len(base.SYMBOLS))].dropna()
    years = pd.to_datetime(context.index, unit="s", utc=True).year
    development = context[np.asarray(years) <= 2022]
    thresholds = {
        "dispersion": [float(value) for value in
                       development.dispersion.quantile([1/3, 2/3])],
        "volume_ratio": [float(value) for value in
                         development.volume_ratio.quantile([1/3, 2/3])],
    }
    context["btc_trend"] = pd.cut(
        context.trend_z24, [-np.inf, -1, 1, np.inf],
        labels=["bear", "flat", "bull"]).astype(str)
    context["btc_eff_regime"] = pd.cut(
        context.efficiency24, [-np.inf, .2, .4, np.inf],
        labels=["range", "mixed", "trend"]).astype(str)
    context["volume_regime"] = pd.cut(
        context.volume_ratio,
        [-np.inf, *thresholds["volume_ratio"], np.inf],
        labels=["low", "normal", "high"]).astype(str)
    context["dispersion_regime"] = pd.cut(
        context.dispersion,
        [-np.inf, *thresholds["dispersion"], np.inf],
        labels=["low", "normal", "high"]).astype(str)
    return context.reset_index(), thresholds


def enrich(library: dict[str, list[dict]], items: list[dict],
           context: pd.DataFrame) -> pd.DataFrame:
    metadata = {monthly.variant_id(item): item for item in items}
    rows = []
    for name, values in library.items():
        item = metadata[name]
        for original in values:
            row = dict(original)
            row.update({
                "variant": name, "style": str(item["style"]),
                "lookback_hours": int(item["lookback_hours"]),
                "target_pct": float(item["target_pct"]),
            })
            rows.append(row)
    frame = pd.DataFrame(rows).merge(
        context, on="entry_time", how="inner", validate="many_to_one")
    direction = np.where(frame.direction.eq("LONG"), 1.0, -1.0)
    aligned_trend = direction * frame.trend_z24
    aligned_breadth = np.where(
        direction > 0, frame.breadth_positive, 1-frame.breadth_positive)
    frame["btc_trend_alignment"] = pd.cut(
        aligned_trend, [-np.inf, -1, 1, np.inf],
        labels=["adverse", "flat", "supportive"]).astype(str)
    frame["breadth_alignment"] = pd.cut(
        aligned_breadth, [-np.inf, .4, .6, np.inf],
        labels=["adverse", "mixed", "supportive"]).astype(str)
    frame["year"] = pd.to_datetime(
        frame.entry_time, unit="s", utc=True).dt.year
    return frame


def stable_experts(frame: pd.DataFrame, fields: tuple[str, ...],
                   min_count: int, min_year_mean: float) -> pd.DataFrame:
    development = frame[frame.year.isin(base.DEVELOPMENT_YEARS)]
    stats = development.groupby(
        ["year", *fields], dropna=False).signed_price_return.agg(
            count="size", mean_return="mean",
            win_rate=lambda values: float((values > 0).mean())).reset_index()
    good = stats[
        stats["count"].ge(min_count)
        & stats.mean_return.ge(min_year_mean)]
    if good.empty:
        return pd.DataFrame()
    stable = good.groupby(list(fields), dropna=False).filter(
        lambda part: set(part.year.astype(int)) == set(base.DEVELOPMENT_YEARS))
    if stable.empty:
        return pd.DataFrame()
    return stable.groupby(list(fields), dropna=False).agg(
        worst_year_mean=("mean_return", "min"),
        mean_return=("mean_return", "mean"),
        worst_year_win_rate=("win_rate", "min"),
        observations=("count", "sum")).reset_index()


def route(frame: pd.DataFrame, fields: tuple[str, ...], experts: pd.DataFrame,
          loss_pause: int, hourly_cap: int) -> list[dict]:
    if experts.empty:
        return []
    selected = frame.merge(
        experts[[*fields, "worst_year_mean", "mean_return"]],
        on=list(fields), how="inner", validate="many_to_one")
    selected = selected.sort_values(
        ["entry_time", "worst_year_mean", "mean_return", "symbol", "variant"],
        ascending=[True, False, False, True, True], kind="mergesort")
    unique = selected.drop_duplicates(
        ["symbol", "entry_time"], keep="first").copy()
    rows = unique.to_dict("records")
    for row in rows:
        row["model_probability"] = float(row["worst_year_mean"])
        row["module"] = f"volume_regime:{row['variant']}"
        row["source"] = "long_history_volume_regime"
    return portfolio.rails.ranked_runtime_gate(
        rows, loss_pause=loss_pause, hourly_cap=hourly_cap)


def stage(rows: list[dict], years: tuple[int, ...], margin: float) -> dict:
    selected = [row for row in rows if pd.Timestamp(
        float(row["entry_time"]), unit="s", tz="UTC").year in years]
    return portfolio.money(selected, margin)


def rank(item: dict, key: str) -> tuple:
    value = item[key]
    return (
        value["dd_pct"] <= 15 and value["trades_per_calendar_day"] >= 3,
        -int(value["negative_months"]), base.month_floor(value),
        float(value["return_pct"]), -float(value["dd_pct"]),
    )


def public(item: dict) -> dict:
    return {key: value for key, value in item.items()
            if not key.startswith("_")}


def main() -> None:
    source = json.loads(base.OUT.read_text(encoding="utf-8"))
    items = source["development_diverse_library"]
    candles, candle_sources = base.load()
    library = monthly.build_library(candles, items)
    context, thresholds = market_context(candles)
    frame = enrich(library, items, context)
    candidates = []
    for fields, min_count, min_mean in itertools.product(
            FIELD_SETS, MIN_COUNTS, MIN_YEAR_MEANS):
        experts = stable_experts(frame, fields, min_count, min_mean)
        rows = route(frame, fields, experts, loss_pause=3, hourly_cap=1)
        candidates.append({
            "fields": list(fields), "minimum_count_each_year": min_count,
            "minimum_mean_each_year_pct": 100 * min_mean,
            "expert_count": len(experts),
            "development_2020_2022_3pct": stage(
                rows, base.DEVELOPMENT_YEARS, .03),
            "experts": experts.to_dict("records"),
        })
    candidates.sort(
        key=lambda item: rank(item, "development_2020_2022_3pct"),
        reverse=True)
    finalists = []
    for item in candidates[:10]:
        fields = tuple(item["fields"])
        experts = pd.DataFrame(item["experts"])
        for loss_pause, hourly_cap in itertools.product(
                base.LOSS_PAUSES, base.HOURLY_CAPS):
            rows = route(frame, fields, experts, loss_pause, hourly_cap)
            finalists.append({
                **item, "loss_pause": loss_pause, "hourly_cap": hourly_cap,
                "confirmation_2023_2024_3pct": stage(
                    rows, base.CONFIRMATION_YEARS, .03),
                "_rows": rows,
            })
    finalists.sort(key=lambda item: (
        *rank(item, "confirmation_2023_2024_3pct"),
        *rank(item, "development_2020_2022_3pct")), reverse=True)
    selected = finalists[0]
    hidden = {
        "margin_3pct": stage(selected["_rows"], base.HIDDEN_YEARS, .03),
        "margin_5pct": stage(selected["_rows"], base.HIDDEN_YEARS, .05),
    }
    hidden_rows = [row for row in selected["_rows"] if pd.Timestamp(
        float(row["entry_time"]), unit="s", tz="UTC").year
        in base.HIDDEN_YEARS]
    pd.DataFrame(hidden_rows).to_pickle(ROWS_OUT)
    three = hidden["margin_3pct"]
    objective = bool(
        three["negative_months"] == 0 and base.month_floor(three) >= 30
        and three["dd_pct"] <= 10
        and three["trades_per_calendar_day"] >= 3)
    report = {
        "status": ("VOLUME_REGIME_SHADOW_CANDIDATE"
                   if objective else "REJECTED_OBJECTIVE"),
        "ready_for_live_orders": False,
        "protocol": {
            "library_selection": "2020-2022 only",
            "expert_stability": "positive expectancy separately in 2020, 2021, 2022",
            "policy_selection": "2023-2024 only",
            "hidden_reveal": "2025-01 through available 2026 archive",
            "context": "closed-bar BTC trend/efficiency, breadth, volume and dispersion",
            "orders": "next-bar market entry only",
            "money": "client 3% or 5% current-equity margin; fixed 10x",
        },
        "context_thresholds_selected_on_2020_2022": thresholds,
        "library_size": len(library), "enriched_rows": len(frame),
        "candidate_count": len(candidates),
        "selected_before_hidden_reveal": public(selected),
        "development_frontier": [public(item) for item in candidates[:10]],
        "confirmation_frontier": [public(item) for item in finalists[:20]],
        "hidden_2025_2026": hidden, "objective_pass": objective,
        "sources": {
            **candle_sources, str(base.OUT): digest(base.OUT),
            str(Path(__file__).resolve()): digest(Path(__file__).resolve()),
        },
        "limitations": [
            "Spot volume is venue-specific and requires X-Perp transfer validation.",
            "All regime fields are coarse and selected before the hidden reveal.",
            "The 2026 archive is incomplete.",
            "Gross returns exclude fees at user request.",
            "No backtest guarantees a future monthly return.",
        ],
    }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True),
                   encoding="utf-8")
    print(json.dumps({
        "status": report["status"], "library_size": len(library),
        "enriched_rows": len(frame), "candidate_count": len(candidates),
        "selected": public(selected), "hidden": hidden,
        "objective_pass": objective,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
