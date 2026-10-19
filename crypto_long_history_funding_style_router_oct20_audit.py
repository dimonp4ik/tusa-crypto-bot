"""Causal funding-conditioned router over literal price-action styles.

The price-rule library was fixed on the separate Binance spot development
sample.  This audit transfers those rules to Binance UM perpetual candles,
uses only the last funding print strictly before entry, and requires each
funding/style expert to be positive in every development and confirmation
year before the 2025-2026 holdout is revealed.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

import crypto_long_history_hourly_cross_section_oct20_audit as base
import crypto_long_history_monthly_style_router_oct20_audit as monthly
import crypto_raw_hourly_direction_oct19_audit as portfolio


HISTORY = base.ROOT / "crypto_futures_5m_funding_history"
BASE_REPORT = base.OUT
OUT = base.ROOT / "crypto_long_history_funding_style_router_oct20.json"
ROWS_OUT = base.ROOT / "crypto_long_history_funding_style_router_oct20.pkl"
FIELD_SETS = (
    ("variant", "direction", "own_crowding"),
    ("variant", "direction", "relative_crowding"),
    ("variant", "direction", "funding_sign"),
    ("variant", "direction", "funding_trend"),
    ("variant", "direction", "own_crowding", "relative_crowding"),
    ("style", "lookback_hours", "target_pct", "direction",
     "own_crowding", "funding_trend"),
)
MIN_COUNTS = (20, 50)
MIN_MEANS = (0.0, .00025)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load() -> tuple[dict[str, dict], dict[str, dict], dict[str, str]]:
    manifest_path = HISTORY / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    candles, funding, sources = {}, {}, {str(manifest_path): digest(manifest_path)}
    for symbol in base.SYMBOLS:
        item = manifest["symbols"][symbol]
        candle_path = Path(item["candles"]["path"])
        funding_path = Path(item["funding"]["path"])
        candle_raw, funding_raw = candle_path.read_bytes(), funding_path.read_bytes()
        if hashlib.sha256(candle_raw).hexdigest() != item["candles"]["sha256"]:
            raise ValueError(f"candle hash mismatch: {symbol}")
        if hashlib.sha256(funding_raw).hexdigest() != item["funding"]["sha256"]:
            raise ValueError(f"funding hash mismatch: {symbol}")
        candles[symbol] = pickle.loads(candle_raw)
        funding[symbol] = pickle.loads(funding_raw)
        sources[str(candle_path)] = item["candles"]["sha256"]
        sources[str(funding_path)] = item["funding"]["sha256"]
    return candles, funding, sources


def funding_features(raw: dict) -> pd.DataFrame:
    frame = pd.DataFrame({
        "funding_time": np.asarray(raw["time"], dtype=np.int64),
        "funding_rate": np.asarray(raw["rate"], dtype=float),
    }).sort_values("funding_time")
    frame["funding_mean3"] = frame.funding_rate.rolling(3, min_periods=3).mean()
    frame["funding_mean9"] = frame.funding_rate.rolling(9, min_periods=6).mean()
    mean90 = frame.funding_rate.rolling(90, min_periods=60).mean()
    std90 = frame.funding_rate.rolling(90, min_periods=60).std(ddof=0).replace(0, np.nan)
    frame["funding_z90"] = (frame.funding_rate - mean90) / std90
    return frame.dropna()


def attach_funding(frame: pd.DataFrame,
                   funding: dict[str, dict]) -> pd.DataFrame:
    """Attach the last funding publication strictly before market entry."""
    pieces = []
    for symbol, part in frame.groupby("symbol", sort=False):
        left = part.sort_values("entry_time").copy()
        left["funding_join_time"] = left.entry_time.astype(np.int64) - 1
        right = funding_features(funding[symbol])
        found = pd.merge_asof(
            left, right, left_on="funding_join_time", right_on="funding_time",
            direction="backward", allow_exact_matches=True)
        pieces.append(found)
    result = pd.concat(pieces, ignore_index=True).dropna(
        subset=["funding_rate", "funding_z90", "funding_mean3", "funding_mean9"])
    if not (result.funding_time < result.entry_time).all():
        raise AssertionError("future funding publication joined to an entry")
    grouped = result.groupby("entry_time", sort=False).funding_rate
    cross_mean = grouped.transform("mean")
    cross_std = grouped.transform("std").replace(0, np.nan)
    result["funding_cross_z"] = (result.funding_rate - cross_mean) / cross_std
    direction = np.where(result.direction.eq("LONG"), 1.0, -1.0)
    directed_own = direction * result.funding_z90
    directed_cross = direction * result.funding_cross_z
    directed_rate = direction * result.funding_rate
    directed_trend = direction * (result.funding_mean3 - result.funding_mean9)
    result["own_crowding"] = pd.cut(
        directed_own, [-np.inf, -.75, .75, np.inf],
        labels=["countercrowded", "neutral", "crowded"]).astype(str)
    result["relative_crowding"] = pd.cut(
        directed_cross, [-np.inf, -.75, .75, np.inf],
        labels=["countercrowded", "neutral", "crowded"]).astype(str)
    result["funding_sign"] = np.where(
        directed_rate < 0, "countercrowded", "crowded")
    result["funding_trend"] = np.where(
        directed_trend < 0, "de_crowding", "crowding")
    result["year"] = pd.to_datetime(
        result.entry_time, unit="s", utc=True).dt.year
    return result.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["funding_cross_z"])


def build_library(candles: dict[str, dict], items: list[dict]) -> pd.DataFrame:
    features, labeled, pieces = {}, {}, []
    for item in items:
        feature_key = (int(item["lookback_hours"]), str(item["style"]))
        if feature_key not in features:
            features[feature_key] = base.feature_rows(
                candles, feature_key[0], feature_key[1])
        geometry = (float(item["target_pct"]), float(item["stop_pct"]),
                    int(item["hold_bars_5m"]))
        label_key = (*feature_key, geometry)
        if label_key not in labeled:
            labeled[label_key] = base.label(
                features[feature_key], candles, geometry)
        chosen = base.entries(
            labeled[label_key], float(item["z_threshold"]),
            float(item["minimum_move_pct"]) / 100).copy()
        chosen["variant"] = monthly.variant_id(item)
        chosen["style"] = str(item["style"])
        chosen["lookback_hours"] = int(item["lookback_hours"])
        chosen["signed_price_return"] = chosen.signed_return
        pieces.append(chosen)
    return pd.concat(pieces, ignore_index=True)


def stable_experts(frame: pd.DataFrame, fields: tuple[str, ...],
                   years: tuple[int, ...], min_count: int,
                   min_mean: float) -> pd.DataFrame:
    stats = frame[frame.year.isin(years)].groupby(
        ["year", *fields], dropna=False).signed_price_return.agg(
            count="size", mean_return="mean",
            win_rate=lambda values: float((values > 0).mean())).reset_index()
    good = stats[stats["count"].ge(min_count) & stats.mean_return.ge(min_mean)]
    if good.empty:
        return pd.DataFrame()
    stable = good.groupby(list(fields), dropna=False).filter(
        lambda part: set(part.year.astype(int)) == set(years))
    if stable.empty:
        return pd.DataFrame()
    return stable.groupby(list(fields), dropna=False).agg(
        worst_year_mean=("mean_return", "min"),
        mean_return=("mean_return", "mean"),
        worst_year_win_rate=("win_rate", "min"),
        observations=("count", "sum")).reset_index()


def confirmed_experts(frame: pd.DataFrame, fields: tuple[str, ...],
                      development: pd.DataFrame, min_count: int,
                      min_mean: float) -> pd.DataFrame:
    if development.empty:
        return development.copy()
    eligible = frame.merge(
        development[list(fields)], on=list(fields), how="inner",
        validate="many_to_many")
    confirmed = stable_experts(
        eligible, fields, base.CONFIRMATION_YEARS, min_count, min_mean)
    if confirmed.empty:
        return development.iloc[:0].copy()
    keys = confirmed[list(fields)].drop_duplicates()
    output = development.merge(
        keys, on=list(fields), how="inner", validate="one_to_one")
    confirmation = confirmed[[*fields, "worst_year_mean", "observations"]].rename(
        columns={"worst_year_mean": "confirmation_worst_mean",
                 "observations": "confirmation_observations"})
    return output.merge(
        confirmation, on=list(fields), how="inner", validate="one_to_one")


def route(frame: pd.DataFrame, fields: tuple[str, ...],
          experts: pd.DataFrame, loss_pause: int,
          hourly_cap: int) -> list[dict]:
    if experts.empty:
        return []
    score_columns = [*fields, "worst_year_mean", "mean_return",
                     "confirmation_worst_mean"]
    selected = frame.merge(
        experts[score_columns], on=list(fields), how="inner",
        validate="many_to_one")
    selected = selected.sort_values(
        ["entry_time", "confirmation_worst_mean", "worst_year_mean",
         "symbol", "variant"],
        ascending=[True, False, False, True, True], kind="mergesort")
    selected = selected.drop_duplicates(["symbol", "entry_time"], keep="first")
    rows = portfolio.candidate_rows(selected, float("-inf"))
    for row in rows:
        row["model_probability"] = float(row["confirmation_worst_mean"])
        row["module"] = f"funding_style:{row['variant']}"
        row["source"] = "long_history_funding_style"
    return portfolio.rails.ranked_runtime_gate(
        rows, loss_pause=loss_pause, hourly_cap=hourly_cap)


def stage(rows: list[dict], years: tuple[int, ...], margin: float) -> dict:
    selected = [row for row in rows if pd.Timestamp(
        float(row["entry_time"]), unit="s", tz="UTC").year in years]
    return portfolio.money(selected, margin)


def rank(item: dict) -> tuple:
    value = item["confirmation_2023_2024_3pct"]
    return (
        value["dd_pct"] <= 15 and value["trades_per_calendar_day"] >= 3,
        -int(value["negative_months"]), base.month_floor(value),
        float(value["return_pct"]), -float(value["dd_pct"]),
        int(item["confirmed_expert_count"]),
    )


def public(item: dict) -> dict:
    return {key: value for key, value in item.items()
            if not key.startswith("_")}


def main() -> None:
    source = json.loads(BASE_REPORT.read_text(encoding="utf-8"))
    items = source["development_diverse_library"]
    candles, funding, sources = load()
    library = build_library(candles, items)
    frame = attach_funding(library, funding)
    candidates = []
    grid = itertools.product(
        FIELD_SETS, MIN_COUNTS, MIN_MEANS, MIN_COUNTS, MIN_MEANS,
        base.LOSS_PAUSES, base.HOURLY_CAPS)
    for (fields, dev_count, dev_mean, confirm_count, confirm_mean,
         loss_pause, hourly_cap) in grid:
        development = stable_experts(
            frame, fields, base.DEVELOPMENT_YEARS, dev_count, dev_mean)
        experts = confirmed_experts(
            frame, fields, development, confirm_count, confirm_mean)
        rows = route(frame, fields, experts, loss_pause, hourly_cap)
        candidates.append({
            "fields": list(fields),
            "development_minimum_count_each_year": dev_count,
            "development_minimum_mean_each_year_pct": 100 * dev_mean,
            "confirmation_minimum_count_each_year": confirm_count,
            "confirmation_minimum_mean_each_year_pct": 100 * confirm_mean,
            "development_expert_count": len(development),
            "confirmed_expert_count": len(experts),
            "loss_pause": loss_pause, "hourly_cap": hourly_cap,
            "development_2020_2022_3pct": stage(
                rows, base.DEVELOPMENT_YEARS, .03),
            "confirmation_2023_2024_3pct": stage(
                rows, base.CONFIRMATION_YEARS, .03),
            "experts": experts.to_dict("records"), "_rows": rows,
        })
    candidates.sort(key=rank, reverse=True)
    selected = candidates[0]
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
        "status": ("FUNDING_STYLE_SHADOW_CANDIDATE"
                   if objective else "REJECTED_OBJECTIVE"),
        "ready_for_live_orders": False,
        "protocol": {
            "price_library": "fixed on separate spot 2020-2022 development report",
            "venue": "Binance UM perpetual candles and funding",
            "funding_join": "last published funding timestamp strictly before entry",
            "development_experts": "positive separately in 2020, 2021, 2022",
            "expert_confirmation": "positive separately in 2023 and 2024",
            "hidden_reveal": "2025-01 through available 2026 archive",
            "entry": "next five-minute market open", "orders": "market only",
            "collision": "stop first",
            "money": "client 3% or 5% current-equity margin; fixed 10x",
        },
        "library_size": len(items), "enriched_rows": len(frame),
        "candidate_count": len(candidates),
        "selected_before_hidden_reveal": public(selected),
        "confirmation_frontier": [public(item) for item in candidates[:20]],
        "hidden_2025_2026": hidden, "objective_pass": objective,
        "sources": {
            **sources, str(BASE_REPORT): digest(BASE_REPORT),
            str(Path(__file__).resolve()): digest(Path(__file__).resolve()),
        },
        "limitations": [
            "The price library and funding buckets remain a finite searched family.",
            "The 2026 archive ends in September.",
            "A survivor still requires direct X-Perp forward shadow evidence.",
            "Gross returns exclude fees at user request.",
            "No backtest guarantees a future monthly return.",
        ],
    }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({
        "status": report["status"], "library_size": len(items),
        "enriched_rows": len(frame), "candidate_count": len(candidates),
        "selected": public(selected), "hidden": hidden,
        "objective_pass": objective,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
