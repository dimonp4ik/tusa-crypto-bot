"""Causal four-hour structure complement for the frozen OKX 1H cascade.

Four-hour bars are built only from completed one-hour candles.  Every setup
enters at the next one-hour open.  Models are fitted on 2020-2021, target and
entry thresholds are selected on 2022, then frozen for 2023 and later years.
"""
from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

import crypto_okx_swap_1h_regime_oct7_audit as base
import crypto_okx_swap_1h_target_cascade_oct7_audit as target_engine
from crypto_runtime_shadow_portfolio_oct3_audit import money, runtime_gate


ROOT = Path("reports/audit_2026_09_08")
CASCADE_ROWS = ROOT / "crypto_okx_swap_1h_target_cascade_oct7.pkl"
CASCADE_REPORT = ROOT / "crypto_okx_swap_1h_target_cascade_oct7.json"
OUT = ROOT / "crypto_okx_swap_4h_structure_complement_oct8.json"
ROWS_OUT = ROOT / "crypto_okx_swap_4h_structure_complement_oct8.pkl"
TARGETS = (.5, 1.0, 1.5, 2.0)
FAMILIES = (
    "trend_persistence",
    "trend_pullback_reclaim",
    "range_break",
    "channel_reclaim",
    "relative_rotation",
    "weekly_rejection",
)
FEATURES = (
    "ret1", "ret2", "ret3", "ret6", "ret12", "ret18", "ret42",
    "body", "range_pct", "atr14_pct", "volume_ratio", "range_ratio",
    "eff6", "eff18", "eff42", "ema3_dist", "ema12_dist", "ema42_dist",
    "upper_wick", "lower_wick", "breakout_up", "breakout_down",
    "position18", "lag_position18", "rv3", "rv6", "rv18", "rv42",
    "fast_slow", "autocorr18", "autocorr42",
    "cross_z1", "cross_z3", "cross_z6", "cross_z18",
    "breadth3", "breadth6", "breadth18", "dispersion6", "dispersion18",
    "btc_ret3", "btc_ret6", "btc_ret18", "btc_ret42",
)
CONFIGS = (
    {"leaves": 15, "leaf": 90, "l2": 4.0},
    {"leaves": 31, "leaf": 150, "l2": 7.0},
)
ENTRY_QUANTILES = (.60, .70, .78, .84, .89, .93, .96, .98)
UPGRADE_QUANTILES = (.70, .80, .88, .93, .97, .99)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def symbol_4h(symbol: str, candle: dict) -> pd.DataFrame:
    times = np.asarray(candle["time"], dtype=np.int64)
    hourly = pd.DataFrame({
        "time": pd.to_datetime(times, unit="s", utc=True),
        "open": np.asarray(candle["open"], dtype=float),
        "high": np.asarray(candle["high"], dtype=float),
        "low": np.asarray(candle["low"], dtype=float),
        "close": np.asarray(candle["close"], dtype=float),
        "volume_quote": np.asarray(candle["volume_quote"], dtype=float),
        "count": 1,
    }).set_index("time")
    bars = hourly.resample("4h", origin="epoch", label="left", closed="left").agg({
        "open": "first", "high": "max", "low": "min", "close": "last",
        "volume_quote": "sum", "count": "sum",
    })
    bars = bars[bars["count"].eq(4)].copy()
    close = bars.close
    log_return = np.log(close).diff()
    previous = close.shift(1)
    true_range = pd.concat(((bars.high - bars.low),
                            (bars.high - previous).abs(),
                            (bars.low - previous).abs()), axis=1).max(axis=1)
    candle_range = (bars.high - bars.low).replace(0, np.nan)
    upper = bars.high - np.maximum(bars.open, bars.close)
    lower = np.minimum(bars.open, bars.close) - bars.low
    prior_high = bars.high.shift(1).rolling(18, min_periods=14).max()
    prior_low = bars.low.shift(1).rolling(18, min_periods=14).min()
    channel = (prior_high - prior_low).replace(0, np.nan)
    position18 = (close - prior_low) / channel
    values: dict[str, object] = {
        "symbol": symbol,
        "entry_time": (bars.index.to_numpy(dtype="datetime64[s]").astype(np.int64)
                       + 4 * 3600),
        "body": (bars.close / bars.open - 1).to_numpy(float),
        "range_pct": (candle_range / bars.close).to_numpy(float),
        "atr14_pct": (true_range.rolling(14, min_periods=10).mean() / close).to_numpy(float),
        "volume_ratio": (bars.volume_quote / bars.volume_quote.shift(1).rolling(
            42, min_periods=30).median()).to_numpy(float),
        "range_ratio": (candle_range / candle_range.shift(1).rolling(
            42, min_periods=30).median()).to_numpy(float),
        "ema3_dist": (close / close.ewm(span=3, adjust=False).mean() - 1).to_numpy(float),
        "ema12_dist": (close / close.ewm(span=12, adjust=False).mean() - 1).to_numpy(float),
        "ema42_dist": (close / close.ewm(span=42, adjust=False).mean() - 1).to_numpy(float),
        "upper_wick": (upper / candle_range).to_numpy(float),
        "lower_wick": (lower / candle_range).to_numpy(float),
        "breakout_up": (close / prior_high - 1).to_numpy(float),
        "breakout_down": (prior_low / close - 1).to_numpy(float),
        "position18": position18.to_numpy(float),
        "lag_position18": position18.shift(1).to_numpy(float),
    }
    for window in (1, 2, 3, 6, 12, 18, 42):
        values[f"ret{window}"] = (close / close.shift(window) - 1).to_numpy(float)
    for window in (6, 18, 42):
        path = close.diff().abs().rolling(window, min_periods=max(4, int(.75 * window))).sum()
        values[f"eff{window}"] = ((close - close.shift(window)).abs() / path).to_numpy(float)
    for window in (3, 6, 18, 42):
        values[f"rv{window}"] = log_return.rolling(
            window, min_periods=max(3, int(.75 * window))).std(ddof=0).to_numpy(float)
    rv3 = log_return.rolling(3, min_periods=3).std(ddof=0)
    rv42 = log_return.rolling(42, min_periods=30).std(ddof=0)
    values["fast_slow"] = (rv3 / rv42.replace(0, np.nan)).to_numpy(float)
    values["autocorr18"] = log_return.rolling(18, min_periods=14).corr(
        log_return.shift(1)).to_numpy(float)
    values["autocorr42"] = log_return.rolling(42, min_periods=30).corr(
        log_return.shift(1)).to_numpy(float)
    frame = pd.DataFrame(values).replace([np.inf, -np.inf], np.nan)
    position = pd.Series(np.arange(len(times)), index=times)
    frame["entry_index"] = frame.entry_time.map(position)
    frame = frame.dropna().copy()
    stamp = pd.to_datetime(frame.entry_time, unit="s", utc=True)
    frame["year"] = stamp.dt.year.astype(int)
    frame["month"] = (stamp.dt.year * 100 + stamp.dt.month).astype(int)
    frame["entry_index"] = frame.entry_index.astype(np.int64)
    return frame


def build_features(candles: dict) -> pd.DataFrame:
    frame = pd.concat([symbol_4h(symbol, candle)
                       for symbol, candle in candles.items()], ignore_index=True)
    grouped = frame.groupby("entry_time", sort=False)
    for window in (1, 3, 6, 18):
        mean = grouped[f"ret{window}"].transform("mean")
        std = grouped[f"ret{window}"].transform("std").replace(0, np.nan)
        frame[f"cross_z{window}"] = (frame[f"ret{window}"] - mean) / std
    for window in (3, 6, 18):
        frame[f"breadth{window}"] = grouped[f"ret{window}"].transform(
            lambda values: (values > 0).mean())
    for window in (6, 18):
        frame[f"dispersion{window}"] = grouped[f"ret{window}"].transform("std")
    btc = frame[frame.symbol.eq("BTCUSDT")][
        ["entry_time", "ret3", "ret6", "ret18", "ret42"]].rename(columns={
            "ret3": "btc_ret3", "ret6": "btc_ret6",
            "ret18": "btc_ret18", "ret42": "btc_ret42"})
    return frame.merge(btc, on="entry_time", how="left",
                       validate="many_to_one").dropna()


def proposals(frame: pd.DataFrame) -> pd.DataFrame:
    pieces = []

    def add(name: str, direction: pd.Series, mask: pd.Series) -> None:
        selected = mask & direction.ne(0)
        part = frame.loc[selected].copy()
        part["direction"] = direction.loc[selected].astype(np.int8)
        part[f"family_{name}"] = 1
        pieces.append(part)

    direction = np.sign(frame.ret18)
    directed_position = frame.position18.where(direction > 0, 1 - frame.position18)
    add("trend_persistence", direction,
        frame.ret18.abs().ge(.018) & (frame.ret3 * direction).ge(.003)
        & (frame.ema12_dist * direction).gt(0) & frame.eff18.ge(.16)
        & directed_position.ge(.55))

    direction = np.sign(frame.ret42)
    add("trend_pullback_reclaim", direction,
        frame.ret42.abs().ge(.025) & (frame.ret2 * direction).between(-.025, .002)
        & (frame.body * direction).ge(.001) & (frame.ema42_dist * direction).gt(0))

    direction = np.sign(frame.body)
    distance = frame.breakout_up.where(direction > 0, frame.breakout_down)
    add("range_break", direction,
        frame.body.abs().ge(.0025) & distance.ge(-.001)
        & frame.range_ratio.ge(1.0) & frame.volume_ratio.ge(.75))

    direction = np.sign(frame.ret18)
    current = frame.position18.where(direction > 0, 1 - frame.position18)
    prior = frame.lag_position18.where(direction > 0, 1 - frame.lag_position18)
    add("channel_reclaim", direction,
        frame.ret18.abs().ge(.012) & prior.le(.5) & current.ge(.55)
        & (frame.body * direction).gt(0))

    direction = np.sign(frame.cross_z18)
    add("relative_rotation", direction,
        frame.cross_z18.abs().ge(1.0) & (frame.ret3 * direction).ge(.002)
        & frame.volume_ratio.ge(.65) & frame.dispersion18.ge(.008))

    direction = -np.sign(frame.ret42)
    rejection = frame.lower_wick.where(direction > 0, frame.upper_wick)
    add("weekly_rejection", direction,
        frame.ret42.abs().ge(.05) & frame.cross_z18.abs().ge(.8)
        & rejection.ge(.32) & (frame.body * direction).gt(0))

    combined = pd.concat(pieces, ignore_index=True)
    family_columns = [f"family_{name}" for name in FAMILIES]
    for column in family_columns:
        if column not in combined:
            combined[column] = 0
    keys = ["symbol", "entry_time", "direction"]
    flags = combined.groupby(keys, sort=False)[family_columns].max().reset_index()
    core = combined.sort_values(keys).drop_duplicates(keys).drop(columns=family_columns)
    return core.merge(flags, on=keys, how="left", validate="one_to_one")


def schema(frame: pd.DataFrame) -> list[str]:
    return [*FEATURES, *[f"family_{name}" for name in FAMILIES],
            *[f"symbol_{name}" for name in sorted(frame.symbol.unique())]]


def matrix(frame: pd.DataFrame, columns: list[str]) -> np.ndarray:
    output = pd.DataFrame(index=frame.index)
    direction = frame.direction.to_numpy(float)
    nondirectional = {
        "range_pct", "atr14_pct", "volume_ratio", "range_ratio",
        "eff6", "eff18", "eff42", "upper_wick", "lower_wick",
        "rv3", "rv6", "rv18", "rv42", "fast_slow",
        "autocorr18", "autocorr42", "dispersion6", "dispersion18",
    }
    for name in FEATURES:
        if name.startswith("breadth"):
            output[name] = np.where(direction > 0, frame[name], 1 - frame[name])
        elif name in ("position18", "lag_position18"):
            output[name] = np.where(direction > 0, frame[name], 1 - frame[name])
        elif name in nondirectional:
            output[name] = frame[name]
        elif name == "breakout_up":
            output[name] = np.where(direction > 0, frame.breakout_up, frame.breakout_down)
        elif name == "breakout_down":
            output[name] = np.where(direction > 0, frame.breakout_down, frame.breakout_up)
        else:
            output[name] = frame[name] * direction
    for family in FAMILIES:
        output[f"family_{family}"] = frame[f"family_{family}"].astype(float)
    for name in columns:
        if name.startswith("symbol_"):
            output[name] = (frame.symbol == name[7:]).astype(float)
    return output.reindex(columns=columns).replace(
        [np.inf, -np.inf], np.nan).fillna(0).to_numpy(np.float32)


def estimator(config: dict) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        max_iter=180, learning_rate=.06, max_leaf_nodes=config["leaves"],
        min_samples_leaf=config["leaf"], l2_regularization=config["l2"],
        random_state=20261008)


def route(frame: pd.DataFrame, entry_threshold: float,
          upgrade_thresholds: dict[float, float],
          allowed_families: tuple[str, ...]) -> pd.DataFrame:
    mask = frame.quality_0_5.ge(entry_threshold)
    mask &= frame[[f"family_{name}" for name in allowed_families]].any(axis=1)
    found = frame.loc[mask].copy()
    target = np.full(len(found), .5, dtype=float)
    for value in (1.0, 1.5, 2.0):
        suffix = str(value).replace(".", "_")
        target[found[f"quality_{suffix}"].to_numpy() >= upgrade_thresholds[value]] = value
    found["selected_target_pct"] = target
    found["signed_return"] = 0.0
    found["exit_time"] = 0
    found["target_hit"] = False
    found["exit_reason"] = 2
    for value in TARGETS:
        suffix = str(value).replace(".", "_")
        selected = found.selected_target_pct.eq(value)
        found.loc[selected, "signed_return"] = found.loc[selected, f"signed_{suffix}"]
        found.loc[selected, "exit_time"] = found.loc[selected, f"exit_{suffix}"]
        found.loc[selected, "target_hit"] = found.loc[selected, f"hit_{suffix}"]
        found.loc[selected, "exit_reason"] = found.loc[selected, f"reason_{suffix}"]
    found["quality"] = found.quality_0_5
    return found


def raw_rows(frame: pd.DataFrame, years: tuple[int, ...], source: str,
             priority: int) -> list[dict]:
    selected = frame[frame.year.isin(years)].sort_values(
        ["entry_time", "quality", "symbol"], ascending=[True, False, True])
    selected = selected.drop_duplicates(["symbol", "entry_time"], keep="first")
    rows = []
    for row in selected.to_dict("records"):
        rows.append({**row,
                     "direction": "LONG" if row["direction"] > 0 else "SHORT",
                     "source": source, "module": source,
                     "source_priority": priority, "module_priority": 0,
                     "outcome": "SL" if row["signed_return"] < 0 else "TP",
                     "signed_price_return": row["signed_return"]})
    return rows


def summarize(rows: list[dict]) -> dict:
    value = money(rows, .03)
    months = list(value.get("monthly_return_pct", {}).values())
    n = len(rows)
    distribution = {str(target): sum(
        float(row.get("selected_target_pct", 0)) == target for row in rows)
        for target in TARGETS}
    value.update({
        "literal_target_hits": sum(bool(row["target_hit"]) for row in rows),
        "literal_target_hit_rate": round(
            100 * sum(bool(row["target_hit"]) for row in rows) / n, 4) if n else 0.0,
        "stops": sum(int(row["exit_reason"]) == 0 for row in rows),
        "timeouts": sum(int(row["exit_reason"]) == 2 for row in rows),
        "months": len(months), "months_ge_30pct": sum(x >= 30 for x in months),
        "minimum_month_pct": round(min(months), 4) if months else 0.0,
        "median_month_pct": round(float(np.median(months)), 4) if months else 0.0,
        "target_distribution": distribution,
        "average_selected_target_pct": round(sum(
            float(row.get("selected_target_pct", 0)) for row in rows) / n, 4) if n else 0.0,
    })
    return value


def baseline_bundle(cascade: pd.DataFrame, years: tuple[int, ...]) -> dict:
    rows = raw_rows(cascade, years, "hourly_cascade", 0)
    accepted = runtime_gate(rows)
    return {"rows": rows, "summary": summarize(accepted)}


def evaluate_pair(cascade: pd.DataFrame, complement: pd.DataFrame,
                  years: tuple[int, ...], baseline: dict | None = None) -> dict:
    baseline = baseline or baseline_bundle(cascade, years)
    cascade_rows = baseline["rows"]
    extra_rows = raw_rows(complement, years, "four_hour_structure", 1)
    complement_accepted = runtime_gate(extra_rows)
    union_accepted = runtime_gate([*cascade_rows, *extra_rows])
    return {
        "cascade": baseline["summary"],
        "complement_alone": summarize(complement_accepted),
        "union": summarize(union_accepted),
        "accepted_complement_in_union": sum(
            row["source"] == "four_hour_structure" for row in union_accepted),
    }


def passes(value: dict, selection: bool) -> bool:
    baseline, union = value["cascade"], value["union"]
    return bool(value["accepted_complement_in_union"] >= (100 if selection else 60)
                and union["return_pct"] > baseline["return_pct"]
                and union["dd_pct"] <= baseline["dd_pct"] + 1.0
                and union["literal_target_hit_rate"] >= (65 if selection else 63)
                and union["trades_per_calendar_day"] >= 2.5)


def family_sets() -> list[tuple[str, ...]]:
    values = [tuple(FAMILIES)]
    values.extend((name,) for name in FAMILIES)
    values.extend(tuple(name for name in FAMILIES if name != omitted)
                  for omitted in FAMILIES)
    values.extend(tuple(pair) for pair in itertools.combinations(FAMILIES, 2))
    return values


def main() -> None:
    candles, sources = base.load()
    raw = build_features(candles)
    frame = proposals(raw).reset_index(drop=True)
    for target in TARGETS:
        target_engine.add_target(frame, candles, target)
    fit = frame.year.isin((2020, 2021)).to_numpy()
    columns = schema(frame.loc[fit])
    x = matrix(frame, columns)
    cascade = pd.read_pickle(CASCADE_ROWS)
    baselines = {(2022,): baseline_bundle(cascade, (2022,))}
    choices = []
    frontier = []
    prevalence = {}
    allowed_sets = family_sets()
    for config in CONFIGS:
        before_config = len(choices)
        scored = frame.copy()
        for target in TARGETS:
            suffix = str(target).replace(".", "_")
            labels = frame[f"hit_{suffix}"].astype(int).to_numpy()
            model = estimator(config)
            model.fit(x[fit], labels[fit])
            scored[f"quality_{suffix}"] = model.predict_proba(x)[:, 1]
            prevalence.setdefault(str(target), round(100 * float(labels[fit].mean()), 4))
        select = scored[scored.year.eq(2022)]
        for entry_quantile in ENTRY_QUANTILES:
            entry_threshold = float(select.quality_0_5.quantile(entry_quantile))
            for upgrade_quantile in UPGRADE_QUANTILES:
                upgrades = {target: float(select[
                    f"quality_{str(target).replace('.', '_')}"
                ].quantile(upgrade_quantile)) for target in (1.0, 1.5, 2.0)}
                routed_all = route(scored, entry_threshold, upgrades, tuple(FAMILIES))
                for allowed in allowed_sets:
                    if allowed == tuple(FAMILIES):
                        routed = routed_all
                    else:
                        routed = routed_all.loc[routed_all[
                            [f"family_{name}" for name in allowed]
                        ].any(axis=1)].copy()
                    value = evaluate_pair(
                        cascade, routed, (2022,), baselines[(2022,)])
                    if not passes(value, True):
                        continue
                    item = {
                        "model": config, "entry_quantile": entry_quantile,
                        "entry_threshold": entry_threshold,
                        "upgrade_quantile": upgrade_quantile,
                        "upgrade_thresholds": {str(k): v for k, v in upgrades.items()},
                        "allowed_families": list(allowed), "selection_2022": value,
                    }
                    gain = value["union"]["return_pct"] - value["cascade"]["return_pct"]
                    dd_cost = max(0.0, value["union"]["dd_pct"]
                                  - value["cascade"]["dd_pct"])
                    rank = gain - 2 * dd_cost
                    frontier.append({**item, "rank": rank})
                    choices.append((rank, item))
        print(json.dumps({"stage": "selection", "model": config,
                          "qualified": len(choices) - before_config}), flush=True)
    choices.sort(key=lambda item: item[0], reverse=True)
    baselines[(2023,)] = baseline_bundle(cascade, (2023,))
    validated = []
    validation_frontier = []
    for config in CONFIGS:
        before_config = len(validated)
        scored = frame.copy()
        for target in TARGETS:
            suffix = str(target).replace(".", "_")
            labels = frame[f"hit_{suffix}"].astype(int).to_numpy()
            model = estimator(config)
            model.fit(x[fit], labels[fit])
            scored[f"quality_{suffix}"] = model.predict_proba(x)[:, 1]
        candidates = [(rank, item) for rank, item in choices
                      if item["model"] == config]
        candidates.sort(key=lambda value: (
            value[1]["entry_threshold"], value[1]["upgrade_quantile"]))
        grouped = itertools.groupby(candidates, key=lambda value: (
            value[1]["entry_threshold"], value[1]["upgrade_quantile"]))
        for _, group in grouped:
            group_values = list(group)
            sample = group_values[0][1]
            upgrades = {float(key): value for key, value
                        in sample["upgrade_thresholds"].items()}
            routed_all = route(
                scored, sample["entry_threshold"], upgrades, tuple(FAMILIES))
            for selection_rank, item in group_values:
                allowed = tuple(item["allowed_families"])
                if allowed == tuple(FAMILIES):
                    routed = routed_all
                else:
                    routed = routed_all.loc[routed_all[
                        [f"family_{name}" for name in allowed]
                    ].any(axis=1)].copy()
                value = evaluate_pair(
                    cascade, routed, (2023,), baselines[(2023,)])
                if not passes(value, False):
                    continue
                gain22 = (item["selection_2022"]["union"]["return_pct"]
                          - item["selection_2022"]["cascade"]["return_pct"])
                gain23 = value["union"]["return_pct"] - value["cascade"]["return_pct"]
                dd_cost22 = max(
                    0.0, item["selection_2022"]["union"]["dd_pct"]
                    - item["selection_2022"]["cascade"]["dd_pct"])
                dd_cost23 = max(
                    0.0, value["union"]["dd_pct"] - value["cascade"]["dd_pct"])
                robust_rank = min(gain22, gain23) - 2 * max(dd_cost22, dd_cost23)
                diagnostic = {**item, "selection_rank": selection_rank,
                              "validation_2023": value,
                              "robust_rank": robust_rank}
                validation_frontier.append(diagnostic)
                validated.append((robust_rank, item, value))
        print(json.dumps({"stage": "validation", "model": config,
                          "qualified": len(validated) - before_config}), flush=True)
    validated.sort(key=lambda item: item[0], reverse=True)
    frozen = validated[0] if validated else None
    public_frozen = validation = test24 = later = None
    status = "REJECTED_2023" if choices else "REJECTED_SELECTION"
    if frozen:
        rank, item, validation = frozen
        scored = frame.copy()
        for target in TARGETS:
            suffix = str(target).replace(".", "_")
            labels = frame[f"hit_{suffix}"].astype(int).to_numpy()
            model = estimator(item["model"])
            model.fit(x[fit], labels[fit])
            scored[f"quality_{suffix}"] = model.predict_proba(x)[:, 1]
        routed = route(
            scored, item["entry_threshold"],
            {float(key): value for key, value in item["upgrade_thresholds"].items()},
            tuple(item["allowed_families"]),
        )
        public_frozen = {**item, "robust_rank": rank}
        baselines[(2024,)] = baseline_bundle(cascade, (2024,))
        test24 = evaluate_pair(cascade, routed, (2024,), baselines[(2024,)])
        if passes(test24, False):
            later = {}
            for year in (2025, 2026):
                baselines[(year,)] = baseline_bundle(cascade, (year,))
                later[str(year)] = evaluate_pair(
                    cascade, routed, (year,), baselines[(year,)])
            status = ("FOUR_HOUR_STRUCTURE_COMPLEMENT_CANDIDATE"
                      if all(passes(value, False) for value in later.values())
                      else "REJECTED_LATER")
            if status == "FOUR_HOUR_STRUCTURE_COMPLEMENT_CANDIDATE":
                routed.to_pickle(ROWS_OUT)
        else:
            status = "REJECTED_2024"
    if status != "FOUR_HOUR_STRUCTURE_COMPLEMENT_CANDIDATE" and ROWS_OUT.exists():
        ROWS_OUT.unlink()
    report = {
        "status": status,
        "protocol": {
            "venue": "OKX global USDT swap", "signal_bar": "closed 4H",
            "entry": "next 1H market open", "fit": [2020, 2021],
            "selection": 2022, "validation_and_ranking": 2023,
            "untouched_test": 2024,
            "hidden_if_test_passes": [2025, 2026], "targets_pct": list(TARGETS),
            "stop_pct": 1.0, "holds_hours": target_engine.HOLD,
            "money": "3% current equity margin, fixed 10x gross",
            "families": list(FAMILIES), "model_grid": list(CONFIGS),
        },
        "feature_rows": len(raw), "proposal_rows": len(frame),
        "fit_literal_prevalence_pct": prevalence,
        "selection_combinations": (len(CONFIGS) * len(ENTRY_QUANTILES)
                                   * len(UPGRADE_QUANTILES) * len(allowed_sets)),
        "qualified_selection": len(choices),
        "qualified_validation": len(validated),
        "frontier_top50": sorted(frontier, key=lambda item: item["rank"], reverse=True)[:50],
        "validation_frontier_top50": sorted(
            validation_frontier, key=lambda item: item["robust_rank"], reverse=True)[:50],
        "frozen_before_2024": public_frozen, "validation_2023": validation,
        "test_2024": test24, "later": later,
        "sources": {**sources, str(CASCADE_ROWS): digest(CASCADE_ROWS),
                    str(CASCADE_REPORT): digest(CASCADE_REPORT),
                    str(Path(__file__)): digest(Path(__file__))},
        "limitations": [
            "Four-hour OHLCV structure has no historical order-book depth.",
            "A historical pass still requires direct EU X-Perp shadow transfer.",
        ],
    }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({key: report[key] for key in (
        "status", "feature_rows", "proposal_rows", "fit_literal_prevalence_pct",
        "selection_combinations", "qualified_selection", "qualified_validation",
        "frozen_before_2024",
        "validation_2023", "test_2024", "later")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
