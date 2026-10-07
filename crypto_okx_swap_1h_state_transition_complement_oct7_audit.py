"""Independent state-transition complement for the frozen OKX 1H cascade."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

import crypto_okx_swap_1h_regime_oct7_audit as base
import crypto_5m_regime_router_literal_oct6_audit as evaluation
from crypto_runtime_shadow_portfolio_oct3_audit import money, runtime_gate


ROOT = Path("reports/audit_2026_09_08")
CASCADE_ROWS = ROOT / "crypto_okx_swap_1h_target_cascade_oct7.pkl"
CASCADE_REPORT = ROOT / "crypto_okx_swap_1h_target_cascade_oct7.json"
OUT = ROOT / "crypto_okx_swap_1h_state_transition_complement_oct7.json"
ROWS_OUT = ROOT / "crypto_okx_swap_1h_state_transition_complement_oct7.pkl"
FAMILIES = ("trend_acceleration", "breadth_turn", "dispersion_reversal",
            "volatility_release", "macro_reclaim", "btc_recovery")
LAG_FEATURES = ("lag_ret3", "lag_ret6", "lag_ret24", "lag_cross_z6",
                "lag_breadth24", "lag_fast_slow", "delta_ret3", "delta_ret6",
                "delta_ret24", "delta_breadth24", "delta_fast_slow")
QUANTILES = (.70, .75, .80, .85, .90, .925, .95, .97, .98, .985, .99)
CONFIGS = (
    {"leaves": 15, "leaf": 180, "l2": 3.0},
    {"leaves": 31, "leaf": 280, "l2": 5.0},
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def transitions(raw: pd.DataFrame) -> pd.DataFrame:
    frame = raw.sort_values(["symbol", "entry_time"]).copy()
    grouped = frame.groupby("symbol", sort=False)
    for name in ("ret3", "ret6", "ret24", "cross_z6", "breadth24", "fast_slow"):
        frame[f"lag_{name}"] = grouped[name].shift(1)
    for name in ("ret3", "ret6", "ret24", "breadth24", "fast_slow"):
        frame[f"delta_{name}"] = frame[name] - frame[f"lag_{name}"]
    frame = frame.dropna(subset=LAG_FEATURES).copy()
    pieces = []

    def add(name: str, direction: pd.Series, mask: pd.Series) -> None:
        selected = mask & direction.ne(0)
        part = frame.loc[selected].copy()
        part["direction"] = direction.loc[selected].astype(np.int8)
        part[f"family_{name}"] = 1
        pieces.append(part)

    direction = np.sign(frame.ret6)
    add("trend_acceleration", direction,
        frame.ret6.abs().ge(.003) & (frame.ret24 * direction).ge(.008)
        & (frame.delta_ret6 * direction).ge(.002) & frame.eff24.ge(.12))

    direction = np.sign(frame.breadth24 - .5)
    crossed = ((direction > 0) & frame.lag_breadth24.le(.5)
               | (direction < 0) & frame.lag_breadth24.ge(.5))
    add("breadth_turn", direction,
        crossed & (frame.ret3 * direction).gt(0)
        & frame.delta_breadth24.abs().ge(.125))

    direction = -np.sign(frame.lag_cross_z6)
    add("dispersion_reversal", direction,
        frame.lag_cross_z6.abs().ge(1.5)
        & frame.cross_z6.abs().lt(frame.lag_cross_z6.abs())
        & (frame.body1 * direction).gt(0))

    direction = np.sign(frame.ret3)
    add("volatility_release", direction,
        frame.lag_fast_slow.le(.8) & frame.fast_slow.ge(1.0)
        & frame.range_ratio.ge(1.0) & (frame.ret3 * direction).ge(.002))

    direction = np.sign(frame.ret168)
    add("macro_reclaim", direction,
        frame.ret168.abs().ge(.02) & (frame.lag_ret24 * direction).le(0)
        & (frame.ret24 * direction).ge(.002) & (frame.body1 * direction).gt(0))

    direction = np.sign(frame.btc_ret168)
    # lag_ret24 is symbol-specific while BTC recovery itself is represented by
    # the current BTC24 sign plus an asset reclaim. This remains fully causal.
    add("btc_recovery", direction,
        frame.btc_ret168.abs().ge(.02) & (frame.btc_ret24 * direction).ge(.002)
        & (frame.lag_ret24 * direction).le(0) & (frame.ret6 * direction).gt(0))

    combined = pd.concat(pieces, ignore_index=True)
    family_columns = [f"family_{name}" for name in FAMILIES]
    for column in family_columns:
        if column not in combined:
            combined[column] = 0
    keys = ["symbol", "entry_time", "direction"]
    flags = combined.groupby(keys, sort=False)[family_columns].max().reset_index()
    data = combined.sort_values(keys).drop_duplicates(keys).drop(columns=family_columns)
    return data.merge(flags, on=keys, how="left", validate="one_to_one")


def schema(frame: pd.DataFrame) -> list[str]:
    return [*base.FEATURES, *LAG_FEATURES,
            *[f"family_{name}" for name in FAMILIES],
            *[f"symbol_{name}" for name in sorted(frame.symbol.unique())]]


def matrix(frame: pd.DataFrame, columns: list[str]) -> np.ndarray:
    output = pd.DataFrame(index=frame.index)
    direction = frame.direction.to_numpy(float)
    nondirectional = {"range1", "atr14_pct", "volume_ratio", "range_ratio",
                      "eff24", "eff72", "upper_wick", "lower_wick",
                      "rv6", "rv24", "rv168", "fast_slow", "release3",
                      "autocorr24", "autocorr168", "lag_fast_slow",
                      "delta_fast_slow"}
    for name in (*base.FEATURES, *LAG_FEATURES):
        if name.startswith("breadth") or name == "lag_breadth24":
            output[name] = np.where(direction > 0, frame[name], 1-frame[name])
        elif name == "delta_breadth24":
            output[name] = frame[name] * direction
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


def raw_rows(frame: pd.DataFrame, years: tuple[int, ...], threshold: float,
             source: str, priority: int,
             allowed_families: tuple[str, ...] | None = None) -> list[dict]:
    mask = frame.year.isin(years) & frame.quality.ge(threshold)
    if allowed_families:
        mask &= frame[[f"family_{name}" for name in allowed_families]].any(axis=1)
    selected = frame[mask].copy()
    selected = selected.sort_values(
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
    value.update({
        "literal_target_hits": sum(bool(row["target_hit"]) for row in rows),
        "literal_target_hit_rate": round(
            100 * sum(bool(row["target_hit"]) for row in rows) / n, 4) if n else 0.0,
        "stops": sum(int(row["exit_reason"]) == 0 for row in rows),
        "timeouts": sum(int(row["exit_reason"]) == 2 for row in rows),
        "months": len(months), "months_ge_30pct": sum(x >= 30 for x in months),
        "minimum_month_pct": round(min(months), 4) if months else 0.0,
        "median_month_pct": round(float(np.median(months)), 4) if months else 0.0,
    })
    return value


def evaluate_pair(cascade: pd.DataFrame, complement: pd.DataFrame,
                  years: tuple[int, ...], threshold: float,
                  allowed_families: tuple[str, ...]) -> dict:
    cascade_rows = raw_rows(cascade, years, .5, "hourly_cascade", 0)
    extra_rows = raw_rows(complement, years, threshold, "state_transition", 1,
                          allowed_families)
    cascade_accepted = runtime_gate(cascade_rows)
    union_accepted = runtime_gate([*cascade_rows, *extra_rows])
    return {
        "cascade": summarize(cascade_accepted),
        "union": summarize(union_accepted),
        "accepted_complement": sum(row["source"] == "state_transition"
                                   for row in union_accepted),
    }


def selection_ok(value: dict) -> bool:
    base_value, union = value["cascade"], value["union"]
    return bool(value["accepted_complement"] >= 100
                and union["return_pct"] > base_value["return_pct"]
                and union["dd_pct"] <= base_value["dd_pct"] + 1.0
                and union["literal_target_hit_rate"] >= 65
                and union["trades_per_calendar_day"] >= 2.5)


def validation_ok(value: dict) -> bool:
    base_value, union = value["cascade"], value["union"]
    return bool(value["accepted_complement"] >= 60
                and union["return_pct"] > base_value["return_pct"]
                and union["dd_pct"] <= base_value["dd_pct"] + 1.0
                and union["literal_target_hit_rate"] >= 63
                and union["trades_per_calendar_day"] >= 2.3)


def main() -> None:
    candles, sources = base.load()
    raw = base.build_features(candles)
    candidate = base.add_outcomes(transitions(raw), candles).reset_index(drop=True)
    cascade = pd.read_pickle(CASCADE_ROWS)
    fit = candidate.year.isin((2020, 2021)).to_numpy()
    columns = schema(candidate.loc[fit])
    x = matrix(candidate, columns)
    labels = candidate.target_hit.astype(int).to_numpy()
    choices = []
    frontier = []
    family_subsets = [tuple(FAMILIES[index] for index in range(len(FAMILIES))
                            if mask & (1 << index))
                      for mask in range(1, 1 << len(FAMILIES))]
    for config in CONFIGS:
        model = HistGradientBoostingClassifier(
            max_iter=180, learning_rate=.06, max_leaf_nodes=config["leaves"],
            min_samples_leaf=config["leaf"], l2_regularization=config["l2"],
            random_state=20261007)
        model.fit(x[fit], labels[fit])
        scored = candidate.copy()
        scored["quality"] = model.predict_proba(x)[:, 1]
        selection_quality = scored.loc[scored.year.eq(2022), "quality"]
        for quantile in QUANTILES:
            threshold = float(selection_quality.quantile(quantile))
            for allowed_families in family_subsets:
                score = evaluate_pair(
                    cascade, scored, (2022,), threshold, allowed_families)
                item = {"model": config, "quantile": quantile,
                        "threshold": threshold,
                        "allowed_families": list(allowed_families),
                        "selection_2022": score}
                if selection_ok(score):
                    frontier.append(item)
                    rank = (score["union"]["return_pct"]
                            - score["cascade"]["return_pct"]
                            - 2 * max(0, score["union"]["dd_pct"]
                                    - score["cascade"]["dd_pct"]))
                    choices.append((rank, item, scored, model))
    choices.sort(key=lambda item: item[0], reverse=True)
    frozen = choices[0] if choices else None
    v23 = test24 = later = None
    status = "REJECTED_SELECTION"
    public_frozen = None
    if frozen:
        rank, item, scored, model = frozen
        public_frozen = {**item, "rank": rank}
        allowed = tuple(item["allowed_families"])
        v23 = evaluate_pair(cascade, scored, (2023,), item["threshold"], allowed)
        if validation_ok(v23):
            test24 = evaluate_pair(cascade, scored, (2024,), item["threshold"], allowed)
            if validation_ok(test24):
                later = {str(year): evaluate_pair(
                    cascade, scored, (year,), item["threshold"], allowed)
                    for year in (2025, 2026)}
                status = ("STATE_TRANSITION_COMPLEMENT_CANDIDATE"
                          if all(validation_ok(value) for value in later.values())
                          else "REJECTED_LATER")
                scored.to_pickle(ROWS_OUT)
            else:
                status = "REJECTED_2024"
        else:
            status = "REJECTED_2023"
    report = {
        "status": status,
        "protocol": {"fit": [2020, 2021], "selection": 2022,
                     "validation": 2023, "untouched_test": 2024,
                     "hidden_if_test_passes": [2025, 2026],
                     "entry": "next hourly market open", "target_pct": .5,
                     "stop_pct": 1.0, "hold_bars": 12,
                     "families": list(FAMILIES), "model_grid": list(CONFIGS),
                     "money": "3% current equity margin, fixed 10x gross"},
        "raw_rows": len(raw), "proposal_rows": len(candidate),
        "fit_literal_prevalence_pct": round(100 * float(labels[fit].mean()), 4),
        "qualified_selection": len(choices),
        "frontier": frontier, "frozen_before_2023": public_frozen,
        "validation_2023": v23, "test_2024": test24, "later": later,
        "sources": {**sources, str(CASCADE_ROWS): digest(CASCADE_ROWS),
                    str(CASCADE_REPORT): digest(CASCADE_REPORT),
                    str(Path(__file__)): digest(Path(__file__))},
    }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({key: report[key] for key in (
        "status", "raw_rows", "proposal_rows", "fit_literal_prevalence_pct",
        "qualified_selection", "frozen_before_2023", "validation_2023",
        "test_2024", "later")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
