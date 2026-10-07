"""Annual causal walk-forward adaptation of the four-hour complement.

For evaluation year Y, target models train only on years before Y-1.  Entry,
target-upgrade and family gates are selected on Y-1, then frozen for Y.  The
2025-2026 folds are revealed only if the 2023-2024 development folds pass.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

import crypto_okx_swap_1h_regime_oct7_audit as base
import crypto_okx_swap_1h_target_cascade_oct7_audit as target_engine
import crypto_okx_swap_4h_structure_complement_oct8_audit as four
from crypto_runtime_shadow_portfolio_oct3_audit import runtime_gate


ROOT = Path("reports/audit_2026_09_08")
CASCADE_ROWS = ROOT / "crypto_okx_swap_1h_target_cascade_oct7.pkl"
CASCADE_REPORT = ROOT / "crypto_okx_swap_1h_target_cascade_oct7.json"
STATIC_REPORT = ROOT / "crypto_okx_swap_4h_structure_complement_oct8.json"
OUT = ROOT / "crypto_okx_swap_4h_walkforward_complement_oct8.json"
ROWS_OUT = ROOT / "crypto_okx_swap_4h_walkforward_complement_oct8.pkl"
MODEL_CONFIG = {"leaves": 15, "leaf": 90, "l2": 4.0}
FAMILY_OPTIONS = (tuple(four.FAMILIES),
                  *((name,) for name in four.FAMILIES))
DEVELOPMENT_FOLDS = ((2022, 2023), (2023, 2024))
HIDDEN_FOLDS = ((2024, 2025), (2025, 2026))


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fit_scores(frame: pd.DataFrame, x: np.ndarray,
               train_years: tuple[int, ...]) -> pd.DataFrame:
    train = frame.year.isin(train_years).to_numpy()
    if train.sum() < 1000:
        raise ValueError(f"insufficient walk-forward training rows: {train.sum()}")
    scored = frame.copy()
    for target in four.TARGETS:
        suffix = str(target).replace(".", "_")
        labels = frame[f"hit_{suffix}"].astype(int).to_numpy()
        model = four.estimator(MODEL_CONFIG)
        model.fit(x[train], labels[train])
        scored[f"quality_{suffix}"] = model.predict_proba(x)[:, 1]
    return scored


def selection_passes(value: dict) -> bool:
    baseline = value["cascade"]
    complement = value["complement_alone"]
    union = value["union"]
    return bool(value["accepted_complement_in_union"] >= 100
                and complement["return_pct"] > 0
                and complement["literal_target_hit_rate"] >= 58
                and union["return_pct"] > baseline["return_pct"]
                and union["dd_pct"] <= baseline["dd_pct"] + 1.0
                and union["literal_target_hit_rate"] >= 63
                and union["trades_per_calendar_day"] >= 2.5)


def evaluation_passes(value: dict) -> bool:
    baseline, union = value["cascade"], value["union"]
    return bool(value["accepted_complement_in_union"] >= 60
                and union["return_pct"] > baseline["return_pct"]
                and union["dd_pct"] <= baseline["dd_pct"] + 1.0
                and union["literal_target_hit_rate"] >= 63
                and union["trades_per_calendar_day"] >= 2.5)


def choose_fold(frame: pd.DataFrame, x: np.ndarray, cascade: pd.DataFrame,
                selection_year: int, evaluation_year: int) -> tuple[dict, pd.DataFrame]:
    train_years = tuple(range(2020, selection_year))
    scored = fit_scores(frame, x, train_years)
    select = scored[scored.year.eq(selection_year)]
    baseline = four.baseline_bundle(cascade, (selection_year,))
    choices = []
    tested = 0
    for entry_quantile in four.ENTRY_QUANTILES:
        entry_threshold = float(select.quality_0_5.quantile(entry_quantile))
        for upgrade_quantile in four.UPGRADE_QUANTILES:
            upgrades = {target: float(select[
                f"quality_{str(target).replace('.', '_')}"
            ].quantile(upgrade_quantile)) for target in (1.0, 1.5, 2.0)}
            routed_all = four.route(
                scored, entry_threshold, upgrades, tuple(four.FAMILIES))
            for allowed in FAMILY_OPTIONS:
                tested += 1
                if allowed == tuple(four.FAMILIES):
                    routed = routed_all
                else:
                    routed = routed_all.loc[routed_all[
                        [f"family_{name}" for name in allowed]
                    ].any(axis=1)].copy()
                value = four.evaluate_pair(
                    cascade, routed, (selection_year,), baseline)
                if not selection_passes(value):
                    continue
                gain = value["union"]["return_pct"] - value["cascade"]["return_pct"]
                dd_cost = max(0.0, value["union"]["dd_pct"]
                              - value["cascade"]["dd_pct"])
                rank = gain - 2 * dd_cost
                item = {
                    "train_years": list(train_years),
                    "selection_year": selection_year,
                    "evaluation_year": evaluation_year,
                    "entry_quantile": entry_quantile,
                    "entry_threshold": entry_threshold,
                    "upgrade_quantile": upgrade_quantile,
                    "upgrade_thresholds": {str(k): v for k, v in upgrades.items()},
                    "allowed_families": list(allowed),
                    "selection_result": value, "rank": rank,
                }
                choices.append((rank, item))
    choices.sort(key=lambda value: value[0], reverse=True)
    if not choices:
        return ({"status": "DISABLED_NO_SELECTION", "tested": tested,
                 "qualified": 0, "selection_year": selection_year,
                 "evaluation_year": evaluation_year}, frame.iloc[0:0].copy())
    _, chosen = choices[0]
    routed = four.route(
        scored, chosen["entry_threshold"],
        {float(key): value for key, value in chosen["upgrade_thresholds"].items()},
        tuple(chosen["allowed_families"]),
    )
    eval_baseline = four.baseline_bundle(cascade, (evaluation_year,))
    evaluation = four.evaluate_pair(
        cascade, routed, (evaluation_year,), eval_baseline)
    chosen.update({
        "status": "PASS" if evaluation_passes(evaluation) else "FAIL",
        "tested": tested, "qualified": len(choices),
        "evaluation_result": evaluation,
    })
    return chosen, routed


def aggregate(cascade: pd.DataFrame, routed_folds: list[tuple[int, pd.DataFrame]]) -> dict:
    years = tuple(year for year, _ in routed_folds)
    cascade_rows = four.raw_rows(cascade, years, "hourly_cascade", 0)
    extra_rows = []
    for year, routed in routed_folds:
        if routed.empty:
            continue
        extra_rows.extend(four.raw_rows(
            routed, (year,), "four_hour_walkforward", 1))
    cascade_accepted = runtime_gate(cascade_rows)
    union_accepted = runtime_gate([*cascade_rows, *extra_rows])
    return {
        "cascade": four.summarize(cascade_accepted),
        "union": four.summarize(union_accepted),
        "accepted_complement_in_union": sum(
            row["source"] == "four_hour_walkforward" for row in union_accepted),
    }


def main() -> None:
    candles, sources = base.load()
    raw = four.build_features(candles)
    frame = four.proposals(raw).reset_index(drop=True)
    for target in four.TARGETS:
        target_engine.add_target(frame, candles, target)
    columns = four.schema(frame[frame.year.isin((2020, 2021))])
    x = four.matrix(frame, columns)
    cascade = pd.read_pickle(CASCADE_ROWS)
    folds = []
    routed_folds = []
    for selection_year, evaluation_year in DEVELOPMENT_FOLDS:
        item, routed = choose_fold(
            frame, x, cascade, selection_year, evaluation_year)
        folds.append(item)
        routed_folds.append((evaluation_year, routed))
        print(json.dumps({"stage": "development_fold",
                          "selection_year": selection_year,
                          "evaluation_year": evaluation_year,
                          "status": item["status"],
                          "qualified": item.get("qualified", 0)}), flush=True)
    development_pass = all(item["status"] == "PASS" for item in folds)
    hidden_revealed = False
    if development_pass:
        hidden_revealed = True
        for selection_year, evaluation_year in HIDDEN_FOLDS:
            item, routed = choose_fold(
                frame, x, cascade, selection_year, evaluation_year)
            folds.append(item)
            routed_folds.append((evaluation_year, routed))
            print(json.dumps({"stage": "hidden_fold",
                              "selection_year": selection_year,
                              "evaluation_year": evaluation_year,
                              "status": item["status"],
                              "qualified": item.get("qualified", 0)}), flush=True)
            if item["status"] != "PASS":
                break
    aggregate_value = aggregate(cascade, routed_folds)
    all_pass = development_pass and hidden_revealed \
        and len(folds) == len(DEVELOPMENT_FOLDS) + len(HIDDEN_FOLDS) \
        and all(item["status"] == "PASS" for item in folds)
    status = ("FOUR_HOUR_WALKFORWARD_COMPLEMENT_CANDIDATE"
              if all_pass else "REJECTED_DEVELOPMENT"
              if not development_pass else "REJECTED_HIDDEN")
    if all_pass:
        chosen = pd.concat([routed[routed.year.eq(year)]
                            for year, routed in routed_folds], ignore_index=True)
        chosen.to_pickle(ROWS_OUT)
    elif ROWS_OUT.exists():
        ROWS_OUT.unlink()
    report = {
        "status": status,
        "protocol": {
            "method": "annual expanding walk-forward",
            "rule": "for evaluation Y, train through Y-2 and select gates on Y-1",
            "development_folds": [list(value) for value in DEVELOPMENT_FOLDS],
            "hidden_folds": [list(value) for value in HIDDEN_FOLDS],
            "hidden_revealed": hidden_revealed,
            "model": MODEL_CONFIG, "family_options": [list(x) for x in FAMILY_OPTIONS],
            "targets_pct": list(four.TARGETS), "stop_pct": 1.0,
            "holds_hours": target_engine.HOLD,
            "money": "3% current equity margin, fixed 10x gross",
        },
        "feature_rows": len(raw), "proposal_rows": len(frame),
        "folds": folds, "aggregate_evaluated_folds": aggregate_value,
        "sources": {**sources, str(CASCADE_ROWS): digest(CASCADE_ROWS),
                    str(CASCADE_REPORT): digest(CASCADE_REPORT),
                    str(STATIC_REPORT): digest(STATIC_REPORT),
                    str(Path(__file__)): digest(Path(__file__))},
        "limitations": [
            "The method is evaluated after prior static-model research and therefore remains research-biased.",
            "A full historical pass would still require direct EU X-Perp shadow transfer.",
        ],
    }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({key: report[key] for key in (
        "status", "feature_rows", "proposal_rows", "folds",
        "aggregate_evaluated_folds")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
