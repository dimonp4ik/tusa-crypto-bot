"""Price fixed TP1 fractions on separate dated OKX order-book cohorts."""

from collections import Counter
from datetime import datetime, timezone
import math
from unittest.mock import patch

from okx_entry_screen_cash_nov21 import entry_features, screened_rows, summarize
from okx_entry_screen_holdout_nov21 import ENTRY_END as SPLIT
from okx_joint_exact_exit_nov21 import ENTRY_END as JOINT_END
from okx_unbankable_tp1_nov21 import old_inputs, read_json, recent_inputs
from okx_xperp_cash_tick_nov19 import CAPITAL, CLOSE_LAGS, FRACTION, LEVERAGE, TP1_SETTINGS, simulate
from okx_xperp_multiday_replay_nov20 import ROOT, file_hash
from stock_liquid_quote_routes_nov21 import write_json


FRACTIONS = {
    **TP1_SETTINGS,
    "requested15": (15., False),
    "requested25": (25., False),
    "requested34": (34., False),
    "requested40": (40., False),
    "requested50": (50., False),
    "quarter_minimum": (25., True),
    "full_tp1": (100., True),
}


def detailed(result):
    summary = summarize(result)
    if "closed" not in result:
        return summary
    trades = [dict(t, code="FULL_TP1") if t["bank_qty"] == t["qty"] else t for t in result["closed"]]
    summary["exit_counts"] = Counter(t["code"] for t in trades)
    return dict(summary, cash_pnl_usd=math.fsum(t["bank_pnl"] + t["runner_pnl"] for t in trades),
                by_bot={bot: dict(
                    trades=len(subset),
                    profitable=sum(t["bank_pnl"] + t["runner_pnl"] > 0 for t in subset),
                    tp1_executions=sum(t["bank_qty"] > 0 for t in subset),
                    tp1_full_exits=sum(t["bank_qty"] == t["qty"] for t in subset),
                    cash_pnl_usd=math.fsum(t["bank_pnl"] + t["runner_pnl"] for t in subset),
                    exit_counts=Counter(t["code"] for t in subset),
                ) for bot in ("crypto", "stocks") if (subset := [t for t in trades if t["bot"] == bot])})


def run_cohort(rows, specs, ticks, books, cap):
    if any(r["entry_status"] == "unknown_quote" for r in rows):
        raise ValueError("study cohort has an unknown entry quote")
    features = {(r["symbol"], r["entry_ms"]): entry_features(r, books[r["symbol"]][r["entry_ms"]])
                for r in rows if r["bot"] == "crypto" and r["entry_status"] == "eligible"}
    policies = dict(baseline=rows, one_same_side_per_hour=list(
        screened_rows(rows, features, same_side_per_hour=1)))
    with patch.dict(TP1_SETTINGS, FRACTIONS):
        return {policy: {str(lag): {name: detailed(simulate(
            candidates, specs, ticks, books, clock=60, lag=lag, setting=name, max_positions=cap))
            for name in FRACTIONS} for lag in CLOSE_LAGS} for policy, candidates in policies.items()}


def verify_control(rows, specs, ticks, books, cap, path, *, joint):
    settings = ("requested5", "half_minimum") if joint else ("requested5",)
    reference = read_json(path)
    for setting in settings:
        actual = simulate(rows, specs, ticks, books, clock=60, lag=200, setting=setting, max_positions=cap)
        if joint:
            expected = reference["scenarios"][f"both_base_cadence|cap6|{setting}|200ms"]["book"]["closed"]
            compare = actual["closed"]
        else:
            expected = reference["continuous_cash_results"]["results"]["baseline"]
            compare = summarize(actual)
        if compare != expected:
            raise ValueError(f"published cash control changed: {setting} in {path.name}")


def main():
    early_control = ROOT / "okx_entry_screen_holdout_nov21.json"
    joint_control = ROOT / "okx_joint_exact_exit_nov21.json"
    rows, specs, ticks, books, *_ = old_inputs()
    verify_control(rows, specs, ticks, books, 12, early_control, joint=False)
    early = [r for r in rows if r["epoch"] < SPLIT]
    later = [r for r in rows if r["epoch"] >= SPLIT]
    print("Earlier controls verified", flush=True)
    cases = {
        "nov05_08": run_cohort(early, specs, ticks, books, 12),
        "nov09_14": run_cohort(later, specs, ticks, books, 12),
    }
    joint, specs, ticks, books, *_ = recent_inputs()
    verify_control(joint, specs, ticks, books, 6, joint_control, joint=True)
    print("Shared-account controls verified", flush=True)
    cases["nov17_19_joint"] = run_cohort(joint, specs, ticks, books, 6)
    span = {label: dict(first=min(r["epoch"] for r in cohort), stop=end, proposals=len(cohort))
            for label, cohort, end in (("nov05_08", early, SPLIT), ("nov09_14", later,
            int(datetime(2026, 11, 15, tzinfo=timezone.utc).timestamp())),
            ("nov17_19_joint", joint, JOINT_END))}
    report = dict(capital_usd=CAPITAL, margin_fraction=FRACTION, leverage=LEVERAGE, windows=span,
                  settings=FRACTIONS, source_hashes={p.name: file_hash(p) for p in (early_control, joint_control)},
                  control_parity=True, live_activation_supported=False, cases=cases, limitations=[
                      "Cohorts each start with $130; results are neither continuous nor monthly.",
                      "Only synthesized market execution against verified public books; no account orders.",
                      "TP1 is still 0.5R and TP2 is still 2R on crypto; the study changes quantity only.",
                      "The stock continuation route in the joint replay is hypothetical and separately PAPER gated live.",
                      "Drawdown is measured on the settled cash ledger, without unrealized loss.",
                      "These already studied dates can falsify a rule; a winning row needs prospective validation.",
                  ])
    write_json(ROOT / "okx_tp1_cash_sensitivity_nov22.json", report)
    for label, policies in cases.items():
        for policy, lags in policies.items():
            for lag, settings in lags.items():
                print(label, policy, lag, {name: round(out["cash_pnl_usd"], 3)
                                            if "cash_pnl_usd" in out else out["reason"]
                                            for name, out in settings.items()}, flush=True)


if __name__ == "__main__":
    main()
