import json
from pathlib import Path

import pandas as pd

import crypto_profit_first_fred_fraction_router_nov1_audit as audit


def test_full_tp1_bank_uses_near_exit_and_half_r():
    frame = pd.DataFrame([{
        "a0p5_l0_runner_r": 2.0,
        "a0p5_l0_banked": True,
        "a0p5_l0_exit_time": 300.0,
        "a0p5_l0_label": "TP_RUNNER",
        "near_exit_time": 200.0,
        "risk_pct": .01,
        "macro_risk_on_score": 0,
    }])
    result = audit.apply_policy(frame, lambda x: pd.Series(True, index=x.index), 1.0)
    assert result.iloc[0].net_r == .5
    assert result.iloc[0].exit_time == 200.0
    assert result.iloc[0].signed_price_return == .005


def test_stop_before_tp1_is_unchanged_by_fraction():
    frame = pd.DataFrame([{
        "a0p5_l0_runner_r": -1.0,
        "a0p5_l0_banked": False,
        "a0p5_l0_exit_time": 300.0,
        "a0p5_l0_label": "SL",
        "near_exit_time": 300.0,
        "risk_pct": .01,
        "macro_risk_on_score": 0,
    }])
    result = audit.apply_policy(frame, lambda x: pd.Series(True, index=x.index), .75)
    assert result.iloc[0].net_r == -1.0
    assert result.iloc[0].exit_time == 300.0


def test_report_keeps_all_later_periods_sealed():
    report = json.loads(Path(audit.OUT).read_text(encoding="utf-8"))
    assert report["protocol"]["selection"] == "2022H1"
    assert report["protocol"]["confirmation"] == ["2022H2", "2023"]
    assert report["protocol"]["sealed_holdouts"] == [
        "2024", "2025", "2026", "direct X-Perp"]
    assert report["target_30pct_each_month_proven"] is False
