import hashlib
import json
from pathlib import Path


ROOT = Path("reports/audit_2026_09_08")


def load(name: str) -> dict:
    return json.loads((ROOT / name).read_text(encoding="utf-8"))


def test_exact_oct31_refresh_has_zero_prefix_or_candle_drift():
    value = load("crypto_profit_first_exact_oct31_1700_forward.json")
    overlap = value["integrity"]["raw_candle_overlap"]
    prefix = value["integrity"]["exact_accepted_prefix"]
    assert overlap["passes"] is True
    assert overlap["symbols"] == 12
    assert overlap["value_mismatches"] == 0
    assert all(row["new_bars_after_old_end"] == 80
               for row in overlap["per_symbol"].values())
    assert prefix["passes"] is True
    assert prefix["old_rows"] == prefix["new_prefix_rows"] == 836
    assert prefix["maximum_net_r_delta"] == 0
    assert prefix["maximum_exit_time_delta_seconds"] == 0
    assert value["new_accepted_rows"] == 6


def test_exact_oct31_results_report_the_full_economic_gate():
    value = load("crypto_profit_first_exact_oct31_1700_forward.json")
    fresh = value["newly_accepted_after_oct30_snapshot"]["margin_3pct"]
    october = value["current"]["october"]
    assert fresh["n"] == 6
    assert fresh["wr"] == 83.33
    assert fresh["return_pct"] == 1.6938
    assert fresh["dd_pct"] == 0.4416
    assert fresh["max_loss_streak"] == 1
    assert fresh["negative_months"] == 0
    assert fresh["trades_per_calendar_day"] == 3.0
    assert fresh["literal_2r_before_be_rate_pct"] == 50.0
    assert fresh["largest_symbol_share_pct"] == 33.3333
    assert fresh["economic_gate"]["frequency_pass"] is True
    assert fresh["economic_gate"]["return_pass"] is False
    assert fresh["economic_gate"]["passes"] is False
    assert october["margin_3pct"]["return_pct"] == 17.6726
    assert october["margin_3pct"]["economic_gate"]["passes"] is False
    assert october["margin_5pct"]["return_pct"] == 30.8346
    assert october["margin_5pct"]["projected_monthly_return_pct"] == 29.7052
    assert october["margin_5pct"]["economic_gate"]["passes"] is False


def test_tp1_volume_bank_fails_first_new_return_gate_and_is_not_activated():
    value = load(
        "crypto_profit_first_exact_tp1_volume_bank_oct31_1700_forward.json")
    baseline = value["baseline_new_forward"]["margin_3pct"]
    candidate = value["candidate_new_forward"]["margin_3pct"]
    assert value["status"] == "FROZEN_TP1_VOLUME_BANK_FORWARD_FAILED"
    assert value["ready_for_live_orders"] is False
    assert value["orders_possible"] is False
    assert value["runtime_change_allowed"] is False
    assert value["integrity"]["passes"] is True
    assert value["new_accepted_rows"] == 6
    assert baseline["return_pct"] == 1.6938
    assert candidate["return_pct"] == 1.0357
    assert candidate["return_pct"] < baseline["return_pct"]
    assert candidate["dd_pct"] == baseline["dd_pct"]
    assert value["forward_gate"] == {
        "return_noninferior": False,
        "drawdown_noninferior": True,
        "negative_months_noninferior": True,
        "passes": False,
    }
    script = Path("crypto_profit_first_exact_tp1_volume_bank_oct31_1700_forward.py")
    assert value["sources"][str(script.resolve())] == hashlib.sha256(
        script.read_bytes()).hexdigest()
