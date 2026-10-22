import hashlib
import json
from pathlib import Path


REPORT = Path(
    "reports/audit_2026_09_08/crypto_okx_swap_1h_stable_direction_oct22.json")
FORWARD_REPORT = Path(
    "reports/audit_2026_09_08/crypto_okx_swap_1h_stable_direction_forward_oct22.json")
MATERIAL_REPORT = Path(
    "reports/audit_2026_09_08/crypto_okx_swap_1h_stable_material_exit_oct22.json")
POLICY = Path("src/frozen_crypto_okx_1h_stable_direction.json")


def test_oct22_stable_direction_candidate_is_shadow_only_and_reproducible():
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    policy = json.loads(POLICY.read_text(encoding="utf-8"))
    direct = report["direct_xperp"]

    assert report["status"] == (
        "STABLE_DIRECTION_AWAITING_NEW_FORWARD_AFTER_2026_10_22")
    assert report["ready_for_live_orders"] is False
    assert report["order_eligible"] is False
    assert report["selected_before_direct"] == (
        "short_plus_long_pullback_or_avax")
    assert report["direct_xperp_passed"] is True
    assert (direct["n"], direct["wr"], direct["return_pct"], direct["dd_pct"]) == (
        142, 72.54, 5.0431, 1.8912)
    assert direct["trades_per_calendar_day"] == 1.919
    assert report["objective_checks"]["at_least_3_trades_per_calendar_day"] is False
    assert policy["route_policy"] == report["selected_before_direct"]
    assert policy["source_report_sha256"] == hashlib.sha256(
        REPORT.read_bytes()).hexdigest()
    assert policy["exit_report_sha256"] == hashlib.sha256(
        MATERIAL_REPORT.read_bytes()).hexdigest()
    assert policy["shadow_only"] is True


def test_oct22_genuinely_new_forward_passes_quality_not_return_objective():
    report = json.loads(FORWARD_REPORT.read_text(encoding="utf-8"))
    fresh = report["fresh_completed"]
    cumulative = report["cumulative_completed"]

    assert report["status"] == "FORWARD_GATE_PASSED_AWAIT_MORE_TIME"
    assert report["ready_for_live_orders"] is False
    assert report["order_eligible"] is False
    assert report["pending_rows"] == 0
    assert (fresh["n"], fresh["wr"], fresh["return_pct"], fresh["dd_pct"]) == (
        56, 71.43, 1.1856, 1.3495)
    assert fresh["trades_per_calendar_day"] == 3.733
    assert fresh["economic_gate"]["projected_monthly_return_pct"] == 2.4048
    assert fresh["objective_pass"] is False
    assert (cumulative["n"], cumulative["wr"], cumulative["return_pct"],
            cumulative["dd_pct"]) == (198, 72.22, 6.2875, 1.8912)


def test_oct22_material_exit_beats_fixed_exit_but_remains_shadow_only():
    report = json.loads(MATERIAL_REPORT.read_text(encoding="utf-8"))
    fresh = report["fresh_post_cutoff"]
    cumulative = report["cumulative_direct"]

    assert report["status"] == "MATERIAL_EXIT_FORWARD_GATE_PASSED_AWAIT_MORE_TIME"
    assert report["ready_for_live_orders"] is False
    assert report["order_eligible"] is False
    assert report["selected_before_later_years_and_direct"]["target_pct"] == 2.0
    assert report["selected_before_later_years_and_direct"]["hold_hours"] == 12
    assert report["later_calendar_checks_passed"] is True
    assert (fresh["n"], fresh["wr"], fresh["return_pct"], fresh["dd_pct"]) == (
        42, 47.62, 3.3321, 1.5825)
    assert fresh["economic_gate"]["projected_monthly_return_pct"] == 6.8579
    assert (cumulative["n"], cumulative["wr"], cumulative["return_pct"],
            cumulative["dd_pct"]) == (172, 44.19, 7.7347, 5.0661)
    assert report["gates"]["improves_fixed_0_5_exit"] is True
    assert report["gates"]["ready_for_live_orders"] is False
