import hashlib
import json
from pathlib import Path


REPORT = Path(
    "reports/audit_2026_09_08/crypto_okx_swap_1h_stable_direction_oct22.json")
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
    assert policy["shadow_only"] is True
