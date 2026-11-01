import json
from pathlib import Path


REPORT = Path(
    "reports/audit_2026_09_08/crypto_profit_first_prequalified_2r_union_nov1.json"
)


def load() -> dict:
    return json.loads(REPORT.read_text(encoding="utf-8"))


def test_prequalified_2r_union_is_shadow_only_after_sealed_late_checks():
    value = load()
    assert value["status"] == "PREQUALIFIED_2R_UNION_SHADOW_CHALLENGER"
    assert value["robust_improvement"] is True
    assert value["runtime_change_allowed"] is False
    assert value["target_30pct_each_month_proven"] is False
    assert value["protocol"]["rule_count"] == 2
    assert value["checks"]["late_return_not_worse"] is True
    assert value["checks"]["every_direct_month_not_worse"] is False
    assert value["checks"]["latest_october_not_worse"] is False


def test_direct_gain_and_risk_shape_both_improve_after_cluster_admission():
    value = load()["direct_xperp"]["margin_3pct"]
    assert value["combined"]["return_pct"] > value["baseline"]["return_pct"]
    assert value["combined"]["dd_pct"] < value["baseline"]["dd_pct"]
    assert (value["combined"]["max_loss_streak"]
            <= value["baseline"]["max_loss_streak"])
