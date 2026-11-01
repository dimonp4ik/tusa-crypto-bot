import json
from pathlib import Path

import crypto_profit_first_adaptive_exit_router_nov1_audit as audit


def test_adaptive_exit_report_rejects_later_holdout_failure():
    report = json.loads(Path(audit.OUT).read_text(encoding="utf-8"))
    assert report["status"] == "REJECTED_ADAPTIVE_EXIT_LATER_HOLDOUT"
    assert report["selected_before_holdouts"][
        "strictly_prior_exit_violations"] == 0
    assert report["historical_holdout_pass"] is False
    assert report["runtime_change_allowed"] is False
