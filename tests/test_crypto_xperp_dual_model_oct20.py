import json
from pathlib import Path

import pandas as pd

import crypto_xperp_dual_model_oct20_audit as audit


def test_profitability_veto_is_inclusive():
    frame = pd.DataFrame({"profit_score": [.4, .5, .6]})
    assert audit.gated(frame, .5).profit_score.tolist() == [.5, .6]


def test_report_is_rejected_and_sources_match():
    report = json.loads(Path(audit.OUT).read_text(encoding="utf-8"))
    assert report["status"] == "REJECTED_OBJECTIVE"
    assert report["ready_for_live_orders"] is False
    assert report["objective_pass"] is False
    for source, expected in report["sources"].items():
        assert audit.digest(Path(source)) == expected
