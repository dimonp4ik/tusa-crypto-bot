import json
from pathlib import Path

import pandas as pd

import crypto_xperp_training_window_oct20_audit as audit


def test_causal_window_excludes_open_outcomes_and_old_entries():
    cutoff = pd.Timestamp("2026-06-01", tz="UTC").timestamp()
    frame = pd.DataFrame({
        "entry_time": [cutoff - 31 * 86400, cutoff - 20 * 86400,
                       cutoff - 1],
        "exit_time": [cutoff - 30 * 86400, cutoff - 19 * 86400,
                      cutoff + 1],
    })
    result = audit.causal_window(frame, "2026-06", 30)
    assert result.entry_time.tolist() == [cutoff - 20 * 86400]


def test_report_is_rejected_and_sources_match():
    report = json.loads(Path(audit.OUT).read_text(encoding="utf-8"))
    assert report["status"] == "REJECTED_OBJECTIVE"
    assert report["ready_for_live_orders"] is False
    assert report["objective_pass"] is False
    for source, expected in report["sources"].items():
        assert audit.digest(Path(source)) == expected
