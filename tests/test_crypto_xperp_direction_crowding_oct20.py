import json
from pathlib import Path

import crypto_xperp_direction_crowding_oct20_audit as audit


def test_direction_crowding_filter_is_strictly_backward_looking():
    rows = [
        {"entry_time": 100, "direction": "LONG"},
        {"entry_time": 200, "direction": "LONG"},
        {"entry_time": 300, "direction": "LONG"},
        {"entry_time": 400, "direction": "LONG"},
        {"entry_time": 500, "direction": "SHORT"},
    ]
    filtered = audit.direction_crowding_filter(rows, 1, .75, 3)
    assert [row["entry_time"] for row in filtered] == [100, 200, 300, 500]


def test_rejected_candidates_still_count_toward_crowding():
    rows = [
        {"entry_time": 100, "direction": "LONG"},
        {"entry_time": 200, "direction": "LONG"},
        {"entry_time": 300, "direction": "LONG"},
        {"entry_time": 400, "direction": "LONG"},
        {"entry_time": 500, "direction": "LONG"},
    ]
    filtered = audit.direction_crowding_filter(rows, 1, .75, 3)
    assert [row["entry_time"] for row in filtered] == [100, 200, 300]


def test_report_is_rejected_and_sources_match():
    report = json.loads(Path(audit.OUT).read_text(encoding="utf-8"))
    assert report["status"] == "REJECTED_OBJECTIVE"
    assert report["ready_for_live_orders"] is False
    assert report["objective_pass"] is False
    for source, expected in report["sources"].items():
        assert audit.digest(Path(source)) == expected
