import json
from pathlib import Path

import pandas as pd

import crypto_xperp_cross_section_reversal_oct20_audit as audit


def test_select_entries_requires_full_universe_and_material_extreme():
    frame = pd.DataFrame({
        "universe_n": [5, 4, 5],
        "cross_z": [1.5, 2.0, .5],
        "signal_return": [.004, .004, .01],
        "direction_n": [-1, -1, -1],
    })
    found = audit.select_entries(frame, 5, 1.0, .002)
    assert found.index.tolist() == [0]
    assert audit.select_entries(frame, 4, 1.0, .002).index.tolist() == [0, 1]


def test_direction_styles_are_opposites():
    candles = {
        "A": {"time": [0, 900, 1800], "close": [100, 102, 102]},
        "B": {"time": [0, 900, 1800], "close": [100, 100, 100]},
        "C": {"time": [0, 900, 1800], "close": [100, 98, 98]},
    }
    reversal = audit.feature_rows(candles, 1, "reversal")
    momentum = audit.feature_rows(candles, 1, "momentum")
    assert reversal.direction_n.tolist() == [
        -value for value in momentum.direction_n.tolist()]


def test_report_is_rejected_and_sources_match():
    report = json.loads(Path(audit.OUT).read_text(encoding="utf-8"))
    assert report["status"] == "REJECTED_OBJECTIVE"
    assert report["ready_for_live_orders"] is False
    assert report["objective_pass"] is False
    for source, expected in report["sources"].items():
        assert audit.digest(Path(source)) == expected
