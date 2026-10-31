import json
from pathlib import Path

import pandas as pd
import pytest

import crypto_profit_first_tp1_volume_next_open_nov1_audit as audit


def test_policy_uses_next_open_value_for_extra_close():
    frame = pd.DataFrame([{
        "sequential_banked": True,
        "sequential_runner_r": 2.0,
        "runner_open_at_action": True,
        "hit_volume_ratio": 1.0,
        "action_open_r": 0.4,
        "risk_pct": 0.01,
    }])
    result = audit.apply_policy(
        frame, volume_threshold=1.5, extra_close_fraction=.75, direct=True)
    expected = .05 * .5 + .75 * .4 + .20 * 2.0
    assert result.iloc[0].net_r == pytest.approx(expected)


def test_policy_leaves_non_actionable_runner_unchanged():
    frame = pd.DataFrame([{
        "sequential_banked": True,
        "sequential_runner_r": 0.0,
        "runner_open_at_action": False,
        "hit_volume_ratio": 1.0,
        "action_open_r": 0.4,
        "risk_pct": 0.01,
    }])
    result = audit.apply_policy(
        frame, volume_threshold=1.5, extra_close_fraction=.75, direct=True)
    assert result.iloc[0].net_r == .025
    assert result.iloc[0].tp1_extra_close_fraction == 0


def test_report_does_not_activate_before_genuine_forward_gate():
    report = json.loads(Path(audit.OUT).read_text(encoding="utf-8"))
    assert report["forward_gate"]["observed_resolved_rows"] < 30
    assert report["forward_gate"]["passes"] is False
    assert report["runtime_change_allowed"] is False
