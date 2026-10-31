import json
from pathlib import Path

import shared_account_dual_bot_nov1_audit as audit


def test_shared_gate_never_exceeds_authorized_margin():
    report = json.loads(Path(audit.REPORT).read_text(encoding="utf-8"))
    for item in report["scenarios"]:
        assert item["gate"]["peak_margin_occupied_pct"] <= 90


def test_five_pct_scenario_uses_latest_complete_months():
    report = json.loads(Path(audit.REPORT).read_text(encoding="utf-8"))
    value = report["five_pct_each"]
    assert value["crypto_margin_pct"] == 5
    assert value["stock_margin_pct"] == 5
    assert set(value["metrics"]["monthly"]) >= {
        "2026-07", "2026-08", "2026-09", "2026-10"}
    assert report["target"]["proven"] is False


def test_position_exiting_at_new_entry_time_releases_margin():
    rows = [
        {"bot": "crypto", "symbol": "BTC", "direction": "LONG",
         "entry_time": 1, "exit_time": 2, "signed_return": 0.01,
         "quality": 1, "module": "test"},
        {"bot": "stocks", "symbol": "SPY", "direction": "LONG",
         "entry_time": 2, "exit_time": 3, "signed_return": 0.01,
         "quality": 1, "module": "test"},
    ]
    accepted, gate = audit.shared_margin_gate(rows, .90, .90)
    assert len(accepted) == 2
    assert gate["peak_open_positions"] == 1
