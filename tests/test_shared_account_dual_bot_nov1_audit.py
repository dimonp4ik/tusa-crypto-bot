import json
from pathlib import Path

import shared_account_dual_bot_nov1_audit as audit


def test_shared_gate_never_exceeds_authorized_margin():
    report = json.loads(Path(audit.REPORT).read_text(encoding="utf-8"))
    for item in report["scenarios"]:
        assert item["gate"]["peak_margin_occupied_pct"] <= 90
        assert item["gate"]["peak_open_positions"] <= audit.MAX_POSITIONS


def test_position_cap_does_not_remove_frozen_route_history():
    report = json.loads(Path(audit.REPORT).read_text(encoding="utf-8"))
    assert report["scope"]["shared_position_cap"] == 12
    for item in report["scenarios"]:
        assert item["gate"]["rejected_by_position_cap"] == {}


def test_five_pct_scenario_uses_latest_complete_months():
    report = json.loads(Path(audit.REPORT).read_text(encoding="utf-8"))
    value = report["five_pct_each"]
    assert value["crypto_margin_pct"] == 5
    assert value["stock_margin_pct"] == 5
    assert set(value["metrics"]["monthly"]) >= {
        "2026-07", "2026-08", "2026-09", "2026-10"}
    assert report["target"]["proven"] is False


def test_margin_releases_only_after_exit_timestamp():
    rows = [
        {"bot": "crypto", "symbol": "BTC", "direction": "LONG",
         "entry_time": 1, "exit_time": 2, "signed_return": 0.01,
         "quality": 1, "module": "test"},
        {"bot": "stocks", "symbol": "SPY", "direction": "LONG",
         "entry_time": 2, "exit_time": 3, "signed_return": 0.01,
         "quality": 1, "module": "test"},
    ]
    accepted, gate = audit.shared_margin_gate(rows, .90, .90)
    assert accepted == rows[:1]
    assert gate["rejected_by_margin_cap"] == {"stocks": 1}

    rows[1]["entry_time"] = 2.1
    accepted, gate = audit.shared_margin_gate(rows, .90, .90)
    assert accepted == rows
    assert gate["peak_open_positions"] == 1


def test_btc_cap_uses_only_candles_closed_strictly_before_entry():
    entry = 172800
    rows = [
        {"bot": "crypto", "symbol": "ETH", "direction": "LONG",
         "entry_time": entry - 900, "exit_time": entry + 900,
         "signed_return": 0.01},
        {"bot": "stocks", "symbol": "SPY", "direction": "LONG",
         "entry_time": entry, "exit_time": entry + 900,
         "signed_return": 0.01},
    ]
    closes = {
        entry - 1800: 100, entry - 1800 - 86400: 100,
        entry - 900: 100, entry - 900 - 86400: 100,
        entry: 97, entry - 86400: 100,
    }
    accepted, gate = audit.shared_margin_gate(
        rows, .03, .03, max_per_direction=1, btc_closes=closes)
    assert accepted == rows
    assert gate["missing_btc_context"] == {}

    rows[1]["entry_time"] += 1
    accepted, gate = audit.shared_margin_gate(
        rows, .03, .03, max_per_direction=1, btc_closes=closes)
    assert accepted == rows[:1]
    assert gate["rejected_by_direction_cap"] == {"stocks": 1}

    del closes[entry - 86400]
    accepted, gate = audit.shared_margin_gate(
        rows, .03, .03, max_per_direction=1, btc_closes=closes)
    assert accepted == rows
    assert gate["missing_btc_context"] == {"stocks": 1}


def test_short_health_uses_only_resolved_shadow_outcomes():
    entry = 2_000_000
    seed = [
        {"bot": "crypto", "direction": "SHORT", "entry_time": entry - 100,
         "exit_time": entry - 1, "signed_return": -0.01}
        for _ in range(14)
    ]
    boundary = dict(seed[0], exit_time=entry)
    first = dict(seed[0], entry_time=entry, exit_time=entry + 1)
    blocked = dict(seed[0], entry_time=entry + 2, exit_time=entry + 3,
                   signed_return=0.5)
    recovered = dict(seed[0], entry_time=entry + 4, exit_time=entry + 5)
    stock = dict(blocked, bot="stocks")
    rows = [recovered, stock, first, *seed, blocked, boundary]

    selected = audit.filter_crypto_shorts(rows)

    assert selected == [recovered, stock, first, *seed, boundary]
