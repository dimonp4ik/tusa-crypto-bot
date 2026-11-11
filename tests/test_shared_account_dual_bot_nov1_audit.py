import json
from pathlib import Path

import shared_account_dual_bot_nov1_audit as audit


def test_shared_gate_never_exceeds_authorized_margin():
    report = json.loads(Path(audit.REPORT).read_text(encoding="utf-8"))
    for item in report["scenarios"]:
        assert item["gate"]["peak_margin_occupied_pct"] <= 90
        assert item["gate"]["peak_open_positions"] <= audit.MAX_POSITIONS


def test_primary_report_models_live_account_reservations():
    report = json.loads(Path(audit.REPORT).read_text(encoding="utf-8"))
    assert report["scope"]["shared_position_cap"] == 12
    for item in report["scenarios"]:
        assert item["gate"]["position_writers"] == 2
        assert item["gate"]["margin_writers"] == 2
        assert item["gate"]["slot_release_delay"] == 900
        assert item["gate"]["peak_open_by_bot"]["crypto"] <= 4
        assert item["gate"]["peak_open_by_bot"]["stocks"] <= 8


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


def test_stock_cluster_holds_short_slot_until_confirmed_release():
    def trade(symbol, entry, exit_time, direction="SHORT"):
        return dict(bot="stocks", symbol=symbol, direction=direction,
                    entry_time=entry, exit_time=exit_time, signed_return=.01)

    first = trade("COINUSDT", 10, 20)
    same_release = trade("MSTRUSDT", 25, 29)
    long = trade("CRCLUSDT", 16, 23, "LONG")
    after_release = trade("MSTRUSDT", 26, 30)
    accepted, gate = audit.shared_margin_gate(
        [first, same_release, long, after_release], .03, .03,
        limit_stock_crypto_cluster=True, slot_release_delay=5)

    assert accepted == [first, long, after_release]
    assert gate["rejected_by_stock_crypto_cluster"] == 1


def test_marked_entry_allocation_is_reused_in_profit_calculation():
    import pytest

    first = dict(bot="stocks", symbol="A", direction="LONG", entry=100,
                 entry_time=0, exit_time=3600, signed_return=.02)
    second = dict(bot="stocks", symbol="B", direction="LONG", entry=100,
                  entry_time=1800, exit_time=4500, signed_return=-.01)
    marks = {("stocks", "A"): {900: 110, 1800: 1}}
    accepted, gate = audit.shared_margin_gate(
        [first, second], .1, .1, sizing_marks=marks)

    assert gate["sizing_basis"] == "prior_candle_marked_equity"
    assert accepted[0]["entry_margin_fraction"] == pytest.approx(.1)
    assert accepted[1]["entry_margin_fraction"] == pytest.approx(.11)
    assert "entry_margin_fraction" not in first
    metrics = audit.account_metrics(accepted, .1, .1)
    assert metrics["return_pct"] == pytest.approx(.9)

    # The mark at exactly the entry timestamp is excluded.
    marks["stocks", "A"][1800] = 1000
    again, _ = audit.shared_margin_gate([first, second], .1, .1, sizing_marks=marks)
    assert again[1]["entry_margin_fraction"] == pytest.approx(.11)

    with pytest.raises(KeyError):
        audit.shared_margin_gate([first, second], .1, .1, sizing_marks={})
