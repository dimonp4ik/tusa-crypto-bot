import json

from shared_account_dual_bot_oct31_audit import (
    CRYPTO_ROWS, REPORT, account_metrics, shared_gate,
)


def row(bot, symbol, direction, entry, exit_, ret=0.01, quality=1.0):
    return {"bot": bot, "symbol": symbol, "direction": direction,
            "entry_time": entry, "exit_time": exit_, "signed_return": ret,
            "quality": quality, "module": "test"}


def test_shared_direction_cap_counts_both_bots():
    rows = [row("crypto", "BTC", "LONG", 100, 500),
            row("stocks", "NVDA", "LONG", 110, 500),
            row("stocks", "AMD", "LONG", 120, 500)]
    accepted, gate = shared_gate(rows, max_total=4, max_direction=2,
                                 max_hour=9, priority="balanced")
    assert len(accepted) == 2
    assert gate["rejected"] == {"direction_cap": 1}
    assert gate["peak_same_direction"] == 2


def test_shared_total_and_rolling_hour_caps():
    total_rows = [row("crypto", "A", "LONG", 100, 500),
                  row("stocks", "B", "SHORT", 110, 500),
                  row("crypto", "C", "LONG", 120, 500)]
    accepted, gate = shared_gate(total_rows, 2, 2, 9, "balanced")
    assert len(accepted) == 2
    assert gate["rejected"] == {"total_cap": 1}

    hour_rows = [row("crypto", "A", "LONG", 100, 101),
                 row("stocks", "B", "SHORT", 200, 201),
                 row("crypto", "C", "LONG", 300, 301)]
    accepted, gate = shared_gate(hour_rows, 9, 9, 2, "balanced")
    assert len(accepted) == 2
    assert gate["rejected"] == {"rolling_hour_cap": 1}


def test_account_metrics_uses_shared_equity():
    rows = [row("crypto", "A", "LONG", 100, 101, 0.10),
            row("stocks", "B", "SHORT", 200, 201, 0.10)]
    result = account_metrics(rows)
    # 3% margin * 10x * 10% move = 3%; then compound the second trade.
    assert result["return_pct"] == 6.09
    assert result["by_bot"]["crypto"]["account_contribution_pct"] == 3.0
    assert result["by_bot"]["stocks"]["account_contribution_pct"] == 3.09


def test_shared_report_uses_latest_exact_crypto_snapshot_and_exchange_guard():
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    assert "oct31_1700" in CRYPTO_ROWS.name
    assert str(CRYPTO_ROWS) in report["sources"]
    assert report["runtime_shared_cap"]["authoritative_source"] == (
        "OKX /api/v5/account/positions")
    assert report["runtime_shared_cap"]["configured_max_margin_fraction"] == .90
