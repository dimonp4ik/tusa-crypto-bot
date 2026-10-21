import json
from pathlib import Path

import pandas as pd

from crypto_15m_frozen_shadow_materiality_oct22_audit import resolve_literal


def test_literal_resolver_uses_stop_first_for_same_bar_collision():
    frame = pd.DataFrame([{
        "symbol": "BTCUSDT", "entry_time": 900, "direction": "LONG",
    }])
    candles = {"BTCUSDT": {
        "time": [0, 900, 1800], "open": [100, 100, 100],
        "high": [100, 102, 100], "low": [100, 98, 100],
        "close": [100, 100, 100], "volume": [1, 1, 1],
    }}

    rows = resolve_literal(frame, candles, 0.01, "test")

    assert len(rows) == 1
    assert rows[0]["outcome"] == "SL"
    assert rows[0]["signed_price_return"] == -0.01


def test_oct22_report_rejects_nonmaterial_shadow_rules():
    report = json.loads(Path(
        "reports/audit_2026_09_08/crypto_15m_frozen_shadow_materiality_oct22.json"
    ).read_text(encoding="utf-8"))

    assert report["status"] == "REJECTED_MATERIALITY_AND_OBJECTIVE"
    assert report["ready_for_live_orders"] is False
    assert all(
        value["original_target_materiality"]["runtime_material_n"] == 0
        for value in report["candidates"].values()
    )
    diagnostic = report["posthoc_one_pct_portfolio_diagnostic"]
    assert diagnostic["not_activation_evidence"] is True
    assert diagnostic["october_with_satellites"]["return_pct"] < 0
