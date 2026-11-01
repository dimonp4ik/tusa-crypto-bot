import json
from pathlib import Path

import crypto_profit_first_static_veto_nov1_audit as audit


def test_static_veto_report_keeps_holdouts_sealed_and_runtime_unchanged():
    report = json.loads(Path(audit.OUT).read_text(encoding="utf-8"))
    assert report["ready_for_live_orders"] is False
    assert report["runtime_change_allowed"] is False
    assert report["protocol"]["sealed_holdouts"] == [
        "2024", "2025", "2026", "direct X-Perp"]
    assert report["candidate_count"] > 0


def test_bucket_keys_are_string_stable():
    frame = __import__("pandas").DataFrame({
        "family": ["breakout"], "direction": ["LONG"]})
    assert audit.bucket_keys(frame, ("family", "direction")).iloc[0] == (
        "breakout", "LONG")
