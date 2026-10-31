import json
from pathlib import Path

import pandas as pd

import crypto_profit_first_fred_macro_oct31_audit as audit


def test_macro_observation_is_delayed_three_calendar_days():
    frame = audit.macro_series("VIXCLS", "vix", percent_change=True)
    raw = pd.read_csv(audit.MACRO_ROOT / "VIXCLS.csv").dropna()
    first_date = pd.Timestamp(raw.iloc[0, 0], tz="UTC")
    assert int(frame.iloc[0].available_at) == int(
        (first_date + pd.Timedelta(days=3)).timestamp())


def test_macro_attachment_has_no_missing_historical_features():
    frame = pd.read_pickle(audit.HISTORICAL).head(50)
    attached = audit.attach_macro(frame)
    assert not attached[audit.MACRO_FEATURES].isna().any().any()


def test_hidden_periods_cannot_select_the_macro_policy():
    report = json.loads(Path(audit.OUT).read_text(encoding="utf-8"))
    assert report["protocol"]["threshold_selection"] == "2022H2 only"
    assert report["protocol"]["confirmation"] == "2023"
    assert report["protocol"]["sealed_holdouts"] == [
        "2024", "2025", "2026", "direct X-Perp"]
    assert report["target_30pct_each_month_proven"] is False
