from __future__ import annotations

import csv
import io
import zipfile

import pandas as pd

import crypto_build_cftc_tff_history as build
import crypto_cftc_tff_material_oct31_audit as audit


def _archive(text: str) -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w") as archive:
        archive.writestr("FinFut24.txt", text)
    return output.getvalue()


def test_official_parser_delays_cme_bitcoin_report_by_seven_days():
    header = ["CFTC_Contract_Market_Code", "Market_and_Exchange_Names",
              "Report_Date_as_YYYY-MM-DD", *build.NUMERIC.values()]
    numbers = ["1,000", "600", "500", "700", "400", "300", "350",
               "100", "90"]
    text = io.StringIO()
    writer = csv.writer(text, lineterminator="\n")
    writer.writerow(header)
    writer.writerow(["133741", "BITCOIN CME", "2024-01-02", *numbers])

    rows = build.parse(_archive(text.getvalue()), 2024)

    assert rows[0]["contract"] == "bitcoin_cme"
    assert rows[0]["available_at_utc"] == "2024-01-09T00:00:00+00:00"
    assert rows[0]["open_interest"] == 1000


def test_attach_uses_only_the_latest_already_available_report():
    trades = pd.DataFrame({
        "entry_time": [99, 100, 199, 200], "direction": ["LONG"]*4})
    cftc = pd.DataFrame({
        "available_time": [100, 200],
        "report_date": ["2023-12-26", "2024-01-02"],
        "am_z": [1.0, 2.0], "lev_z": [3.0, 4.0],
        "dealer_z": [5.0, 6.0], "other_z": [7.0, 8.0],
        "spread_z": [-2.0, -2.0],
    })

    attached = audit.attach(trades, cftc)

    assert attached.entry_time.tolist() == [100, 199, 200]
    assert attached.am_z.tolist() == [1.0, 1.0, 2.0]


def test_rule_applies_positioning_relative_to_trade_direction():
    frame = pd.DataFrame({
        "direction": ["LONG", "SHORT", "LONG", "SHORT"],
        "am_z": [.2, -.2, -.2, .2], "lev_z": [0.0]*4,
        "dealer_z": [0.0]*4, "other_z": [0.0]*4,
        "spread_z": [0.0]*4,
    })

    selected = audit.apply_rule(
        frame, {"name": "asset_manager_confirms", "am_min": 0.0})

    assert selected.index.tolist() == [0, 1]
