import pandas as pd

import crypto_raw_regime_direction_oct19_audit as audit


def test_enrich_uses_same_timestamp_market_and_closed_btc_context():
    rows = []
    for symbol, r4, r16, r96, atr, position in (
            ("BTCUSDT", .02, .04, .08, .01, .7),
            ("ETHUSDT", -.01, .02, .03, .02, .4)):
        for direction in (1, -1):
            rows.append({
                "entry_time": 3600, "symbol": symbol,
                "direction_n": direction, "return_4": r4,
                "return_16": r16, "return_96": r96,
                "atr14_fraction": atr, "range32_position": position,
            })

    result = audit.enrich(pd.DataFrame(rows))
    eth_long = result[(result.symbol == "ETHUSDT")
                      & (result.direction_n == 1)].iloc[0]

    assert eth_long.universe_n == 2
    assert round(eth_long.cross_mean_return_4, 6) == .005
    assert eth_long.btc_return_4 == .02
    assert round(eth_long.relative_return_4, 6) == -.015
    assert eth_long.direction_btc_alignment_16 == .04


def test_direction_alignment_changes_sign_for_short():
    rows = []
    for direction in (1, -1):
        rows.append({
            "entry_time": 3600, "symbol": "BTCUSDT",
            "direction_n": direction, "return_4": .02,
            "return_16": .04, "return_96": .08,
            "atr14_fraction": .01, "range32_position": .7,
        })

    result = audit.enrich(pd.DataFrame(rows)).sort_values("direction_n")

    assert result.direction_btc_alignment_4.tolist() == [-.02, .02]
