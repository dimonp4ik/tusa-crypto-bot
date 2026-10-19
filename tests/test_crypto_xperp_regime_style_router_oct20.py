import numpy as np
import pandas as pd

import crypto_xperp_regime_style_router_oct20_audit as audit


def test_market_context_uses_last_closed_candles():
    times = np.arange(20, dtype=np.int64) * 900
    candles = {}
    for number, symbol in enumerate(("BTCUSDT", "ETHUSDT", "LINKUSDT",
                                     "XLMUSDT", "XRPUSDT"), 1):
        candles[symbol] = {
            "time": times,
            "close": 100 + number + np.arange(20, dtype=float),
        }
    found = audit.market_context(candles)
    assert int(found.entry_time.iloc[-1]) == int(times[-1] + 900)
    assert found.btc_efficiency8.iloc[-1] == 1.0
    assert found.market_breadth_extreme4.iloc[-1] == .5


def test_router_selects_exactly_one_style_per_row():
    frame = pd.DataFrame([
        {"entry_time": 1, "symbol": "BTC", "style": "reversal", "x": .1},
        {"entry_time": 1, "symbol": "BTC", "style": "momentum", "x": .1},
        {"entry_time": 2, "symbol": "ETH", "style": "reversal", "x": .9},
        {"entry_time": 2, "symbol": "ETH", "style": "momentum", "x": .9},
    ])
    found = audit.routed(frame, "x", .5, "momentum")
    assert found["style"].tolist() == ["reversal", "momentum"]
