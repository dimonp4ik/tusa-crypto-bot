import numpy as np
import pandas as pd

import crypto_5m_literal_geometry_router_oct6_audit as audit


def test_outcomes_preserve_feature_row_identity_across_symbols():
    frame = pd.DataFrame([
        {"symbol": "AAA", "entry_index": 0, "direction": 1},
        {"symbol": "BBB", "entry_index": 0, "direction": 1},
        {"symbol": "AAA", "entry_index": 0, "direction": -1},
        {"symbol": "BBB", "entry_index": 0, "direction": -1},
    ])
    candles = {
        symbol: {
            "time": np.array([0], dtype=np.int64),
            "open": np.array([100.0]), "high": np.array([100.0]),
            "low": np.array([100.0]), "close": np.array([100.0]),
        }
        for symbol in ("AAA", "BBB")
    }

    result = audit.add_outcomes(frame, candles, .5, 1)

    expected = frame[["symbol", "entry_index", "direction"]]
    pd.testing.assert_frame_equal(
        result[["symbol", "entry_index", "direction"]], expected)
