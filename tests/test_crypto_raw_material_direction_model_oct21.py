import numpy as np
import pandas as pd

import crypto_raw_material_direction_model_oct21_audit as audit


def feature_frame():
    row = {
        "symbol": "BTCUSDT", "entry_index": 100, "entry_time": 3600,
        "year": 2020, "month": "2020-01",
        "ret1": .01, "ret4": .02, "ret12": .03, "ret24": .04,
        "ret72": .05, "ret168": .06,
        "trend_z12": 1.0, "trend_z24": 1.1,
        "trend_z72": 1.2, "trend_z168": 1.3,
        "eff1": .2, "eff4": .2, "eff12": .2, "eff24": .2,
        "eff72": .2, "eff168": .2, "volume_ratio": 1.0,
        "breakout_up24_atr": .5, "breakout_down24_atr": -.5,
        "breakout_up72_atr": .4, "breakout_down72_atr": -.4,
    }
    return pd.DataFrame([row])


def test_each_state_proposes_both_directions():
    result = audit.directional_frame(feature_frame())

    assert len(result) == 2
    assert set(result.direction.astype(int)) == {-1, 1}
    assert result.entry_time.nunique() == 1


def test_directional_matrix_flips_trend_features():
    result = audit.directional_frame(feature_frame())
    design = audit.schema(result)
    values = audit.matrix(result, design)
    ret1_index = design["raw_fields"].index("ret1")

    assert values[0, ret1_index] == -values[1, ret1_index]


def test_geometry_grid_is_material_and_market_compatible():
    assert all(target >= 2 for target, _, _ in audit.GEOMETRIES)
    assert all(stop >= .75 for _, stop, _ in audit.GEOMETRIES)
    assert all(hold >= 288 for _, _, hold in audit.GEOMETRIES)


def test_labels_preserve_feature_row_identity_across_symbols():
    frame = pd.DataFrame([
        {"symbol": "AAA", "entry_index": 0, "entry_time": 0,
         "year": 2020, "month": "2020-01", "direction": 1},
        {"symbol": "BBB", "entry_index": 0, "entry_time": 0,
         "year": 2020, "month": "2020-01", "direction": 1},
        {"symbol": "AAA", "entry_index": 0, "entry_time": 0,
         "year": 2020, "month": "2020-01", "direction": -1},
        {"symbol": "BBB", "entry_index": 0, "entry_time": 0,
         "year": 2020, "month": "2020-01", "direction": -1},
    ])
    candles = {
        symbol: {
            "time": np.array([0], dtype=np.int64),
            "open": np.array([100.0]), "high": np.array([100.0]),
            "low": np.array([100.0]), "close": np.array([100.0]),
        }
        for symbol in ("AAA", "BBB")
    }

    result = audit.label_identity(frame, candles, (2.0, 1.0, 1))

    expected = frame[["symbol", "entry_index", "direction"]]
    pd.testing.assert_frame_equal(
        result[["symbol", "entry_index", "direction"]], expected)


def test_mark_to_market_drawdown_includes_open_trade_path():
    source = {
        "BTCUSDT": {
            "time": np.array([0, 300], dtype=np.int64),
            "close": np.array([90.0, 101.0]),
        }
    }
    rows = [{
        "symbol": "BTCUSDT", "entry_time": 0, "exit_time": 600,
        "entry_price": 100.0, "direction_n": 1, "source_priority": 0,
        "signed_price_return": .01,
    }]

    value = audit.mtm_drawdown(rows, source)

    assert value["mtm_dd_pct"] == 3.0
    assert value["max_concurrent_positions"] == 1
