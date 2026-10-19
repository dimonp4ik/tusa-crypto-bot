import pandas as pd

import crypto_xperp_agreement_router_oct20_audit as audit


def test_intersection_requires_same_symbol_time_style_and_direction():
    left = pd.DataFrame([
        {"symbol": "BTC", "entry_time": 1, "style": "reversal",
         "direction_n": 1, "value": "keep"},
        {"symbol": "ETH", "entry_time": 1, "style": "momentum",
         "direction_n": -1, "value": "drop"},
    ])
    right = pd.DataFrame([
        {"symbol": "BTC", "entry_time": 1, "style": "reversal",
         "direction_n": 1},
        {"symbol": "ETH", "entry_time": 1, "style": "reversal",
         "direction_n": -1},
    ])
    found = audit.intersection(left, right)
    assert found.value.tolist() == ["keep"]
