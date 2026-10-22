import pandas as pd

import crypto_okx_swap_1h_stable_target_router_oct22_audit as audit


def test_stable_symbol_policy_routes_only_frozen_material_symbols():
    frame = pd.DataFrame({
        "entry_time": [1_700_000_000] * 3,
        "symbol": ["ETHUSDT", "BTCUSDT", "XLMUSDT"],
        "btc_ret168": [0.0] * 3,
        "direction": [-1] * 3,
        "family_trend": [1.0] * 3,
    })
    assert audit.target_mask(frame, "stable_symbols_2pct").tolist() == [
        True, False, False]
