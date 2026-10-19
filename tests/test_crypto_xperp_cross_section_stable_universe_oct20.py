import pickle

import pandas as pd

import crypto_raw_hourly_direction_oct19_audit as parent
import crypto_xperp_cross_section_stable_universe_oct20_audit as audit


def test_selected_universe_has_history_before_may():
    cutoff = pd.Timestamp("2026-05-01", tz="UTC").timestamp()
    for symbol in audit.SYMBOLS:
        path = parent.rails.DIRECT_CACHE / f"{symbol}_15min_current.pkl"
        values = pickle.loads(path.read_bytes())
        assert min(values["time"]) < cutoff
