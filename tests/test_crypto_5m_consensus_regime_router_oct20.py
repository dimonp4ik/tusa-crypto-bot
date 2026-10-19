import pandas as pd

import crypto_5m_consensus_regime_router_oct20_audit as audit


def test_signal_consensus_counts_distinct_evidence_and_conflict():
    events = pd.DataFrame([
        {"entry_time": 300, "direction": 1, "symbol": "BTC", "family": "a"},
        {"entry_time": 300, "direction": 1, "symbol": "ETH", "family": "b"},
        {"entry_time": 300, "direction": 1, "symbol": "ETH", "family": "a"},
        {"entry_time": 300, "direction": -1, "symbol": "XRP", "family": "a"},
        {"entry_time": 600, "direction": -1, "symbol": "BTC", "family": "a"},
    ])

    found = audit.add_signal_consensus(events)
    long = found[(found.entry_time == 300) & (found.direction == 1)].iloc[0]
    short = found[(found.entry_time == 300) & (found.direction == -1)].iloc[0]
    solo = found[found.entry_time == 600].iloc[0]

    assert long.agreeing_symbols == 2
    assert long.agreeing_families == 2
    assert long.directional_consensus == "majority"
    assert short.directional_consensus == "conflicted"
    assert solo.consensus_symbols == "solo"
    assert solo.directional_consensus == "strong"
