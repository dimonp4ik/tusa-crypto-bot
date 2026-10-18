import numpy as np
import pandas as pd

import crypto_profit_first_walkforward_target_router_oct18_audit as audit


def frame_for_routing():
    rows = []
    for risk in (.003, .03):
        row = {
            "symbol": "BTCUSDT", "entry_time": 1_000 + len(rows),
            "direction": "LONG", "family": "breakout", "risk_pct": risk,
        }
        for target in audit.TARGETS:
            suffix = str(target).replace(".", "_")
            row[f"net_r_{suffix}"] = target
            row[f"outcome_{suffix}"] = "TP2"
            row[f"exit_time_{suffix}"] = 2_000 + target
        rows.append(row)
    return pd.DataFrame(rows)


def test_router_cannot_choose_target_outside_literal_move_band():
    frame = frame_for_routing()
    predictions = np.array([
        [100.0, 50.0, 1.0],  # only 2R gives >=0.5% for 0.3% risk
        [1.0, 50.0, 100.0],  # only 0.5R gives <=2% for 3% risk
    ])
    routed = audit.route(frame, predictions)
    assert routed.target_r.tolist() == [2.0, .5]
    moves = routed.risk_pct * routed.target_r
    assert moves.between(audit.MIN_TARGET_MOVE,
                         audit.MAX_TARGET_MOVE).all()


def test_candidate_rows_keep_highest_quality_family_per_symbol_bar():
    frame = pd.DataFrame([
        {"symbol": "ETHUSDT", "entry_time": 1_000, "direction": "LONG",
         "family": "breakout", "model_quality": .2, "risk_pct": .01,
         "net_r": 1.0, "outcome": "TP2", "exit_time": 2_000,
         "target_r": 1.0},
        {"symbol": "ETHUSDT", "entry_time": 1_000, "direction": "SHORT",
         "family": "range_reversion", "model_quality": .8, "risk_pct": .01,
         "net_r": -1.0, "outcome": "SL", "exit_time": 2_000,
         "target_r": 1.0},
    ])
    rows = audit.candidate_rows(frame, threshold=0.0)
    assert len(rows) == 1
    assert rows[0]["family"] == "range_reversion"
    assert rows[0]["model_probability"] == .8
    assert rows[0]["outcome"] == "SL"
