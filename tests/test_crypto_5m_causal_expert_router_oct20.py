import pandas as pd

import crypto_5m_causal_expert_router_oct20_audit as audit


def test_prior_months_excludes_current_month():
    assert audit.prior_months("2026-05", 3) == [
        "2026-02", "2026-03", "2026-04"]


def test_choose_experts_uses_only_prior_exit_months_and_best_geometry():
    common = {
        "family": ["pullback", "pullback", "pullback"],
        "direction_name": ["LONG"] * 3,
        "session": [1.0] * 3,
        "exit_month": ["2026-02", "2026-03", "2026-05"],
    }
    fast = pd.DataFrame({
        **common, "signed_price_return": [.01, .01, -.50],
        "geometry": ["fast"] * 3,
    })
    slow = pd.DataFrame({
        **common, "signed_price_return": [.005, .005, .50],
        "geometry": ["slow"] * 3,
    })

    chosen = audit.choose_experts(
        {"fast": fast, "slow": slow},
        ("family", "direction_name", "session"), "2026-05", 3,
        min_count=2, min_mean_return=0, min_win_rate=.5)

    assert chosen.geometry.tolist() == ["fast"]


def test_geometry_filter_requires_half_to_two_percent_target_move():
    frame = pd.DataFrame({
        "atr_pct": [.0049, .005, .02, .0201],
        "pnl_s1_t1": [1.0, 2.0, 3.0, 4.0],
        "exit_s1_t1": [1, 2, 3, 4],
    })

    selected = audit.geometry_frame(frame, "s1_t1")

    assert selected.index.tolist() == [1, 2]
