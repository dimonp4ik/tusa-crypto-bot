import math

import crypto_actual_move_xperp_oct20_audit as audit


def test_projection_matches_constant_equity_compounding():
    found = audit.projected_month(.01, 3, .03)
    expected = 100 * ((1 + .03 * 10 * .01) ** 90 - 1)
    assert math.isclose(found, expected)


def test_frozen_impulse_geometry_is_material_and_market_based():
    assert .5 <= audit.IMPULSE_SPEC["target_pct"] <= 2
    assert audit.IMPULSE_SPEC["stop_pct"] == 1.0


def test_required_signed_move_is_predeclared():
    assert audit.REQUIRED_SIGNED_MOVE_3_TRADES_DAY_PCT == .973137
