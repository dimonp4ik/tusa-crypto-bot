import math

import crypto_profit_first_material_xperp_oct20_audit as audit


def test_required_move_reaches_monthly_objective():
    move = audit.required_signed_move(.03, 3)
    factor = 1 + .03 * audit.LEVERAGE * move
    assert math.isclose(factor ** 90, 1.30, rel_tol=0, abs_tol=1e-12)


def test_projected_month_uses_observed_frequency():
    move = audit.required_signed_move(.05, 4)
    assert math.isclose(audit.projected_month(move, 4, .05), 30.0,
                        rel_tol=0, abs_tol=1e-10)
