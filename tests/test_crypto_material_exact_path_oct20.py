import math

import crypto_material_exact_path_oct20_audit as audit


def test_partial_runner_math():
    assert audit.partial_net_r("SL", -1.0, .05) == -1.0
    assert audit.partial_net_r("BE", 0.0, .05) == .025
    assert math.isclose(audit.partial_net_r("TP2", 2.0, .05), 1.925)


def test_zero_fraction_is_full_breakeven_runner():
    assert audit.partial_net_r("BE", 0.0, 0.0) == 0.0
    assert audit.partial_net_r("TP2", 2.0, 0.0) == 2.0
