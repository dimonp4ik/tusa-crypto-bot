import pandas as pd

import crypto_profit_first_material_ensemble_oct18_audit as audit


def test_union_and_intersection_use_expected_cutoffs():
    frame = pd.DataFrame({"family": ["breakout"]})
    broad = {"bear": {"threshold": .3, "near_fraction": .05}}
    family = {"bear|breakout": {"btc_regime": "bear", "family": "breakout",
                                 "threshold": .4, "near_fraction": .1}}
    policies = audit.policy_maps(frame, broad, family)
    assert policies["union"]["bear|breakout"]["threshold"] == .3
    assert policies["intersection"]["bear|breakout"]["threshold"] == .4


def test_intersection_excludes_missing_family_route():
    frame = pd.DataFrame({"family": ["breakout", "range_reversion"]})
    broad = {"bull": {"threshold": .3, "near_fraction": .05}}
    family = {"bull|breakout": {"btc_regime": "bull", "family": "breakout",
                                 "threshold": .4, "near_fraction": .1}}
    policies = audit.policy_maps(frame, broad, family)
    assert "bull|range_reversion" in policies["union"]
    assert "bull|range_reversion" not in policies["intersection"]
