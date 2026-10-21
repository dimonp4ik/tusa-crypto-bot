import pandas as pd

import crypto_raw_material_regime_experts_oct21_audit as audit


def test_expert_must_be_positive_in_each_development_year():
    frame = pd.DataFrame([
        {"year": 2022, "group": "stable", "signed_return": .01},
        {"year": 2023, "group": "stable", "signed_return": .02},
        {"year": 2022, "group": "unstable", "signed_return": .03},
        {"year": 2023, "group": "unstable", "signed_return": -.01},
    ])

    result = audit.stable_experts(frame, ("group",), 1, 0)

    assert result.group.tolist() == ["stable"]


def test_route_uses_only_frozen_expert_groups():
    frame = pd.DataFrame([
        {"group": "keep", "model_probability": .4},
        {"group": "drop", "model_probability": .9},
    ])
    experts = pd.DataFrame([{
        "group": "keep", "worst_year_mean": .001,
        "average_mean": .002,
    }])

    result = audit.route(frame, ("group",), experts, "test")

    assert result.group.tolist() == ["keep"]
    assert result.spec_name.tolist() == ["test"]
