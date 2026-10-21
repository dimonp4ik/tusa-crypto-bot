import pandas as pd

import crypto_raw_material_walkforward_oct21_audit as audit


def test_fold_excludes_calibration_and_target_labels():
    fit, calibration = audit.fold_years(2025)

    assert fit == (2020, 2021, 2022, 2023)
    assert calibration == 2024
    assert 2024 not in fit
    assert 2025 not in fit


def test_rolling_expert_requires_both_prior_years():
    common = {
        "direction_name": "LONG", "score_regime": "high",
        "breadth_alignment": "supportive", "dispersion_regime": "normal",
        "win_rate": .5,
    }
    rows = []
    for year, mean in ((2023, .01), (2024, -.01)):
        for _ in range(30):
            rows.append({**common, "year": year, "signed_return": mean})

    result = audit.stable_experts(pd.DataFrame(rows), (2023, 2024))

    assert result.empty
