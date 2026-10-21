import pandas as pd

import crypto_5m_material_loss_cause_oct22_audit as audit


def test_experts_require_positive_expectancy_in_both_years():
    rows = []
    for year, value in ((2022, .01), (2023, -.01)):
        rows.extend({"year": year, "group": "unstable",
                     "signed_return": value} for _ in range(30))
    for year, value in ((2022, .01), (2023, .02)):
        rows.extend({"year": year, "group": "stable",
                     "signed_return": value} for _ in range(30))

    result = audit.experts(pd.DataFrame(rows), ("group",), 30, 0)

    assert result.group.tolist() == ["stable"]
