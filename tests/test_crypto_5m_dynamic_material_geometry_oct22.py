import pandas as pd

import crypto_5m_dynamic_material_geometry_oct22_audit as audit


def test_geometry_identity_detects_reordering():
    frame = pd.DataFrame({"symbol": ["BTC", "ETH"],
                          "entry_time": [1, 2], "direction": [1, -1]})
    reordered = frame.iloc[::-1].reset_index(drop=True)

    assert not audit.identity(frame).equals(audit.identity(reordered))


def test_all_dynamic_choices_are_material():
    assert all(target >= 1 for target, _, _ in audit.GEOMETRIES)
