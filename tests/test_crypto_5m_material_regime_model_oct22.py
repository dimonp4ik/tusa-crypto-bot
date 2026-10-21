import numpy as np
import pandas as pd

import crypto_5m_material_regime_model_oct22_audit as audit


def test_material_geometries_do_not_use_micro_targets():
    assert all(target >= 1 for target, _, _ in audit.GEOMETRIES)
    assert all(stop >= .75 for _, stop, _ in audit.GEOMETRIES)
    assert all(hold >= 144 for _, _, hold in audit.GEOMETRIES)


def test_scored_keeps_scores_aligned_with_rows():
    frame = pd.DataFrame({"row": [0, 1, 2]})
    scores = np.array([.1, .9, .8])

    result = audit.scored(frame, scores, "model", .8)

    assert result.row.tolist() == [1, 2]
    assert result.model_probability.tolist() == [.9, .8]
