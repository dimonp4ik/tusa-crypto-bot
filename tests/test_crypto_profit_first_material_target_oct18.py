import pandas as pd
import pytest

import crypto_profit_first_material_target_oct18_audit as audit


def test_barbell_uses_configured_runner_target():
    frame = pd.DataFrame({
        "near_outcome": ["SL", "TP2", "TP2"],
        "outcome": ["SL", "SL", "TP2"],
    })
    result = audit.barbell_frame(frame, .05, 4.0)
    assert result.net_r.tolist() == pytest.approx([-1.0, .025, 3.825])


def test_targets_include_material_extensions():
    assert audit.TARGETS == (2.0, 3.0, 4.0)


def test_model_shape_is_fixed_before_later_periods():
    assert audit.MODEL == {"max_leaf_nodes": 7,
                           "min_samples_leaf": 300,
                           "learning_rate": .05}
