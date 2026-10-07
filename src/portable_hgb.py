"""Tiny pure-Python evaluator for frozen sklearn histogram trees."""
from __future__ import annotations

import math


def _tree_value(nodes: list[list], values: list[float]) -> float:
    index = 0
    while True:
        feature, threshold, missing_left, left, right, value, is_leaf = nodes[index]
        if is_leaf:
            return float(value)
        observed = float(values[int(feature)])
        if math.isnan(observed):
            index = int(left if missing_left else right)
        else:
            index = int(left if observed <= float(threshold) else right)


def predict_probability(model: dict, values: list[float]) -> float:
    """Return the binary-class probability for one ordered feature row."""
    if len(values) != len(model["features"]):
        raise ValueError("portable HGB feature count mismatch")
    raw = float(model["baseline"])
    for nodes in model["trees"]:
        raw += _tree_value(nodes, values)
    if raw >= 0:
        return 1.0 / (1.0 + math.exp(-raw))
    exp_value = math.exp(raw)
    return exp_value / (1.0 + exp_value)


def score_named(model: dict, features: dict[str, float]) -> float:
    """Evaluate a mapping; missing values follow the model's missing branch."""
    values = [float(features.get(name, math.nan)) for name in model["features"]]
    return predict_probability(model, values)
