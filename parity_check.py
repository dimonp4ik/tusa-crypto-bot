#!/usr/bin/env python3
"""
Fast parity checks between live bot helpers and research/backtest helpers.

Run before trusting a new backtest change:
  python parity_check.py
"""

from __future__ import annotations

import importlib.util
from importlib.machinery import ModuleSpec
import os
import sys
import types


ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)


if importlib.util.find_spec("dotenv") is None:
    dotenv_stub = types.ModuleType("dotenv")
    dotenv_stub.__spec__ = ModuleSpec("dotenv", loader=None)
    dotenv_stub.load_dotenv = lambda *args, **kwargs: None
    sys.modules["dotenv"] = dotenv_stub

if importlib.util.find_spec("requests") is None:
    requests_stub = types.ModuleType("requests")
    requests_stub.__spec__ = ModuleSpec("requests", loader=None)
    requests_stub.post = lambda *args, **kwargs: None
    requests_stub.get = lambda *args, **kwargs: None
    sys.modules["requests"] = requests_stub


from backtest import calculate_tp_sl_local, gross_r_for_outcome  # noqa: E402
from src.telegram_notifier import calculate_tp_sl  # noqa: E402


def _almost_equal(a: float, b: float, eps: float = 1e-8) -> bool:
    return abs(float(a) - float(b)) <= eps


def check_tp_sl_parity() -> list[str]:
    cases = [
        dict(price=100.0, direction="LONG", atr=1.2, recent_high=104.0, recent_low=98.5),
        dict(price=100.0, direction="SHORT", atr=1.2, recent_high=101.5, recent_low=96.0),
        dict(price=12.345, direction="LONG", atr=0.09, recent_high=12.9, recent_low=12.1, tp1_level=12.7, tp2_level=13.1),
        dict(price=12.345, direction="SHORT", atr=0.09, recent_high=12.6, recent_low=11.9, tp1_level=12.0, tp2_level=11.7),
        dict(price=2450.0, direction="LONG", atr=0.0, recent_high=0.0, recent_low=0.0),
        dict(price=2450.0, direction="SHORT", atr=0.0, recent_high=0.0, recent_low=0.0),
        # TP2 structural in the 1.0-1.5R zone — guards the live 1.5R min-distance rule
        dict(price=100.0, direction="LONG", atr=0.0, recent_high=0.0, recent_low=0.0,
             tp1_level=104.0, tp2_level=104.3),
        dict(price=100.0, direction="SHORT", atr=0.0, recent_high=0.0, recent_low=0.0,
             tp1_level=96.0, tp2_level=95.7),
    ]

    failures = []
    for case in cases:
        live = calculate_tp_sl(**case)
        test = calculate_tp_sl_local(**case)
        if not all(_almost_equal(a, b) for a, b in zip(live, test)):
            failures.append(f"TP/SL mismatch {case}: live={live} backtest={test}")
    return failures


def check_r_model() -> list[str]:
    failures = []
    entry, tp1, tp2, sl = 100.0, 101.0, 102.0, 99.0
    expected = {"TP2": 1.5, "TP1": 0.5, "SL": -1.0, "EXPIRED": 0.0}
    for outcome, value in expected.items():
        actual = gross_r_for_outcome(outcome, entry, tp1, tp2, sl)
        if not _almost_equal(actual, value):
            failures.append(f"R model mismatch {outcome}: got {actual}, expected {value}")
    return failures


def main() -> int:
    failures = check_tp_sl_parity() + check_r_model()
    if failures:
        print("FAIL")
        for item in failures:
            print(f"- {item}")
        return 1
    print("PASS: live/backtest TP-SL parity and R model checks are OK.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
