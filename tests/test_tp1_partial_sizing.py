import os
os.environ["PYTHON_DOTENV_DISABLED"] = "1"

from contextlib import ExitStack
import time
from unittest.mock import patch

import pytest

from src import autotrader as at
from src import okx_trader as okx


@pytest.mark.parametrize(
    "position,lot,minimum,requested,expected_size,expected_pct,minimum_pct",
    [
        (5, 1, 1, 5, 0, 0, 20),          # BTC: 5% is below one contract.
        (17, 1, 1, 5, 0, 0, 100 / 17),   # ETH: the same failure is explicit.
        (2.9, .1, .1, 5, .1, 100 / 29, 100 / 29),
        (43, 1, 1, 5, 2, 200 / 43, 100 / 43),
        (5, 1, 1, 100, 5, 100, 20),
    ],
)
def test_partial_close_respects_contract_lot_without_rounding_up(
        position, lot, minimum, requested, expected_size, expected_pct,
        minimum_pct):
    plan = okx.calc_partial_close(
        position, requested, {"lotSz": lot, "minSz": minimum})

    assert plan["size"] == pytest.approx(expected_size)
    assert plan["actual_pct"] == pytest.approx(expected_pct)
    assert plan["minimum_size"] == pytest.approx(minimum)
    assert plan["minimum_pct"] == pytest.approx(minimum_pct)


def test_partial_close_uses_exchange_minimum_when_above_one_lot():
    plan = okx.calc_partial_close(
        20, 10, {"lotSz": 1, "minSz": 3})

    assert plan["size"] == 0
    assert plan["minimum_size"] == 3
    assert plan["minimum_pct"] == 15


def test_partial_close_decimal_floor_does_not_lose_an_exact_lot():
    plan = okx.calc_partial_close(
        2.9, 5, {"lotSz": .1, "minSz": .1})

    assert plan["size"] == pytest.approx(.1)


def test_live_open_keeps_full_runner_and_reports_unplaceable_five_percent():
    spec = {"ctVal": 1, "lotSz": 1, "minSz": 1, "tickSz": .1}
    signal = {
        "id": 11, "symbol": "BTCUSDT", "direction": "LONG",
        "entry_price": 7.8, "sl": 7, "tp1": 8.5, "tp2": 10,
    }
    messages = []

    with ExitStack() as stack:
        for name, value in [
            ("_creds_of", {"test": True}),
            ("at_has_open_position", False),
            ("at_set_balance", None),
            ("get_bot_state", None),
            ("set_bot_state", None),
            ("set_signal_size_mult", None),
            ("at_create_entry_intent", 91),
            ("at_update_entry_intent_protection", None),
            ("at_activate_entry_intent", True),
        ]:
            stack.enter_context(patch.object(at, name, return_value=value))
        stack.enter_context(patch.object(
            at, "_dm", side_effect=lambda _uid, text: messages.append(text)))
        for name, value in [
            ("validate_autotrade_account", (True, {
                "position_mode": "net_mode", "permissions": ("read_only", "trade")})),
            ("get_balance", (True, 130)),
            ("get_open_position_margin", (
                True, {"margin_usd": 0, "count": 0, "inst_ids": ()})),
            ("get_xperp_spec", spec),
            ("get_last_price", 7.8),
            ("ensure_leverage", (True, None)),
            ("get_order_book", {
                "ts": time.time() * 1000,
                "bids": [["7.799", "100"]], "asks": [["7.8", "100"]],
            }),
            ("place_market_entry", (True, "entry-1")),
            ("get_position_avg_px", 7.8),
            ("place_protection_oco", (True, "stop-1")),
        ]:
            stack.enter_context(patch.object(at.okx, name, return_value=value))
        stack.enter_context(patch.object(
            at.okx, "get_position_size",
            side_effect=[(True, 0), (True, 5)]))
        partial_order = stack.enter_context(patch.object(
            at.okx, "place_tp1_partial", return_value=(True, "tp1-1")))

        at._open_for_user(
            {"user_id": 7, "size_mode": "percent", "size_value": 3,
             "tp1_close_pct": 5},
            signal, "BTC-USDT-SWAP", "BTC")

    partial_order.assert_not_called()
    assert messages
    assert "минимум 20.0%" in messages[-1]
    assert "Частичный ордер не выставлен" in messages[-1]
    assert "весь объём остаётся под трейлингом" in messages[-1]
