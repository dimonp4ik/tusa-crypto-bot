from unittest.mock import patch

import pytest

import main


@pytest.mark.parametrize("exchange,expected", [
    ((True, -1.27), "$-1.27"),
    ((True, 0.), "$+0.00"),
    ((False, "incomplete"), "не подтверждён OKX"),
])
def test_sweep_report_distinguishes_loss_zero_and_unconfirmed_history(exchange, expected):
    user = {"user_id": 15, "last_balance": 130.}
    state = {"watermark_ts": 1_795_248_000., "total_swept": 0.}
    with (patch.object(main.autotrader, "_creds_of", return_value={"fixture": True}),
          patch.object(main, "sweep_state", return_value=state) as watermark,
          patch.object(main._okx, "get_balance", return_value=(False, None)),
          patch.object(main._okx, "get_realized_pnl_since", return_value=exchange) as request,
          patch.object(main, "_at_reply_kb") as reply):
        main._sweep_show_report(15, 15, user)
    watermark.assert_called_once_with(15)
    request.assert_called_once_with({"fixture": True}, state["watermark_ts"])
    _, text, buttons = reply.call_args.args
    assert expected in text
    assert "Последний известный баланс" in text
    assert "по фьючерсам счёта" in text
    assert buttons == [[{"text": "« Назад", "callback_data": "at_menu"}]]


def test_unconfirmed_amount_cannot_spend_or_advance_the_watermark():
    with (patch.object(main, "at_get", return_value={"allowed": True}),
          patch.object(main, "_sweep_pending", return_value=(None, 0.)),
          patch.object(main, "sweep_commit") as commit,
          patch.object(main, "_sweep_execute_buy") as buy,
          patch.object(main, "_answer_callback") as callback):
        assert main._sweep_handle_callback("cb", 15, 15, "sw_buy_BTC")
        commit.assert_not_called()
        buy.assert_not_called()
        assert "не подтвердил" in callback.call_args.args[1]


def test_manual_ticker_requires_private_chat_and_uses_only_current_confirmed_amount():
    user = {"allowed": True, "user_id": 15}
    with (patch.object(main, "TELEGRAM_TOKEN", "fixture-telegram-token"),
          patch.object(main, "DEPLOYMENT_MODE", "live"),
          patch.object(main, "upsert_user"),
          patch.object(main, "_is_admin", return_value=False),
          patch.object(main, "_is_super_admin", return_value=False),
          patch.object(main, "at_get", return_value=user),
          patch.object(main, "_sweep_pending", return_value=(20., 8.)) as balance,
          patch.object(main, "_sweep_execute_buy") as buy,
          patch.object(main, "_reply"),
          patch.dict(main._sweep_pending_manual, {15: 12.}, clear=True)):
        client = main.app.test_client()
        headers = {"X-Telegram-Bot-Api-Secret-Token": main._telegram_webhook_secret()}
        group = {"message": {"chat": {"id": -15}, "from": {"id": 15}, "text": "BTC"}}
        direct = {"message": {"chat": {"id": 15}, "from": {"id": 15}, "text": "BTC"}}
        assert client.post("/webhook", json=group, headers=headers).status_code == 200
        assert main._sweep_pending_manual[15] == 12.
        balance.assert_not_called()
        buy.assert_not_called()
        assert client.post("/webhook", json=direct, headers=headers).status_code == 200
        buy.assert_called_once_with(15, 15, user, "BTC", 8.)
        assert client.post("/webhook", json=direct, headers=headers).status_code == 200
        assert buy.call_count == 1
