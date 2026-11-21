from unittest.mock import patch

import main


def test_only_authenticated_updates_access_account_menu(monkeypatch):
    monkeypatch.setattr(main, "TELEGRAM_TOKEN", "test-webhook-token")
    monkeypatch.setattr(main, "DEPLOYMENT_MODE", "live")
    monkeypatch.setattr(main, "keystore_ready", lambda: True)
    client = main.app.test_client()
    update = {"message": {"chat": {"id": 42, "type": "private"}, "from": {"id": 42}, "text": "🤖 Автотрейдинг"}}
    header = "X-Telegram-Bot-Api-Secret-Token"
    user = {"allowed": True, "api_key_enc": "test", "size_mode": "percent"}
    with (patch.object(main, "upsert_user"),
          patch.object(main, "at_get", return_value=user) as account,
          patch.object(main, "_at_show_menu") as menu):
        assert client.post("/webhook", json=update).status_code == 401
        assert client.post("/webhook", json=update, headers={header: "wrong"}).status_code == 401
        account.assert_not_called()
        menu.assert_not_called()
        accepted = client.post("/webhook", json=update, headers={header: main._telegram_webhook_secret()})
        assert accepted.status_code == 200
        account.assert_called_once_with(42)
        menu.assert_called_once_with(42, user)


def test_registration_retries_until_telegram_acknowledges_secret(monkeypatch):
    monkeypatch.setattr(main, "TELEGRAM_TOKEN", "test-webhook-token")
    monkeypatch.setattr(main, "_webhook_registered", False)
    monkeypatch.setattr(main, "_app_url", lambda: "https://test.invalid")
    with patch.object(main._requests, "post") as transport:
        response = transport.return_value
        response.status_code = 200
        response.json.return_value = {"ok": True}
        transport.side_effect = [main._requests.exceptions.Timeout(), response]
        main._setup_webhook()
        assert not main._webhook_registered
        main._setup_webhook()
        assert main._webhook_registered
        main._setup_webhook()
        assert transport.call_count == 2
        assert transport.call_args.kwargs["json"] == {
            "url": "https://test.invalid/webhook",
            "secret_token": main._telegram_webhook_secret(),
        }
