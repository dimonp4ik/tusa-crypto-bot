import download_current_xperp_history as downloader


class Response:
    def __init__(self, status_code, body, headers=None):
        self.status_code = status_code
        self._body = body
        self.headers = headers or {}

    def raise_for_status(self):
        return None

    def json(self):
        return self._body


class Session:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0

    def get(self, *args, **kwargs):
        self.calls += 1
        return self.responses.pop(0)


def test_get_json_retries_public_rate_limit(monkeypatch):
    sleeps = []
    monkeypatch.setattr(downloader.time, "sleep", sleeps.append)
    session = Session([
        Response(429, {}, {"Retry-After": "2"}),
        Response(200, {"code": "0", "data": [1]}),
    ])

    value = downloader.get_json(session, "https://example.invalid", {})

    assert value == {"code": "0", "data": [1]}
    assert session.calls == 2
    assert sleeps == [2.0]
