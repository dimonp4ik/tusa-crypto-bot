import download_current_xperp_5m_history as downloader


def test_snapshot_anchor_is_shared_closed_interval_boundary():
    now = 1_800_000_127
    assert downloader.snapshot_anchor_ms("5m", now) == 1_800_000_000_000
    assert downloader.snapshot_anchor_ms("15m", now) == 1_800_000_000_000


def test_snapshot_anchor_rejects_unknown_interval():
    try:
        downloader.snapshot_anchor_ms("1h", 1_800_000_127)
    except ValueError as error:
        assert "unsupported bar" in str(error)
    else:
        raise AssertionError("unknown interval accepted")
