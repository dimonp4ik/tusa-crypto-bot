import json

import pytest

from public_candle_window import fetch_window, validate_rows


def test_finalized_15m_reference_cache_is_validated_before_use(tmp_path):
    path = tmp_path / 'TEST_15m_900_2700.json'
    rows = [[str(t * 1000), '100', '102', '99', '101', '0', '0', '0', '1']
            for t in (900, 1800)]
    path.write_text(json.dumps(dict(code='0', data=rows)), encoding='utf-8')
    candles, digest = fetch_window('TEST', 900, 2700, 900, tmp_path)
    assert candles['time'] == [900, 1800]
    assert candles['volume'] == [0, 0]
    assert len(digest) == 64
    rows[1][8] = '0'
    with pytest.raises(ValueError, match='incomplete'):
        validate_rows(rows, 900, 2700, 900)


def test_conflicting_finalized_bar_is_not_silently_accepted():
    row = ['900000', '100', '102', '99', '101', '1', '0', '0', '1']
    revised = row.copy()
    revised[4] = '102'
    with pytest.raises(ValueError, match='conflicting'):
        validate_rows([row, revised], 900, 1800, 900)
