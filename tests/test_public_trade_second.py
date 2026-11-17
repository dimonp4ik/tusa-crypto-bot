import json

import pytest

from public_trade_second import fetch_second_trades, validate_minute_seconds, validate_second_trades


def expected():
    return dict(open=100., high=101., low=100., close=101., volume=3.)


def trades():
    return [dict(instId='TEST', ts='1000230', tradeId='2', px='101', sz='2'),
            dict(instId='TEST', ts='1000120', tradeId='1', px='100', sz='1')]


def test_complete_contract_volume_and_price_path_are_required():
    result = validate_second_trades(trades(), 1000, expected(), instrument='TEST')
    assert [r[0] for r in result] == [1000120, 1000230]
    with pytest.raises(ValueError, match='volume'):
        validate_second_trades(trades()[:1], 1000, expected(), instrument='TEST')
    with pytest.raises(ValueError, match='OHLC'):
        validate_second_trades(trades(), 1000, dict(expected(), high=102.), instrument='TEST')


def test_cached_trades_without_boundary_coverage_are_not_accepted(tmp_path):
    body = dict(instrument='TEST', start=1000, volume_unit='contracts',
                covered_before_ms=1000000, data=trades())
    (tmp_path / 'TEST_trades_1000.json').write_text(json.dumps(body), encoding='utf-8')
    with pytest.raises(ValueError, match='coverage'):
        fetch_second_trades('TEST', 1000, expected(), tmp_path)


def test_zero_volume_carry_is_not_an_executed_trade():
    assert validate_second_trades([], 1000, dict(expected(), volume=0.), instrument='TEST') == []


def test_pre_trade_carried_price_cannot_replace_the_first_real_minute_trade():
    seconds = dict(time=list(range(1020, 1080)), open=[90.] * 60, high=[90.] * 60,
                   low=[90.] * 60, close=[90.] * 60, volume=[0.] * 60)
    for key in ('open', 'high', 'low', 'close'):
        seconds[key][20:] = [100.] * 40
    seconds['volume'][20] = 3.
    validate_minute_seconds(dict(open=100., high=100., low=100., close=100., volume=3.), seconds, 1020)
    with pytest.raises(ValueError, match='OHLC'):
        validate_minute_seconds(dict(open=90., high=100., low=90., close=100., volume=3.), seconds, 1020)
