import pytest

import crypto_observed_be_nov16_audit as audit


def fixture(last_ticks):
    seconds = dict(time=list(range(120)), **{k: [] for k in ('open', 'high', 'low', 'close', 'volume')})
    ticks = {11: [(11100, 1, 102., 1.), (11900, 2, 100., 1.)], **last_ticks}
    carry = 100.
    for second in range(120):
        rows = ticks.get(second, [])
        prices = [r[2] for r in rows] or [carry]
        for key, value in dict(open=prices[0], high=max(prices), low=min(prices),
                               close=prices[-1], volume=sum(r[3] for r in rows)).items():
            seconds[key].append(value)
        carry = prices[-1]
    minutes = dict(time=[0, 60], **{k: [] for k in ('open', 'high', 'low', 'close', 'volume')})
    for start in (0, 60):
        live = [i for i in range(start, start + 60) if seconds['volume'][i] > 0]
        values = dict(open=seconds['open'][live[0]], high=max(seconds['high'][i] for i in live),
                      low=min(seconds['low'][i] for i in live), close=seconds['close'][live[-1]],
                      volume=sum(seconds['volume'][start:start + 60]))
        for key, value in values.items():
            minutes[key].append(value)
    return minutes, seconds, ticks


def run_case(monkeypatch, tmp_path, last_ticks):
    minutes, seconds, ticks = fixture(last_ticks)
    def candles(inst, start, end, interval, folder):
        if interval == 60:
            return minutes, 'minute-hash'
        assert interval == 1 and end - start == 60
        return {k: v[start:end] for k, v in seconds.items()}, 'second-hash'
    def trades(inst, start, expected, folder):
        assert expected['volume'] == sum(r[3] for r in ticks[start])
        return ticks[start], 'trade-hash'
    monkeypatch.setattr(audit, 'fetch_window', candles)
    monkeypatch.setattr(audit, 'fetch_second_trades', trades)
    monkeypatch.setattr(audit.time, 'sleep', lambda _: None)
    row = dict(id=1, symbol='TEST', direction='LONG', entry_price=100., sl=99., tp1=100.5, tp2=102.,
               opened_at=11.2, closed_at=90.25, tp1_hit_at=None, status='SL_HIT', realized_r=-1.,
               strategy_runner_mode='fixed_be', strategy_bar_seconds=900, strategy_stop_on_close=False)
    return audit.reconstruct(row, 'TEST', tmp_path)


def test_pre_publication_target_cannot_hide_a_later_real_stop(monkeypatch, tmp_path):
    result = run_case(monkeypatch, tmp_path, {80: [(80000, 3, 98.9, 1.)]})
    for path in result['scenarios'].values():
        assert path['kind'] == 'SL'
        assert path['public_tp1_at'] is None
        assert path['whole_position_r'] == pytest.approx(-1.1)
        assert path['max_favorable_r_before_exit'] == 0.


def test_target_after_recorded_closure_cannot_turn_the_mark_into_a_profit(monkeypatch, tmp_path):
    result = run_case(monkeypatch, tmp_path, {90: [(90100, 3, 100., 1.), (90800, 4, 103., 1.)]})
    for path in result['scenarios'].values():
        assert path['kind'] == 'MODEL_CLOSE_MARK'
        assert path['public_tp1_at'] is None
        assert path['whole_position_r'] == 0.
        assert path['max_favorable_r_before_exit'] == 0.
