import pytest

from contract_volume_history_nov19 import parse
from contract_volume_replay_nov19 import paired, stats, stock_exit


def test_capture_preserves_contract_volume_when_base_volume_is_zero():
    row = ('0', '100', '101', '99', '100', '.05', '0', '5', '1')
    frame = parse([row, row], 0, 900)
    assert frame['volume_base'] == [0.]
    assert frame['volume_contracts'] == [.05]
    with pytest.raises(ValueError, match='incomplete historical interval'):
        parse([row], 0, 1800)


def test_stock_paired_replay_detects_earlier_fractional_stop_before_target():
    times = list(range(0, 18 * 900, 900))
    flat = [100.] * 16
    frame = dict(time=times, open=flat + [100., 102.], high=flat + [100., 103.],
                 low=flat + [98., 102.], close=flat + [99., 103.],
                 volume_base=[1.] * 16 + [0., 1.], volume_contracts=[1.] * 18)
    row = dict(symbol='TEST', bot='stocks', entry=100., entry_time=times[16],
               direction='LONG', risk_pct=.01, target_pct=2., hold_bars=2)
    outcome = stats(paired([row], {'TEST': (frame, {}, '')}))
    assert outcome['transitions'] == {'TP->SL': 1}
    assert outcome['comparable_sum_delta_r'] == pytest.approx(-3.)
    assert outcome['changed'][0]['contracts']['exit_time'] == times[17]


def test_stock_model_refuses_to_claim_a_time_exit_at_a_carried_price():
    frame = dict(time=[0, 900], open=[100., 100.1], close=[100.1, 100.1],
                 high=[100.1, 100.1], low=[100., 100.1], volume=[1., 0.])
    row = dict(entry=100., direction='LONG', risk_pct=.01, target_pct=2., hold_bars=2)
    assert stock_exit(frame, 0, row) == {'known': False, 'reason': 'no_traded_terminal_candle'}
