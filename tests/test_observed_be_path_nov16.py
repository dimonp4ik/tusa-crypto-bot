import pytest

from observed_be_path_nov16 import PricePath


def row(**extra):
    return dict(dict(direction='LONG', entry_price=100., sl=99., tp1=100.5, tp2=102.,
                     opened_at=11., tp1_hit_at=70.25, strategy_bar_seconds=900), **extra)


def test_observed_stop_does_not_use_a_retrace_that_preceded_its_activation():
    path = PricePath(row(), 'observed_tp1')
    for stamp, price in ((20., 100.6), (70.1, 99.9), (70.5, 100.4), (80., 102.1)):
        path.consume(stamp, price)
    assert path.finish()['kind'] == 'TP2'
    assert path.finish()['nominal_five_percent_r'] == pytest.approx(1.925)


def test_observed_breakeven_can_exit_before_a_later_same_strategy_bar_target():
    delayed = PricePath(row(), 'next_strategy_bar')
    observed = PricePath(row(), 'observed_tp1')
    for stamp, price in ((20., 100.6), (70.5, 99.9), (80., 102.1)):
        delayed.consume(stamp, price)
        observed.consume(stamp, price)
    assert delayed.finish()['kind'] == 'TP2'
    assert observed.finish()['kind'] == 'BE_TRIGGER'
    assert observed.finish()['whole_position_r'] == pytest.approx(-.1)


def test_model_observation_does_not_invent_an_executed_partial_tp1():
    path = PricePath(row(), 'observed_tp1')
    path.consume(71., 99.9)
    out = path.finish()
    assert out['public_tp1_at'] is None and not out['tp1_before_exit']
    assert out['whole_position_r'] == out['nominal_five_percent_r']


def test_future_breakeven_inside_a_bar_forces_finer_chronological_evidence():
    path = PricePath(row(), 'observed_tp1')
    assert path.needs_detail(dict(time=60., seconds=60, high=100.4, low=99.9, volume=1.))
    assert not path.needs_detail(dict(time=0., seconds=60, high=100.4, low=99.9, volume=1.))


@pytest.mark.parametrize('direction,low,high', [('LONG',99.2,100.4),('SHORT',99.6,100.8)])
def test_untouched_bar_preserves_excursions_missing_from_its_close(direction,low,high):
    data=row()
    if direction=='SHORT':
        data.update(direction=direction,sl=101.,tp1=99.5,tp2=98.)
    path=PricePath(data,'next_strategy_bar')
    path.consume_untouched_bar(dict(time=20.,seconds=1,volume=5,low=low,high=high,close=100.))
    result=path.finish()
    assert result['max_favorable_r_before_exit']==pytest.approx(.4)
    assert result['max_adverse_r_through_exit']==pytest.approx(-.8)


def test_stop_bar_extrema_cannot_be_consumed_without_trade_order():
    path=PricePath(row(),'next_strategy_bar')
    with pytest.raises(ValueError,match='ordered'):
        path.consume_untouched_bar(dict(time=20.,seconds=1,volume=5,low=98.,high=103.,close=102.))
    path.consume(20.1,98.9)
    path.consume(20.2,103.)
    assert path.finish()['max_favorable_r_before_exit']==0.
    assert path.finish()['max_adverse_r_through_exit']==pytest.approx(-1.1)
