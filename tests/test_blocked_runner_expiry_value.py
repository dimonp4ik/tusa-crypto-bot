import pytest
import main
from test_blocked_venue_tracker import setup,candles


@pytest.mark.parametrize('direction',['LONG','SHORT'])
@pytest.mark.parametrize('fraction',[0.,.05,.5])
def test_blocked_runner_expiry_includes_remaining_mark_instead_of_only_tp1(direction,fraction):
    s=setup(max_bars=1);s.update(strategy_tp1_close_frac=fraction,direction=direction)
    bars=candles([(100.,101.2,99.5,100.5)])
    if direction=='SHORT':
        s.update(sl=102.,tp1=99.,tp2=96.)
        bars=candles([(100.,100.5,98.8,99.5)])
    value=main._resolve_unsent_venue_setup(s,bars,now=2700.)
    assert value[:3]==('TP1',1,0)
    assert value[3]==pytest.approx(fraction*.5+(1-fraction)*.25)


def test_future_target_cannot_replace_the_partial_plus_runner_expiry_mark():
    s=setup(max_bars=1)
    bars=candles([(100.,101.2,99.5,100.5),(100.5,105.,100.2,104.)])
    assert main._resolve_unsent_venue_setup(s,bars,now=3600.)[3]==pytest.approx(.2625)


def test_empty_terminal_bar_cannot_mark_the_remaining_runner_as_profit():
    s=setup(max_bars=2)
    bars=candles([(100.,101.2,99.5,100.5),(105.,105.,105.,105.)])
    bars['volume'][1]=0.
    assert main._resolve_unsent_venue_setup(s,bars,now=3600.) is None
