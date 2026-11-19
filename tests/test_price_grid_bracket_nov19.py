import pytest
from price_grid_bracket_nov19 import rounded_bracket
from src.okx_trader import round_to_tick


def test_price_levels_use_the_same_rounding_as_live_and_recheck_actual_risk():
    row=dict(direction='LONG',entry=100.,sl=99.449,tp1=100.579,tp2=101.859)
    out=rounded_bracket(row,{'tickSz':.01})
    assert out is not None
    assert out['sl']==round_to_tick(row['sl'],.01)
    assert out['tp1']==round_to_tick(row['tp1'],.01)
    assert out['risk_pct']==pytest.approx((100-out['sl'])/100)


def test_coarse_grid_cannot_turn_a_collapsed_target_into_a_profitable_path():
    row=dict(direction='LONG',entry=10.,sl=9.95,tp1=10.05,tp2=10.1)
    assert rounded_bracket(row,{'tickSz':1}) is None


def test_missing_tick_is_unknown_and_short_target_order_is_preserved():
    row=dict(direction='SHORT',entry=100.,sl=100.5,tp1=99.4,tp2=98.8)
    assert rounded_bracket(row,{'tickSz':.01}) is not None
    with pytest.raises(ValueError,match='missing_price_grid'):rounded_bracket(row,{})


def test_rounding_cannot_squeeze_stop_or_first_target_below_declared_floor():
    row=dict(direction='LONG',entry=100.01,sl=99.70997,tp1=100.51005,tp2=101.72)
    out=rounded_bracket(row,{'tickSz':.01})
    assert out is not None
    assert out['risk_pct']>=.003
    assert out['target1_fraction']>=.005
    for field in ('sl','tp1','tp2'):assert round_to_tick(out[field],.01)==out[field]
