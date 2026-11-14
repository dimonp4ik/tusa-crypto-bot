import pytest
from crypto_minute_be_parity_nov15 import aggregate_traded_minutes


def test_empty_minute_carry_is_not_a_traded_extreme_or_open():
    bars=[[110.,110.,110.,110.],[100.,102.,99.,101.],[101.,101.,101.,101.]]
    assert aggregate_traded_minutes(bars,[0.,2.,0.])==(100.,102.,99.,101.)


def test_all_empty_period_requires_flat_previous_close():
    assert aggregate_traded_minutes([[100.]*4]*15,[0.]*15)==(100.,)*4
    with pytest.raises(ValueError,match='non-flat'):
        aggregate_traded_minutes([[100.]*4,[101.]*4],[0.,0.])
