import pytest
from stock_dense_minutes_nov19 import reconcile


def source():
    parent=dict(time=[0],open=[100.],high=[102.],low=[100.],close=[101.],volume=[3.])
    minute=dict(time=list(range(0,900,60)),open=[90.]+[101.]*14,high=[90.]+[101.]*14,
        low=[90.]+[101.]*14,close=[90.]+[101.]*14,volume=[0.]*15)
    for key,v in dict(open=100.,high=102.,low=100.,close=101.,volume=3.).items():minute[key][1]=v
    return minute,parent


def test_only_real_trade_minutes_define_parent_ohlc_and_contract_volume_is_reconciled():
    m,p=source();assert reconcile(m,p,0,900)==dict(positive_parent_bars=1,zero_parent_bars=0)
    m['volume'][1]=2.9
    with pytest.raises(ValueError,match='contract_volume_mismatch'):reconcile(m,p,0,900)


def test_missing_or_conflicting_prices_do_not_create_a_verified_refinement():
    m,p=source();m['high'][1]=103
    with pytest.raises(ValueError,match='ohlc_mismatch'):reconcile(m,p,0,900)
    m,p=source();m['time'][2]=121
    with pytest.raises(ValueError,match='not_contiguous'):reconcile(m,p,0,900)
