import pytest
from download_native_expansion_nov19 import strict_instruments,parse


def spec(inst,rule='xperp',category='1'):
    return dict(instId=inst,ruleType=rule,state='live',ctType='linear',settleCcy='USD',instCategory=category)


def test_query_filter_is_not_trusted_and_premarket_or_inverse_are_not_xperp():
    rows=[spec('BTC-USD_UM_XPERP-310404'),spec('BTC-USD-261127',rule='normal'),
        spec('OPENAI-USD_UM_XPERP-310905',rule='pre_market',category='3')]
    assert list(strict_instruments(rows))==['BTC']


def test_trade_and_mark_history_schemas_preserve_volume_distinction():
    trade=[[0,'100','101','99','100.5','2','2','200','1']]
    mark=[[0,'100','101','99','100.5','1']]
    a,gap=parse(trade,'trade',0,900);b,_=parse(mark,'mark',0,900)
    assert a['volume']==[2.] and 'volume' not in b and not gap


def test_conflicting_revisions_and_unaligned_bars_never_make_a_clean_cache():
    row=[0,'100','101','99','100.5','1']
    with pytest.raises(ValueError,match='conflicting'):
        parse([row,[0,'100','101','99','100.6','1']],'mark',0,900)
    with pytest.raises(ValueError,match='unaligned'):
        parse([[1,*row[1:]]],'mark',0,900)
