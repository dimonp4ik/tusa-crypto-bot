import copy
import pytest
from crypto_peer_breadth_nov17 import allows,peer_features


def candles():
    out={}
    for symbol in ['OWN',*[f'P{i}' for i in range(7)]]:
        close=[100.+i*.1 for i in range(30)]
        out[symbol]=dict(time=[i*900 for i in range(30)],close=close,
                         high=[v+.2 for v in close],low=[v-.2 for v in close])
    return out


def test_own_symbol_and_future_candles_cannot_change_breadth():
    data=candles();expected=peer_features(data,'OWN',20*900)
    altered=copy.deepcopy(data)
    for symbol,row in altered.items():
        for i in range(30):
            if symbol=='OWN' or i>=20:
                for key in ('close','high','low'): row[key][i]*=100
    assert peer_features(altered,'OWN',20*900)==expected
    assert expected['peers']==7 and expected['up_1h']==1


def test_truncated_prefix_matches_full_data_and_stale_peers_are_excluded():
    data=candles();expected=peer_features(data,'OWN',20*900)
    prefix={s:{k:v[:20] for k,v in c.items()} for s,c in data.items()}
    assert peer_features(prefix,'OWN',20*900)==expected
    prefix['P0']['time'][-1]-=900
    assert peer_features(prefix,'OWN',20*900)['peers']==6
    prefix['P1']['time'][-1]-=900
    assert peer_features(prefix,'OWN',20*900)==dict(peers=5,known_at=20*900)


def test_family_reversal_can_survive_slow_opposition_after_fast_breadth_turns():
    f=dict(peers=7,up_1h=1.,down_1h=0.,median_1h_atr=1.,
           up_4h=0.,down_4h=1.,median_4h_atr=-2.)
    assert allows('family_context',dict(direction='LONG',family='range_reversion'),f)
    assert not allows('family_context',dict(direction='LONG',family='trend_pullback'),f)
    assert allows('opposition_1h',dict(direction='LONG'),f)
    assert not allows('opposition_4h',dict(direction='LONG'),f)
    assert allows('opposition_both',dict(direction='LONG'),f)


def test_short_rule_is_mirror_of_long_rule_and_missing_peers_refuse_only_variants():
    f=dict(peers=7,up_1h=1.,down_1h=0.,median_1h_atr=1.,
           up_4h=1.,down_4h=0.,median_4h_atr=2.)
    assert not allows('opposition_either',dict(direction='SHORT'),f)
    assert not allows('opposition_both',dict(direction='SHORT'),f)
    assert allows('trend_opposition_both',dict(direction='SHORT',family='range_reversion'),f)
    assert not allows('trend_opposition_both',dict(direction='SHORT',family='trend_pullback'),f)
    assert allows('opposition_either',dict(direction='LONG'),f)
    assert allows('baseline',{},dict(peers=0))
    assert not allows('agreement_1h',dict(direction='LONG'),dict(peers=0))
    with pytest.raises(ValueError): peer_features(candles(),'OWN',20*900+.1)
