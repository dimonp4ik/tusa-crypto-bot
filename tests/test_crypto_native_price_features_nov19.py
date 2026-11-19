import copy
import numpy as np
from crypto_native_price_features_nov19 import NativeContext,FEATURES


def datasets():
    n=200;t=[1779000000+j*900 for j in range(n)];close=[100+j*.01+(j%3)*.03 for j in range(n)]
    c=dict(time=t,close=close,open=[v-.01 for v in close],high=[v+.2 for v in close],
        low=[v-.2 for v in close],volume=[1.]*n)
    return {s:copy.deepcopy(c) for s in ('TEST','BTCUSDT','ETHUSDT','A','B','C')}


def entry():
    return dict(symbol='TEST',entry_time=1779000000+141*900,direction='LONG',family='breakout',risk_pct=.01)


def test_features_never_read_future_asset_peer_or_volume_values():
    source=datasets();before=NativeContext(source).row(entry());changed=copy.deepcopy(source)
    for c in changed.values():
        for k in ('open','high','low','close'):c[k][141:]=[10000.]*59
        c['volume'][141:]=[1e9]*59
    np.testing.assert_array_equal(before,NativeContext(changed).row(entry()))
    assert len(before)==len(FEATURES)


def test_past_outcomes_and_raw_candidate_future_columns_are_not_features():
    context=NativeContext(datasets());row=entry();before=context.row(row)
    row.update(exit_time=1e30,net_r=1e10,runner_price_return=999,is_frozen_candidate=True)
    np.testing.assert_array_equal(before,context.row(row))
    assert not set(FEATURES)&{'exit_time','net_r','runner_price_return','is_frozen_candidate'}


def test_leave_one_out_peers_exclude_the_traded_asset():
    context=NativeContext(datasets());before=context.row(entry())
    changed=datasets();changed['TEST']['close'][140]*=1.001
    after=NativeContext(changed).row(entry())
    for name in ('peer_count','peer_median4_atr','peer_median16_atr','peer_support4','peer_support16'):
        assert before[FEATURES.index(name)]==after[FEATURES.index(name)]
