import copy
import numpy as np
import pytest
from crypto_native_long_phase_nov19 import long_phase_features,LONG_FEATURES


def candles():
    t0=1779000300;n=450;close=[100+j*.01+(j%7)*.02 for j in range(n)]
    c=dict(time=[t0+j*900 for j in range(n)],open=close,close=close,
        high=[v+.2 for v in close],low=[v-.2 for v in close],volume=[1.]*n)
    return {s:copy.deepcopy(c) for s in ('BTCUSDT','ETHUSDT')}


def test_future_btc_and_eth_bars_never_change_long_phase():
    data=candles();stamp=data['BTCUSDT']['time'][400]
    before=long_phase_features(data,stamp,'LONG');changed=copy.deepcopy(data)
    for c in changed.values():
        for k in ('open','high','low','close'):c[k][400:]=[10000.]*50
    np.testing.assert_array_equal(before,long_phase_features(changed,stamp,'LONG'))
    assert len(before)==len(LONG_FEATURES)


def test_missing_closed_long_bar_is_unknown_instead_of_future_filled():
    data=candles();stamp=data['BTCUSDT']['time'][400]
    for key in data['ETHUSDT']:del data['ETHUSDT'][key][300]
    with pytest.raises(ValueError,match='missing_long_closed_context'):
        long_phase_features(data,stamp,'SHORT')
