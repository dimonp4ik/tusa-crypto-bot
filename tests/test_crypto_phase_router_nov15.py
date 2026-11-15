import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch
import crypto_phase_router_nov15 as lab


def test_shock_has_priority_over_trend_and_missing_phase_is_rejected():
    f=pd.DataFrame(dict(asset_eff96=[.1,.4,.4],asset_trend_atr=[0,2,2],
        asset_volume_ratio=[1,1,3],asset_body_atr=[0,.5,2]))
    assert lab.phases(f).tolist()==['range','trend','shock']
    f.loc[0,'asset_eff96']=np.nan
    with pytest.raises(ValueError,match='unavailable'):
        lab.phases(f)


def test_selection_cannot_use_later_results():
    def value(policy,worst,later):
        return dict(policy=policy,selection=dict(n=100,worst_observed_month_pct=worst,return_pct=5),
            confirmation=dict(n=100,worst_observed_month_pct=worst,return_pct=5),later_return=later)
    a=value('bbb',0.,-100.);b=value('aaa',-1.,1000.)
    assert lab.choose([a,b]) is a
    a['confirmation']['n']=99
    assert lab.choose([a,b]) is b


def test_phase_expert_can_choose_a_different_family_without_future_labels():
    f=pd.DataFrame(dict(symbol=['BTCUSDT']*3,entry_time=[900,900,1800],
        family=['breakout','range_reversion','breakout'],asset_eff96=[.4,.4,.1],
        asset_trend_atr=[2,2,0],asset_volume_ratio=[1]*3,asset_body_atr=[.2]*3,
        net_r=[-100,100,-100]))
    with (patch.object(lab.runtime.base,'standardized',side_effect=lambda v,_:v.to_dict('records')),
          patch.object(lab.runtime,'runtime_gate',side_effect=lambda v:v)):
        rows=lab.routed(f,[2.,.1,2.],[.1,2.,.1],{'b':1.,'a':1.},'bab')
    assert [(r['entry_time'],r['family']) for r in rows]==[(900,'range_reversion'),(1800,'breakout')]


def test_frozen_expert_preserves_first_passing_family_instead_of_max_probability():
    f=pd.DataFrame(dict(symbol=['BTCUSDT']*2,entry_time=[900,900],
        family=['breakout','range_reversion'],asset_eff96=[.1]*2,
        asset_trend_atr=[0]*2,asset_volume_ratio=[1]*2,asset_body_atr=[.2]*2))
    with (patch.object(lab.runtime.base,'standardized',side_effect=lambda v,_:v.to_dict('records')),
          patch.object(lab.runtime,'runtime_gate',side_effect=lambda v:v)):
        rows=lab.routed(f,[0.]*2,[0.]*2,{'b':1.,'a':1.,'c':.3},'ccc',[.9,.4])
    assert len(rows)==1 and rows[0]['family']=='range_reversion'
