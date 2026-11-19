import pandas as pd
import pytest
from liquidity_trap_joint_nov19_audit import select_rows,attach_outcome,with_exact_tick


def test_missing_legacy_tick_is_filled_only_from_exact_contract():
    spec=dict(instId='A-USD_UM_XPERP-310404',lotSz='1')
    instruments=[dict(instId=spec['instId'],tickSz='0.01'),dict(instId='A-USD_UM_XPERP-310405',tickSz='1')]
    assert with_exact_tick(spec,instruments)['tickSz']=='0.01'
    assert 'tickSz' not in spec
    with pytest.raises(ValueError,match='missing_or_duplicate'):
        with_exact_tick(spec,instruments[1:])
    with pytest.raises(ValueError,match='snapshot_changed'):
        with_exact_tick(dict(spec,tickSz='0.1'),instruments)


def test_route_cohort_purges_unmatured_fast_losses_and_future_or_other_universe():
    cutoff=10000000.;start=cutoff-1000000
    frame=pd.DataFrame([dict(family='failed_break_accept',entry_time=t,exit_time=e,origin=origin)
        for t,e,origin in [(cutoff-40*900,cutoff-39*900,'base'),
            (cutoff-900,cutoff-1,'base'),(cutoff+900,cutoff+1800,'base'),
            (cutoff-40*900,cutoff-39*900,'expansion')]])
    assert select_rows(frame,'new_union','base',start,cutoff).index.tolist()==[0]
    assert select_rows(frame,'new_union','expanded_current_snapshot',start,cutoff).index.tolist()==[0,3]


def test_unknown_price_path_is_retained_as_unknown_instead_of_a_flat_profit():
    c=dict(time=[i*900 for i in range(40)],open=[100.]*40,high=[100.1]*40,low=[99.9]*40,
        close=[100.]*40,volume=[0.]*40)
    row=dict(entry_index=0,entry_time=0,direction='LONG',entry=100.,sl=99.5,tp1=100.5,tp2=101.,
        expire_at=None,risk_pct=.005)
    out=attach_outcome(row,c,True)
    assert out['known'] is False
    assert out['unknown_reason']=='no_positive_volume_after_entry'
    assert 'net_r' not in out
