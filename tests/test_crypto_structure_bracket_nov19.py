import copy
import pytest
from crypto_structure_bracket_nov19 import stop_geometry,replay_fixed_targets
from crypto_runner_exit_path_nov16 import replay


def source():
    n=160
    return dict(time=[900*i for i in range(n)],open=[100.]*n,high=[100.2]*n,low=[99.8]*n,close=[100.]*n)


def test_structure_stop_never_uses_future_extrema():
    c=source();c['low'][98]=98.8
    row=dict(entry=100.,sl=99.,direction='LONG',entry_time=100*900)
    before=stop_geometry(c,100,row,'swing4_widen');changed=copy.deepcopy(c)
    changed['low'][100:]=[1.]*60
    assert stop_geometry(changed,100,row,'swing4_widen')==before
    assert before['stop']<99


def test_unconfirmed_pivot_is_not_available_at_entry():
    c=source();c['low'][99]=98.
    row=dict(entry=100.,sl=99.,direction='LONG',entry_time=100*900)
    assert stop_geometry(c,100,row,'confirmed_pivot_widen') is None


def test_wider_stop_keeps_the_original_take_profit_prices_and_cash_units():
    c=source();c['low'][100]=98.7;c['high'][101]=102.2
    result=replay_fixed_targets(c,100,direction='LONG',entry=100,stop=98.5,original_risk=1)
    assert result['outcome']=='TP_RUNNER'
    assert result['runner_r']==2 # Fixed price102, not stop-risk-scaled103.
    assert result['exit_time']==102*900


def test_stop_gap_is_a_larger_cash_loss_and_short_geometry_is_symmetric():
    c=source();c['open'][100]=98.;c['low'][100]=97.9;c['high'][100]=100.2
    result=replay_fixed_targets(c,100,direction='LONG',entry=100,stop=99,original_risk=1)
    assert result['runner_r']==-2
    mirror=source();mirror['open'][100]=102.;mirror['high'][100]=102.1;mirror['low'][100]=99.8
    result=replay_fixed_targets(mirror,100,direction='SHORT',entry=100,stop=101,original_risk=1)
    assert result['runner_r']==-2


def test_control_preserves_original_stop_subtraction_target_arithmetic():
    c=source();entry=133.3;stop=134.66000000000003
    for k in ('open','high','low','close'):c[k]=[entry]*160
    c['low'][101]=130.58;c['low'][102]=130.57
    old=replay(c,100,direction='SHORT',entry=entry,stop=stop)
    new=replay_fixed_targets(c,100,direction='SHORT',entry=entry,stop=stop,original_risk=abs(entry-stop))
    assert new==old
