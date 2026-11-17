import copy
import pytest
from crypto_delayed_native_entry_nov17 import confirmation,choose_policy,opportunity_key


def data(side='LONG'):
    c=dict(time=[900,1800,2700,3600],open=[100.,100.,100.2,100.3],
        high=[100.3,100.5,100.6,105.],low=[99.8,99.9,100.,95.],
        close=[100.,100.2,100.3,100.])
    if side=='SHORT':
        c={k:([200-x for x in v] if k in ('open','high','low','close') else v) for k,v in c.items()}
        c['high'],c['low']=c['low'],c['high']
    return c,dict(direction=side,entry=100.,sl=98.5 if side=='LONG' else 101.5,entry_time=1800.)


@pytest.mark.parametrize('side',['LONG','SHORT'])
def test_known_close_confirmation_never_reads_next_open_or_future_range(side):
    c,row=data(side)
    before=confirmation(c,1,row,delay=1,rule='progress')
    assert before['entry_time']==2700 and before['sl']==row['sl']
    assert before['reason'] is None
    altered=copy.deepcopy(c)
    for k in ('open','high','low','close'):altered[k][2:]=[1e6,1e-6]
    assert confirmation(altered,1,row,delay=1,rule='progress')==before


def test_waiting_is_not_an_open_position_and_structural_failure_invalidates_entry():
    c,row=data();c['low'][1]=98.
    assert confirmation(c,1,row,delay=1,rule='body')['reason']=='structural_stop_failed_before_entry'
    c,row=data();c['high'][1]=104.
    assert confirmation(c,1,row,delay=1,rule='body')['reason']=='original_target_exhausted_before_entry'


def test_incomplete_and_noncontiguous_confirmation_are_unknown():
    c,row=data();c['time'][1]=1799
    assert confirmation(c,1,row,delay=1,rule='body')['reason']=='confirmation_gap'
    c,row=data();c={k:v[:2] for k,v in c.items()}
    assert confirmation(c,1,row,delay=2,rule='body')['reason']=='missing_confirmation'


def test_new_entry_risk_and_confirmed_direction_remain_actual_requirements():
    c,row=data();c['open'][1]=101.;c['high'][1]=102.;c['close'][1]=101.5
    assert confirmation(c,1,row,delay=1,rule='body')['reason']=='delayed_risk_outside_1_to_2pct'
    c,row=data();c['open'][1]=100.4
    assert confirmation(c,1,row,delay=1,rule='body')['reason']=='confirmation_rule_rejected'


def test_rotation_uses_only_mature_opportunities_and_unselected_rows_pay_zero():
    day=40*86400
    old=[dict(symbol=f'S{i}',direction='LONG',entry_time=day-2*86400,btc_regime='range') for i in range(100)]
    baseline={opportunity_key(r):dict(r,risk_pct=.01,net_r=.1) for r in old}
    rare={opportunity_key(r):dict(r,risk_pct=.01,net_r=.2) for r in old[:20]}
    variants=dict(baseline=baseline,rare=rare)
    choice=choose_policy(old,variants,day=day,regime='range')
    assert choice['policy']=='baseline'
    assert choice['scores']['rare']['mean']==pytest.approx(.0004)
    recent=dict(symbol='FUTURE',direction='LONG',entry_time=day-900,btc_regime='range')
    variants['rare'][opportunity_key(recent)]=dict(recent,risk_pct=100.,net_r=1e20)
    assert choose_policy(old+[recent],variants,day=day,regime='range')==choice
