import pickle
import pandas as pd
import pytest
from crypto_phase_breakeven_nov15 import LEVELS, attach_outcomes, policy_frame, prefix


def test_protection_uses_entry_phase_and_banks_only_after_first_target():
    frame=pd.DataFrame(dict(phase=['range','trend','shock']))
    for level in LEVELS:
        p=prefix(level)
        frame[p+'_outcome']=['test']*3
        frame[p+'_runner']=[level]*3
        frame[p+'_banked']=[True,False,True]
        frame[p+'_exit']=[100+level]*3
    result=policy_frame(frame,(.5,1.,1.25))
    assert result.protection_activation_r.tolist()==[.5,1.,1.25]
    assert result.net_r.tolist()==pytest.approx([.025+.95*.5,1.,.025+.95*1.25])
    assert result.exit_time.tolist()==[100.5,101.,101.25]
    with pytest.raises(ValueError,match='unknown protection'):
        policy_frame(frame,(.5,2.,1.))
    frame.loc[0,'phase']='not_a_phase'
    with pytest.raises(ValueError,match='unknown entry phase'):
        policy_frame(frame,(.5,1.,1.25))


def test_unknown_phase_keeps_original_protection_instead_of_removing_entry():
    frame=pd.DataFrame(dict(phase=['unknown']))
    for level in LEVELS:
        p=prefix(level)
        frame[p+'_outcome']=['test'];frame[p+'_runner']=[level]
        frame[p+'_banked']=[False];frame[p+'_exit']=[100+level]
    result=policy_frame(frame,(99.,99.,99.))
    assert len(result)==1
    assert result.protection_activation_r.iloc[0]==.5
    assert result.net_r.iloc[0]==.5


def test_label_cache_is_invalidated_when_candles_change_and_rejects_short_window(tmp_path):
    data=dict(time=list(range(0,48*900,900)),open=[100.]*48,high=[100.4]*48,
        low=[99.8]*48,close=[100.1]*48)
    source=tmp_path/'candles.pkl';cache=tmp_path/'labels.pkl'
    paths={('direct','TEST'):source}
    frame=pd.DataFrame([dict(period='direct',symbol='TEST',entry_time=0,direction='LONG',
        entry=100.,sl=99.,phase='range')])
    source.write_bytes(pickle.dumps(data))
    first,_=attach_outcomes(frame,paths,cache)
    assert first[prefix(.5)+'_runner'].iloc[0]==pytest.approx(.1)
    data['close'][-1]=100.3
    source.write_bytes(pickle.dumps(data))
    second,_=attach_outcomes(frame,paths,cache)
    assert second[prefix(.5)+'_runner'].iloc[0]==pytest.approx(.3)
    data={k:v[:-1] for k,v in data.items()}
    source.write_bytes(pickle.dumps(data))
    with pytest.raises(ValueError,match='incomplete runner window'):
        attach_outcomes(frame,paths,cache)


def test_partial_cash_time_is_derived_from_same_replay_not_a_different_raw_label(tmp_path):
    source=tmp_path/'candles.pkl'
    data=dict(time=list(range(0,48*900,900)),open=[100.]*48,high=[100.6]*48,
        low=[99.8]*48,close=[100.1]*48)
    source.write_bytes(pickle.dumps(data))
    frame=pd.DataFrame([dict(period='direct',symbol='TEST',entry_time=0,direction='LONG',
        entry=100.,sl=99.,phase='range',near_exit_time=99999)])
    result,_=attach_outcomes(frame,{('direct','TEST'):source},tmp_path/'labels.pkl',stress=True)
    assert result.near_exit_time.iloc[0]==900
    assert result[prefix(.5)+'_exit'].iloc[0]==900
