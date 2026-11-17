import copy
from crypto_causal_rule_rotation_nov17 import choose_policy

DAY=100*86400


def rows():
    peer=dict(peers=7,up_1h=1.,down_1h=0.,median_1h_atr=1.,
              up_4h=1.,down_4h=0.,median_4h_atr=1.)
    return [dict(entry_time=DAY-20000+i*10,exit_time=DAY-10000+i*10,
                 btc_regime='bull',direction='LONG',family='trend_pullback',
                 risk_pct=.01,net_r=1.,peer=peer) for i in range(110)]


def test_future_or_same_day_outcomes_cannot_change_daily_selection():
    history=rows();expected=choose_policy(history,day_start=DAY,regime='bull')
    later=[dict(history[0],entry_time=DAY-10,exit_time=DAY+i,net_r=-1000) for i in range(100)]
    assert choose_policy(history+later,day_start=DAY,regime='bull')==expected
    assert expected['completed']==110 and expected['policy']=='baseline'


def test_a_candidate_with_future_entry_is_never_treated_as_completed():
    history=rows();expected=choose_policy(history,day_start=DAY,regime='bull')
    bad=dict(history[0],entry_time=DAY+1,exit_time=DAY-1,net_r=-100)
    assert choose_policy(history+[bad],day_start=DAY,regime='bull')==expected


def test_negative_completed_payoffs_select_cash_without_mutating_evidence():
    history=rows()
    for r in history:r['net_r']=-1.
    original=copy.deepcopy(history)
    result=choose_policy(history,day_start=DAY,regime='bull')
    assert result['policy']=='cash' and result['scope']=='same_btc_regime'
    assert history==original


def test_missing_phase_evidence_uses_global_history_then_explicit_warmup():
    result=choose_policy(rows(),day_start=DAY,regime='bear')
    assert result['scope']=='all_btc_regimes' and result['policy']=='baseline'
    result=choose_policy(rows()[:99],day_start=DAY,regime='bull')
    assert result['scope']=='warmup_baseline' and result['completed']==99
