import pandas as pd
from stock_session_joint_nov19_audit import route,paper_portfolio,qualifying


def test_inverse_short_and_regular_long_share_economic_direction_cap():
    rows=[dict(symbol=s,entry_time=t,exit_time=100000,direction=d,family='open_range_retest',
        risk_pct=.01,signed_return=.01,net_r=1,sequential_outcome='TARGET',signal_strength=1)
        for s,t,d in [('QQQUSDT',0,'LONG'),('SOXSUSDT',900,'SHORT'),('SPYUSDT',1800,'LONG'),
            ('AMDUSDT',3700,'LONG'),('TSLAUSDT',5600,'LONG')]]
    admitted=paper_portfolio(pd.DataFrame(rows))
    assert admitted.symbol.tolist()==['QQQUSDT','SOXSUSDT','SPYUSDT','AMDUSDT']
    assert admitted.market_exposure.tolist()==[1,1,1,1]


def test_stock_prerequisite_cannot_be_mistaken_for_joint_goal():
    cases={'half_minimum':{'crypto':{'calendar_months':{
        m:dict(return_contribution_pct=dict(stocks=1,crypto=-1),stocks_closed=20,crypto_closed=30,cash_drawdown_pct=2)
        for m in ('2026-08','2026-09')}}}}
    assert qualifying(cases,'stocks')[0]
    assert not qualifying(cases,'crypto')[0]
    cases['half_minimum']['crypto']['calendar_months']['2026-09']['stocks_closed']=19
    assert not qualifying(cases,'stocks')[0]


def test_selection_route_uses_full_horizon_maturity_and_declared_universe():
    f=pd.DataFrame([dict(family='open_range_retest',entry_time=t,exit_time=e,origin=o)
        for t,e,o in [(100000,100900,'base'),(150000,150900,'base'),(100000,100900,'expansion')]])
    assert route(f,'session_union','base',0,160000).index.tolist()==[0]
    assert route(f,'session_union','expanded_current_snapshot',0,160000).index.tolist()==[0,2]
