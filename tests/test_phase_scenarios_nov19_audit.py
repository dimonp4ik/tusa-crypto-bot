import pandas as pd
from phase_scenarios_nov19_audit import portfolio,cash_rows


def test_exact_stock_portfolio_function_runs_without_importing_sister_main():
    f=pd.DataFrame([dict(symbol=s,entry_time=100000.,exit_time=100900.,direction='LONG',
        family='trend_retest_confirm',signal_strength=1.,signed_return=.01,net_r=1.,sequential_outcome='TP_RUNNER')
        for s in ('A','B','C')])
    assert len(portfolio(f,'stocks'))==2


def test_cash_normalization_keeps_the_actual_tp1_price_return():
    f=pd.DataFrame([dict(bank_price_return=.005497,runner_price_return=.0175,risk_pct=.004,
        entry=100,sl=99.6,sequential_banked=True)])
    row=cash_rows(f,'crypto')[0]
    assert row['risk_pct']*.5==.005497
    assert abs(row['risk_pct']*row['sequential_runner_r']-.0175)<1e-12
    assert row['actual_risk_fraction']==.004
