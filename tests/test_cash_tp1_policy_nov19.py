import pytest
from cash_tp1_policy_nov19 import simulate
from lot_aware_tp1_audit import simulate as original


def row(symbol='A',start=0,end=100):
    return dict(symbol=symbol,bot='crypto',direction='LONG',entry_time=start,exit_time=end,
        entry=10,risk_pct=.01,sequential_banked=True,near_exit_time=start+10,sequential_runner_r=-1)


@pytest.mark.parametrize('pct',[0,5,25,50,75])
def test_plain_policy_keeps_prior_cash_engine_parity(pct):
    rows=[row('A'),row('B',21,120)];specs={s:dict(ctVal=1,lotSz=1,minSz=1) for s in ('A','B')}
    options=dict(capital=130.,tp1_pct=pct,slots=12,bot_slots=dict(crypto=4,stocks=8),simultaneous_writers=2,release_delay=900)
    old=original(rows,specs,**options);new=simulate(rows,specs,**options)
    for key in ('ending_capital','return_pct','cash_drawdown_pct','trades','pnl_by_bot','peak_positions'):
        assert new[key]==old[key]


def test_indivisible_tp1_exits_at_bank_time_and_releases_slot_after_delay():
    first=row('A');first['entry']=25 # 130*3%*10 /25 floors to one contract.
    second=row('B',21,200)
    specs={s:dict(ctVal=1,lotSz=1,minSz=1) for s in ('A','B')}
    options=dict(capital=130.,tp1_pct=50,slots=2,bot_slots=dict(crypto=1,stocks=1),
        simultaneous_writers=1,release_delay=10)
    plain=simulate([first,second],specs,**options)
    minimum=simulate([first,second],specs,**options,minimum_close=True)
    assert plain['trades']==1
    assert minimum['trades']==2
    assert minimum['ledger'][0]['full_tp1_close'] is True
    assert minimum['ledger'][0]['exit_time']==10
    assert minimum['ledger'][0]['model_exit_time']==100
    assert minimum['ledger'][0]['runner_pnl']==0
    assert minimum['ledger'][0]['bank_pnl']>0


def test_zero_tp1_keeps_whole_runner_even_in_minimum_mode():
    value=simulate([row()],{'A':dict(ctVal=1,lotSz=1,minSz=1)},tp1_pct=0,minimum_close=True)
    assert value['ledger'][0]['bank_pnl']==0
    assert value['ledger'][0]['full_tp1_close'] is False
