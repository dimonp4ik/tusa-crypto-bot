from datetime import datetime,timezone
import pytest

from account_cash_guard_nov17 import RealizedCashGuard
from lot_aware_tp1_audit import simulate


def trade(symbol,bot,start,end,r):
    return dict(symbol=symbol,bot=bot,direction='LONG',entry_time=start,exit_time=end,
                entry=10.,risk_pct=.02,sequential_banked=False,sequential_runner_r=r)


def test_shared_cash_loss_pauses_the_other_bot_and_outstanding_close_still_realizes():
    rows=[trade('a','crypto',1,10,-6),trade('b','stocks',2,20,1),trade('c','stocks',11,30,1)]
    specs={r['symbol']:dict(ctVal=1,lotSz=1,minSz=1) for r in rows}
    guard=RealizedCashGuard(capital=100)
    result=simulate(rows,specs,capital=100,slots=12,bot_slots=dict(crypto=4,stocks=8),
                    entry_guard=guard,capture_events=True)
    assert [r['symbol'] for r in result['ledger']]==['a','b']
    assert result['skipped']['equity_guard']==1
    assert result['cash_events'][-1]['bot']=='stocks'
    assert result['cash_events'][-1]['equity']==pytest.approx(97.)
    assert guard.states['stocks']['paused'] is True
    assert guard.first_drawdown_blocks['stocks']['time']==11


def test_guard_uses_only_cash_strictly_before_entry_and_events_exclude_slot_releases():
    rows=[trade('a','crypto',1,10,-6),trade('b','stocks',10,20,1)]
    specs={r['symbol']:dict(ctVal=1,lotSz=1,minSz=1) for r in rows}
    guard=RealizedCashGuard(capital=100)
    result=simulate(rows,specs,capital=100,slots=12,bot_slots=dict(crypto=4,stocks=8),
                    entry_guard=guard,capture_events=True,release_delay=900)
    assert result['trades']==2
    assert len(result['cash_events'])==2
    assert result['cash_events'][0]['time']==10
    assert result['ledger'][0]['exit_time']==10


def test_permanent_guard_does_not_reset_at_month_change_or_after_recovery():
    guard=RealizedCashGuard(capital=100)
    july=datetime(2026,7,31,tzinfo=timezone.utc).timestamp()
    august=datetime(2026,8,1,tzinfo=timezone.utc).timestamp()
    assert guard(bot='crypto',now=july,equity=100) is None
    assert guard(bot='crypto',now=july+1,equity=96)=='drawdown limit'
    assert guard(bot='crypto',now=august,equity=110)=='drawdown limit'


def test_default_replay_and_always_allowed_guard_have_identical_legacy_results():
    rows=[trade('a','crypto',1,10,1),trade('b','stocks',11,20,-1)]
    specs={r['symbol']:dict(ctVal=1,lotSz=1,minSz=1) for r in rows}
    base=simulate(rows,specs,capital=100,slots=12,bot_slots=dict(crypto=4,stocks=8))
    assert base==simulate(rows,specs,capital=100,slots=12,bot_slots=dict(crypto=4,stocks=8),
                           entry_guard=lambda **kwargs:None)
