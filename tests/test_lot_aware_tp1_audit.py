import pytest
from lot_aware_tp1_audit import simulate


@pytest.mark.parametrize("field,value", [("sequential_runner_r",float("nan")),("exit_time",-1),("entry",0)])
def test_invalid_trade_cannot_produce_a_performance_report(field, value):
    row = dict(symbol="LINK", direction="LONG", entry_time=0, exit_time=20,
        entry=10, risk_pct=.1, sequential_banked=False, near_exit_time=10,
        sequential_runner_r=-1)
    row[field] = value
    with pytest.raises(ValueError, match="invalid trade inputs"):
        simulate([row], {"LINK":dict(ctVal=1,lotSz=1,minSz=1)})


def test_three_contract_position_five_percent_zero_fifty_percent_one_contract():
    specs = {"LINK": dict(ctVal=1, lotSz=1, minSz=1)}
    rows = [dict(symbol="LINK", direction="LONG", entry_time=0, exit_time=20,
        entry=10, risk_pct=.1, sequential_banked=True, near_exit_time=10,
        sequential_runner_r=0)]
    five = simulate(rows, specs, capital=100, tp1_pct=5)
    fifty = simulate(rows, specs, capital=100, tp1_pct=50)
    assert five["ending_capital"] == 100
    assert five["tp1_hits_without_bank"] == 1
    assert fifty["ledger"][0]["contracts"] == 3
    assert fifty["ledger"][0]["actual_partial_pct"] == pytest.approx(100/3)
    assert fifty["ending_capital"] == 100.5


def test_partial_cash_affects_later_entry_only_after_its_timestamp():
    specs = {k: dict(ctVal=1, lotSz=.1, minSz=.1) for k in ("A", "B")}
    a = dict(symbol="A", direction="LONG", entry_time=0, exit_time=30,
        entry=10, risk_pct=.2, sequential_banked=True, near_exit_time=10, sequential_runner_r=0)
    b = dict(a, symbol="B", entry_time=10, near_exit_time=20)
    tied = simulate([a, b], specs, capital=100, margin_fraction=.1, tp1_pct=50)
    later = simulate([a, dict(b, entry_time=11)], specs,
                     capital=100, margin_fraction=.1, tp1_pct=50)
    assert tied["ledger"][1]["contracts"] == 10
    assert later["ledger"][1]["contracts"] == 10.5


def test_slot_is_not_freed_by_partial_exit():
    specs = {k: dict(ctVal=1, lotSz=1, minSz=1) for k in ("A", "B")}
    a = dict(symbol="A", direction="LONG", entry_time=0, exit_time=30,
        entry=10, risk_pct=.1, sequential_banked=True, near_exit_time=10, sequential_runner_r=0)
    b = dict(a, symbol="B", entry_time=15, near_exit_time=20)
    result = simulate([a,b], specs, capital=100, tp1_pct=50, slots=1)
    assert result["trades"] == 1
    assert result["skipped"]["slots"] == 1


def test_shared_local_slots_stay_reserved_after_close_until_release():
    def trade(symbol,bot,start,end):
        return dict(symbol=symbol,bot=bot,direction='LONG',entry_time=start,exit_time=end,
            entry=10,risk_pct=.1,sequential_banked=False,sequential_runner_r=1)
    rows=[trade('a','crypto',0,10),trade('b','stocks',11,20),
          trade('c','crypto',12,30),trade('d','crypto',911,920)]
    specs={r['symbol']:dict(ctVal=1,lotSz=1,minSz=1) for r in rows}
    result=simulate(rows,specs,capital=100,slots=2,bot_slots=dict(crypto=1,stocks=1),release_delay=900)
    assert [r['symbol'] for r in result['ledger']]==['a','b','d']
    assert result['peak_positions']==2
    assert result['peak_by_bot']==dict(crypto=1,stocks=1)
    assert sum(result['pnl_by_bot'].values())==pytest.approx(result['ending_capital']-100,abs=1e-6)


def test_margin_headroom_reserves_for_both_writers():
    row=dict(symbol='a',bot='crypto',direction='LONG',entry_time=0,exit_time=10,
        entry=10,risk_pct=.1,sequential_banked=False,sequential_runner_r=1)
    result=simulate([row],{'a':dict(ctVal=1,lotSz=1,minSz=1)},capital=100,
        slots=12,bot_slots=dict(crypto=4,stocks=8),simultaneous_writers=2,margin_cap=.05)
    assert result['trades']==0
    assert result['skipped']['margin']==1
