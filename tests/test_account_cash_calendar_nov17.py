import pytest
from account_cash_guard_nov17 import calendar_months,stamp


def test_calendar_closures_cross_month_boundary_without_restarting_capital():
    result=dict(cash_events=[dict(time=stamp('2026-07-31'),bot='crypto',pnl=-4),
                             dict(time=stamp('2026-08-01'),bot='stocks',pnl=10)],
                ledger=[dict(entry_time=stamp('2026-07-30'))],
                guard_rejections=[dict(time=stamp('2026-08-02'))])
    months=calendar_months(result,capital=100)
    assert months['2026-07']['closing_cash']==96
    assert months['2026-08']['opening_cash']==96
    assert months['2026-08']['closing_cash']==106
    assert months['2026-08']['entries']==0
    assert months['2026-08']['return_contribution_pct']['stocks']==pytest.approx(1000/96)
    assert months['2026-08']['rejected_by_account_guard']==1
    assert months['2026-07']['cash_drawdown_pct']==4
