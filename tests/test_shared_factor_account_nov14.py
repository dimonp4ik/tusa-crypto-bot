import pandas as pd
from shared_factor_account_nov14_audit import stock_ready_rows


def test_stock_readiness_uses_twenty_exits_strictly_before_entry():
    rows=[dict(symbol=f'h{i}',entry_time=0,exit_time=i+1,net_r=1) for i in range(20)]
    rows += [dict(symbol='equal',entry_time=20,exit_time=100,net_r=-999),
             dict(symbol='new',entry_time=21,exit_time=100,net_r=-999)]
    frame=pd.DataFrame(rows)
    assert stock_ready_rows(frame).symbol.tolist()==['new']
    frame.loc[frame.symbol.eq('equal'),'net_r']=999
    assert stock_ready_rows(frame).symbol.tolist()==['new']


def test_only_latest_twenty_outcomes_count_and_rejected_signals_remain_observable():
    rows=[dict(symbol=f'h{i}',entry_time=0,exit_time=i+1,net_r=1000 if i==0 else -1)
          for i in range(21)]
    rows += [dict(symbol='new',entry_time=30,exit_time=40,net_r=1000)]
    assert stock_ready_rows(pd.DataFrame(rows)).empty
