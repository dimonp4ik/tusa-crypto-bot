import pandas as pd
from stock_session_nov19_review import readiness_reasons


def test_paper_history_is_strictly_closed_before_entry_and_explains_negative_twenty():
    rows=[dict(symbol=str(i),entry_time=i*100,exit_time=i*100+50,net_r=-1,signed_return=-.01) for i in range(21)]
    r=readiness_reasons(pd.DataFrame(rows))
    assert r==dict(insufficient20_completed=20,paper_r_nonpositive=1)
    rows[19]['exit_time']=rows[20]['entry_time']
    assert readiness_reasons(pd.DataFrame(rows))==dict(insufficient20_completed=21)
