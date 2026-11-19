import numpy as np
import pandas as pd
from crypto_native_price_calibration_nov19 import calibrated_scores


def history():
    day=100*86400;rows=[]
    for j in range(250):
        t=day-10*86400+j*900
        rows.append(dict(symbol='A',entry_time=t,exit_time=t+900,direction='LONG',family='breakout',
            long_btc_regime='bear',raw_prediction=.003,runner_price_return=-.01))
    rows.append(dict(symbol='A',entry_time=day,exit_time=day+900,direction='LONG',family='breakout',
        long_btc_regime='bear',raw_prediction=.003,runner_price_return=.1))
    return pd.DataFrame(rows),day


def test_current_and_future_labels_do_not_change_todays_correction():
    frame,day=history();first,decisions=calibrated_scores(frame,calibration_start=day-20*86400)
    changed=frame.copy();changed.loc[changed.entry_time.ge(day),'runner_price_return']=-1e9
    second,_=calibrated_scores(changed,calibration_start=day-20*86400)
    assert first.iloc[-1].predicted_price_return<.003
    assert first.iloc[-1].predicted_price_return==second.iloc[-1].predicted_price_return
    assert decisions[-1]['last_history_exit']<day
    assert decisions[-1]['last_history_full_horizon']<day


def test_no_fit_period_or_unfinished_horizon_labels_enter_calibration():
    frame,day=history();before,_=calibrated_scores(frame,calibration_start=day)
    assert before.iloc[-1].predicted_price_return==.003
    quick=frame.copy();quick['entry_time']=day-900;quick['exit_time']=day-1
    # Distinct symbols avoid duplicate-candidate validation while all horizons
    # remain unfinished. These many fast known labels must not gain a vote.
    quick['symbol']=['A'+str(j) for j in range(len(quick))]
    now=frame.iloc[[-1]]
    result,decisions=calibrated_scores(pd.concat([quick,now],ignore_index=True),calibration_start=day-20*86400)
    assert result.iloc[-1].predicted_price_return==.003
    assert decisions[-1]['history_rows']==0
