import numpy as np
from asset_entry_features_research import feature_table,FIELDS
from asset_entry_features_research import attach


def sample():
    close=[100+i*.01 for i in range(120)]
    return dict(time=[i*900 for i in range(120)],open=[p-.02 for p in close],
        high=[p+.1 for p in close],low=[p-.1 for p in close],close=close,volume=[100]*120)


def test_entry_bar_and_future_ohlcv_cannot_change_entry_features():
    c=sample();expected=feature_table(c).loc[110*900]
    assert expected.notna().all()
    for key in ('open','high','low','close','volume'):
        changed={k:list(v) for k,v in c.items()}
        changed[key][110:]=[999]*(len(c['time'])-110)
        np.testing.assert_array_equal(feature_table(changed).loc[110*900].to_numpy(),expected.to_numpy())


def test_missing_time_step_invalidates_history_instead_of_bridging_gap():
    c=sample();c['time'][100]+=1
    assert feature_table(c).loc[110*900].isna().all()


def test_insufficient_history_cannot_emit_fabricated_features():
    assert feature_table(sample()).iloc[96].isna().all()
    assert list(feature_table(sample()).columns)==list(FIELDS)


def test_bulk_attachment_preserves_duplicate_signal_rows_and_time_alignment(tmp_path):
    import pickle
    import pandas as pd
    c=sample();c['volume'][99]=200;c['volume'][109]=400
    path=tmp_path/'candles.pkl';path.write_bytes(pickle.dumps(c))
    signals=pd.DataFrame(dict(period=['p']*4,symbol=['A']*4,
        entry_time=[110*900,100*900,110*900,50*900],id=['a','b','c','d']),index=[9,2,7,4])
    result=attach(signals,{('p','A'):path})
    assert result.id.tolist()==['a','b','c']
    assert result.asset_volume_ratio.tolist()==[4,2,4]
    assert result.attrs['missing_feature_rows']==1
