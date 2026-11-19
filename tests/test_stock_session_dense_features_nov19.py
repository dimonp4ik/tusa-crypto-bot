from copy import deepcopy
from stock_session_dense_features_nov19 import FeatureContext,proposals,FEATURES
from stock_session_entries_nov19 import SessionContext
from test_stock_session_entries_nov19 import source


def contexts(data):
    output={}
    for s in ('TESTUSDT','QQQUSDT','SPYUSDT','AUSDT','BUSDT','CUSDT'):
        c=deepcopy(data);output[s]=SessionContext({k:v for k,v in c.items() if k!='volume'},c)
    return output


def test_dense_proposals_have_nonvacuous_causal_features_and_no_outcome_inputs():
    c,j,put=source();cutoff=c['time'][j+5]+900
    cc=contexts(c);features=FeatureContext(cc)
    rows=[r for r in proposals(cc['TESTUSDT'],'TESTUSDT',dict(tickSz='.001'),features)
        if c['time'][j]<r['entry_time']<=cutoff]
    assert rows and len([k for k in rows[0] if k.startswith('feature__')])==len(FEATURES)
    changed=deepcopy(c)
    for k in ('open','high','low','close'):changed[k][j+6:]=[500.]*len(changed[k][j+6:])
    altered=contexts(changed)
    again=[r for r in proposals(altered['TESTUSDT'],'TESTUSDT',dict(tickSz='.001'),FeatureContext(altered))
        if c['time'][j]<r['entry_time']<=cutoff]
    assert rows==again
    row=rows[-1];point=cc['TESTUSDT'].at(row['entry_time'])
    assert features.row(row,point)==features.row(dict(row,net_r=100000,exit_time=1e20,known=False),point)


def test_dense_entry_requires_real_opening_and_current_trade_activity():
    c,j,_=source();c['volume'][j]=0;cc=contexts(c)
    rows=proposals(cc['TESTUSDT'],'TESTUSDT',dict(tickSz='.001'),FeatureContext(cc))
    end=c['time'][j]+6*3600
    assert not [r for r in rows if c['time'][j]<r['entry_time']<end]
