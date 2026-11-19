from crypto_active_entry_runner_nov19_audit import activity_gate


def test_only_sixteen_closed_bars_are_used_and_future_mutation_changes_nothing():
    c=dict(time=[900*i for i in range(20)],volume=[1.]*20)
    assert activity_gate(c,16)
    c['volume'][16:]=[0.,float('nan'),-100.,1e20]
    assert activity_gate(c,16)


def test_threshold_and_last_print_are_both_required():
    c=dict(time=[900*i for i in range(16)],volume=[0.]*4+[1.]*12)
    assert activity_gate(c,16)
    c['volume'][4]=0
    assert not activity_gate(c,16)
    c['volume']=[1.]*15+[0.]
    assert not activity_gate(c,16)


def test_invalid_or_missing_history_cannot_pass():
    c=dict(time=[900*i for i in range(16)],volume=[1.]*16)
    assert not activity_gate(c,15)
    c['volume'][0]=float('nan')
    assert not activity_gate(c,16)
