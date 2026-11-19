import pytest

from okx_xperp_l2_entry_replay_nov19 import START, attempt, sample


def event(offset, action='update', bids=(), asks=()):
    return dict(instId='TEST-X', action=action, ts=START*1000+offset, bids=bids, asks=asks)


def opening():
    return event(10, 'snapshot', [[99.99, 3]], [[100.01, 2], [100.02, 3]])


def test_book_samples_use_preceding_state_and_apply_absolute_deletions():
    values, coverage = sample([opening(), event(20, asks=[[100.01, 0], [100.02, 4]])],
                              'TEST-X', [START*1000+x for x in (10, 11, 20, 21)])
    assert values[START*1000+10] is None
    assert values[START*1000+11]['asks'] == [(100.01, 2.), (100.02, 3.)]
    assert values[START*1000+20]['asks'] == [(100.01, 2.), (100.02, 3.)]
    assert values[START*1000+21]['asks'] == [(100.02, 4.)]
    assert coverage['snapshot'] == coverage['update'] == 1


def test_archive_rejects_other_contracts_and_nonmonotonic_updates():
    with pytest.raises(ValueError, match='instrument mismatch'):
        sample([opening(), dict(event(20), instId='OTHER')], 'TEST-X', [])
    with pytest.raises(ValueError, match='nonmonotonic'):
        sample([opening(), event(9)], 'TEST-X', [])


def test_entry_checks_actual_order_size_and_book_timestamp():
    book = dict(ts=1000, bids=[(99.99, 3)], asks=[(100.01, 3)])
    spec = dict(ctVal=1, lotSz=.01, minSz=.01)
    fitted = attempt(book, 1001, 'LONG', spec, 99., 102.)
    assert fitted['status'] == 'eligible'
    assert fitted['contracts'] == pytest.approx(.38)
    assert attempt(book, 4001, 'LONG', spec)['status'] == 'stale order book'
    assert attempt(book, 1001, 'LONG', spec, 100.02, 102.)['status'] == 'executable price outside signal bracket'
