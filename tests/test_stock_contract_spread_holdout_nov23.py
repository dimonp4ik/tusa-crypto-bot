from stock_contract_spread_holdout_nov23 import same_prices


def test_correcting_volume_requires_matching_prices_and_closed_bar_times():
    frozen = dict(time=[0, 900], open=[100., 100.], high=[101., 102.], low=[99., 99.],
                  close=[100., 101.], volume=[0., 0.])
    confirmed = dict(frozen, volume=[.1, .2], volume_contracts=[.1, .2])
    assert same_prices(confirmed, frozen)
    assert not same_prices(dict(confirmed, time=[0, 1800]), frozen)
    assert not same_prices(dict(confirmed, high=[101., 102.01]), frozen)
