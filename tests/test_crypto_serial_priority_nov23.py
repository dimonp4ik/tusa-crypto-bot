from crypto_serial_priority_nov23 import PACES, serial


def proposal(symbol, *, epoch=0):
    return dict(symbol=symbol, bot='crypto', direction='LONG', epoch=epoch, model_quality=.6,
                entry=100., stop=98., target=104., paper_reference_ready=False)


def book(stamp, *, wide=False):
    bid, ask = (99.8, 100.2) if wide else (99.99, 100.)
    return dict(ts=stamp - 5, bids=[(bid, 30)], asks=[(ask, 30)])


def test_serial_scan_retries_publication_and_queues_later_account_fill():
    rows = [proposal(symbol) for symbol in ('AAA', 'BBB', 'CCC', 'DDD')] + [proposal('EEE', epoch=3600)]
    quotes = {
        'AAA': {at: book(at, wide=True) for at in (11_100, 11_550)},
        'BBB': {11_650: book(11_650), 12_500: book(12_500, wide=True), 12_800: book(12_800)},
        'CCC': {at: book(at) for at in (12_000, 13_900)},
    }
    specs = {row['symbol']: dict(ctVal=1., lotSz=.01, minSz=.01) for row in rows}
    ticks = dict.fromkeys(specs, .01)
    result = list(serial(rows, quotes, specs, ticks, PACES['fast'], 'alphabetical'))
    assert [row['entry_status'] for row in result] == [
        'spread_at_publish', 'eligible', 'eligible', 'publication_rate_limit', 'publication_rate_limit',
    ]
    assert result[0]['publication_attempts'] == [11_100, 11_550]
    assert result[1]['execution_attempts'] == [12_500, 12_800]
    assert result[2]['publication_attempts'] == [12_000]
    assert result[2]['execution_attempts'] == [13_900]


def test_publication_budget_counts_even_when_account_rejects():
    rows = [proposal(symbol) for symbol in ('AAA', 'BBB', 'CCC')]
    quotes = {
        'AAA': {11_100: book(11_100), 11_950: book(11_950, wide=True),
                12_250: book(12_250, wide=True), 12_950: book(12_950, wide=True)},
        'BBB': {at: book(at) for at in (11_450, 14_050)},
    }
    specs = {row['symbol']: dict(ctVal=1., lotSz=.01, minSz=.01) for row in rows}
    result = list(serial(rows, quotes, specs, dict.fromkeys(specs, .01), PACES['fast'], 'alphabetical'))
    assert [row['entry_status'] for row in result] == ['execution_spread', 'eligible', 'publication_rate_limit']
    assert result[0]['published_ms'] == 11_350
    assert result[1]['execution_attempts'] == [14_050]
