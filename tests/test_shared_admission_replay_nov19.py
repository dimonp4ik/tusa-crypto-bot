from shared_admission_replay_nov19 import case


def row(symbol, bot, entry_time):
    return dict(symbol=symbol, bot=bot, entry_time=entry_time, exit_time=entry_time+900,
                direction='LONG', entry=10., risk_pct=.01, sequential_banked=False,
                sequential_runner_r=1.)


def specs(rows):
    return {r['symbol']:dict(ctVal=1, lotSz=1, minSz=1) for r in rows}


def test_tie_order_only_changes_owner_when_one_shared_slot_is_available():
    rows = [row('C','crypto',1_785_600_000), row('S','stocks',1_785_600_000)]
    for first, second in (('crypto','stocks'), ('stocks','crypto')):
        result = case(rows, specs(rows), priority=first, policy='half_minimum', external=11)
        assert result['admission']['accepted'] == {first:1,second:0}
        assert result['admission']['rejected'][second] == {'shared_slots':1}


def test_combined_slot_bucket_is_split_into_local_limit_and_instrument_occupied():
    start = 1_785_600_000
    rows = [row(f'C{i}','crypto',start) for i in range(5)]
    rows.append(row('C0','stocks',start+900))
    result = case(rows, specs(rows), priority='crypto', policy='half_minimum')
    assert result['peak_positions'] == 4
    assert result['admission']['rejected'] == {
        'crypto':{'local_slots':1}, 'stocks':{'instrument_occupied':1}}


def test_margin_pressure_and_unaffordable_lots_are_distinct_reasons():
    rows = [row('C','crypto',1_785_600_000)]
    result = case(rows, specs(rows), priority='crypto', policy='half_minimum', external_margin=.89)
    assert result['admission']['rejected']['crypto'] == {'shared_margin':1}
    expensive = {'C':dict(ctVal=10, lotSz=1,minSz=1)}
    result = case(rows, expensive, priority='crypto', policy='half_minimum')
    assert result['admission']['rejected']['crypto'] == {'minimum_lot':1}
