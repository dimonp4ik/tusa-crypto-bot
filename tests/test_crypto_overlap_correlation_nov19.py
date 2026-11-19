import numpy as np

from crypto_overlap_correlation_nov19 import ClosedCorrelations, replay


def candles(changes, volume=None):
    prices = np.exp(np.r_[0., np.cumsum(changes)])
    return dict(time=list(range(0, len(prices)*900, 900)), close=prices,
                volume=np.ones(len(prices)) if volume is None else volume)


def test_pair_estimate_uses_only_completed_trades_and_requires_real_overlap():
    a = np.random.default_rng(19).normal(0,.001,120)
    b = a.copy(); b[100:] = 10*a[100:][::-1]
    gate = ClosedCorrelations({'A':candles(a), 'B':candles(b)})
    assert gate.get('A','B',90_000) > .99
    volume = np.ones(121); volume[70:90] = 0
    gate = ClosedCorrelations({'A':candles(a), 'B':candles(b,volume)})
    assert gate.get('A','B',90_000) is None


def test_same_pair_correlation_controls_pnl_direction_for_overlapping_positions_only():
    class Proxy:
        def get(self, a, b, decision):
            return .7
    def row(symbol, ts, end, direction='LONG'):
        return dict(symbol=symbol, entry_time=ts, exit_time=end, direction=direction,
                    known=True, outcome='TP', source_priority=3, module_priority=999)
    records = [row('A',0,3000),row('B',900,4000),row('C',1800,5000,'SHORT'),row('B',4500,6000)]
    accepted, blocks, unknown = replay(records, Proxy(), 'corr_065')
    assert [(r['symbol'],r['entry_time']) for r in accepted] == [('A',0),('C',1800),('B',4500)]
    assert blocks == {'correlated':1} and unknown is None
