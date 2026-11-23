"""Causal liquidity sweep, displacement and first block retest on closed bars."""

from dataclasses import dataclass
from math import isfinite
from statistics import median


@dataclass(frozen=True)
class Sequence:
    sweep: int
    equal_pivots: bool
    displacement: int | None = None
    retest: int | None = None
    block: int | None = None
    reclaim_level: float | None = None


def sequences(bars, direction, tick):
    """Use signed prices so the price and timing tests mirror in both directions."""
    if direction not in ('LONG', 'SHORT') or not isfinite(tick) or tick <= 0:
        raise ValueError('Invalid side or tick')
    if len(bars) < 28:
        return []
    if any(not all(isfinite(v) for v in row) or not 0 < row[2] <= min(row[0], row[3])
           <= max(row[0], row[3]) <= row[1] or row[4] < 0 for row in bars):
        raise ValueError('Invalid closed bars')
    sign = 1 if direction == 'LONG' else -1
    opens, highs, lows, closes, volumes = zip(*(
        (o, h, low, c, v) if sign == 1 else (-o, -low, -h, -c, v)
        for o, h, low, c, v in bars
    ))
    pivots = [i for i in range(2, len(bars) - 2)
              if volumes[i] > 0 and lows[i] < min(lows[i-2:i])
              and lows[i] <= min(lows[i+1:i+3])]
    found = []
    for s in range(max(20, len(bars) - 9), len(bars)):
        typical = median(highs[i] - lows[i] for i in range(s-20, s))
        typical_volume = median(volumes[s-20:s])
        if typical <= 0 or typical_volume <= 0 or volumes[s] < typical_volume:
            continue
        tolerance = max(tick, .20 * typical)
        breach = max(tick, .10 * typical)
        reclaim = max(tick, .05 * typical)
        prior = [i for i in pivots if max(2, s-20) <= i <= s-3]
        live = [i for i in prior if lows[i] - lows[s] >= breach
                and closes[s] - lows[i] >= reclaim
                and min(lows[i+1:s]) >= lows[i] - tick]
        if not live or closes[s] - lows[s] < .6 * (highs[s] - lows[s]):
            continue
        equal = any(earlier <= recent-4 and abs(lows[earlier]-lows[recent]) <= tolerance
                    and min(lows[earlier+1:s]) >= min(lows[earlier], lows[recent])-tick
                    for recent in live for earlier in prior)
        structure = max(highs[s-3:s])
        reclaimed = max(lows[i] for i in live)
        displacement = None
        for d in range(s+1, min(s+4, len(bars))):
            if lows[d] < lows[s] or closes[d] < reclaimed-tick:
                break
            if (volumes[d] > 0 and closes[d] >= structure+reclaim
                    and closes[d]-opens[d] >= .65*typical
                    and closes[d]-lows[d] >= .65*(highs[d]-lows[d])):
                displacement = d
                break
        retest = block = None
        if displacement is not None:
            opposite = [i for i in range(s, displacement) if volumes[i] > 0 and closes[i] < opens[i]]
            if opposite:
                block = opposite[-1]
                near, far = opens[block], closes[block]
                # A close through the body or a second breach invalidates the zone.
                if min(closes[block+1:displacement+1]) >= far:
                    for r in range(displacement+1, min(displacement+5, len(bars))):
                        if lows[r] < lows[s] or closes[r] < far:
                            break
                        if volumes[r] > 0 and lows[r] <= near+tick:
                            if closes[r] >= near+reclaim and closes[r] > opens[r]:
                                retest = r
                            break
        found.append(Sequence(s, equal, displacement, retest, block, sign*reclaimed))
    return found


def state(bars, direction, tick):
    """The decision sees only patterns completed in the last two closed bars."""
    if bars is None:
        return None
    observations = sequences(bars, direction, tick)
    sign = 1 if direction == 'LONG' else -1
    edge = 2 if sign == 1 else 1
    fresh = len(bars)-2
    flags = set()
    for observation in observations:
        extreme = bars[observation.sweep][edge]
        later = bars[observation.sweep+1:]
        if any(sign*(b[edge]-extreme) < 0 for b in later):
            continue
        if sign*(bars[-1][3]-observation.reclaim_level) < -tick:
            continue
        if observation.sweep >= fresh:
            flags.add('sweep')
            if observation.equal_pivots:
                flags.add('equal_sweep')
        if observation.displacement is not None and observation.displacement >= fresh:
            flags.add('displacement')
            if observation.equal_pivots:
                flags.add('equal_displacement')
        if (observation.retest is not None and observation.retest >= fresh
                and sign*(bars[-1][3]-bars[observation.block][3]) >= 0):
            flags.add('block_retest')
    return flags
