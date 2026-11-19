"""Research-only runner paths; new stops participate from the next candle."""
import math


def replay(candle, index, *, direction, entry, stop, target_r=2., activation_r=.5,
           trail_distance_r=None, stress=False, bars=48):
    if direction not in ('LONG', 'SHORT'):
        raise ValueError('unknown direction')
    if not all(math.isfinite(v) for v in (entry, stop, target_r, activation_r)):
        raise ValueError('nonfinite exit geometry')
    sign = 1 if direction == 'LONG' else -1
    risk = sign * (entry - stop)
    if min(entry, stop, risk) <= 0 or target_r <= .5 or activation_r < .5:
        raise ValueError('invalid exit geometry')
    if trail_distance_r is not None and (
            not math.isfinite(trail_distance_r) or not 0 < trail_distance_r <= 2 or target_r <= 2):
        raise ValueError('trailing starts after 2R and requires a further target')
    if index < 0 or bars <= 0 or index + bars > len(candle['time']):
        raise ValueError('incomplete runner horizon')
    times = candle['time'][index:index + bars]
    if any(b - a != 900 for a, b in zip(times, times[1:])):
        raise ValueError('noncontiguous runner horizon')
    banked = protected = False
    first_time = None
    trail = None
    best_r = 0.
    for i in range(index, index + bars):
        t = int(candle['time'][i]) + 900
        o, h, l, c = (float(candle[k][i]) for k in ('open', 'high', 'low', 'close'))
        if not all(math.isfinite(v) and v > 0 for v in (o, h, l, c)) or not l <= min(o, c) <= max(o, c) <= h:
            raise ValueError('invalid OHLC')
        level = entry if protected else stop
        if trail is not None:
            level = max(level, trail) if sign > 0 else min(level, trail)
        stop_hit = l <= level if sign > 0 else h >= level
        if stop_hit:
            fill = min(o, level) if sign > 0 else max(o, level)
            kind = 'TRAIL' if trail is not None else 'LOCK' if protected else 'TP1_STOP' if banked else 'SL'
            return dict(outcome=kind, runner_r=sign * (fill - entry) / risk,
                        banked=banked, exit_time=t, near_exit_time=first_time)
        favorable_r = sign * ((h if sign > 0 else l) - entry) / risk
        first_target = entry + sign * risk * .5
        activation = entry + sign * risk * activation_r
        target = entry + sign * risk * target_r
        first_hit = h >= first_target if sign > 0 else l <= first_target
        activation_hit = h >= activation if sign > 0 else l <= activation
        target_hit = h >= target if sign > 0 else l <= target
        if first_hit and first_time is None:
            first_time = t
        if stress and not protected and (banked or first_hit) and activation_hit:
            touch = l <= entry if sign > 0 else h >= entry
            if touch:
                return dict(outcome='LOCK_ACTIVATION_AMBIGUOUS', runner_r=0., banked=True,
                            exit_time=t, near_exit_time=first_time)
        best_r = max(best_r, favorable_r)
        next_trail = None
        if trail_distance_r is not None and best_r >= 2:
            next_trail = entry + sign * risk * (best_r - trail_distance_r)
            improves = sign * (next_trail - level) > 0
            touch = l <= next_trail if sign > 0 else h >= next_trail
            if stress and improves and touch:
                return dict(outcome='TRAIL_ACTIVATION_AMBIGUOUS',
                    runner_r=sign * (next_trail - entry) / risk, banked=True,
                    exit_time=t, near_exit_time=first_time)
        if target_hit:
            return dict(outcome='TP_RUNNER', runner_r=target_r, banked=True,
                        exit_time=t, near_exit_time=first_time)
        banked = banked or first_hit
        if banked and activation_hit:
            protected = True
        # Never use a low/high that preceded the newly observed extreme as
        # proof of execution at its newly computed trailing stop.
        if next_trail is not None:
            crossed_at_close = c <= next_trail if sign > 0 else c >= next_trail
            if crossed_at_close:
                return dict(outcome='TRAIL_MARK_ALREADY_CROSSED', runner_r=sign * (c - entry) / risk,
                    banked=banked, exit_time=t, near_exit_time=first_time)
            trail = (next_trail if trail is None else
                     max(trail, next_trail) if sign > 0 else min(trail, next_trail))
    return dict(outcome='TP_EXPIRED' if banked else 'EXPIRED',
        runner_r=sign * (c - entry) / risk, banked=banked,
        exit_time=t, near_exit_time=first_time)
