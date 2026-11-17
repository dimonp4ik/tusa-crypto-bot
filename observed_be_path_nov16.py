"""Public-price path scenarios, never a reconstruction of private fills."""
import math


class PricePath:
    def __init__(self, row, mode):
        self.sign = 1 if row['direction'] == 'LONG' else -1
        if row['direction'] not in ('LONG', 'SHORT'):
            raise ValueError('invalid direction')
        self.entry, self.stop, self.tp1, self.tp2 = (float(row[k]) for k in ('entry_price', 'sl', 'tp1', 'tp2'))
        self.risk = self.sign * (self.entry - self.stop)
        if self.risk <= 0 or self.sign * (self.tp1 - self.entry) <= 0 or self.sign * (self.tp2 - self.tp1) < 0:
            raise ValueError('invalid public-price bracket')
        self.mode = mode
        self.banked = False
        self.public_tp1_at = None
        self.mark = None
        self.mark_known_by = None
        self.event = None
        self.be_at = math.inf
        self.max_favorable_r = 0.
        self.max_adverse_r = 0.
        self.bar_seconds = int(row['strategy_bar_seconds'])
        observed = row.get('tp1_hit_at')
        if observed is not None and observed < row['opened_at']:
            raise ValueError('TP1 observation predates publication')
        if mode in ('observed_tp1', 'observed_tp1_plus_60'):
            if observed is not None:
                self.be_at = float(observed) + (60 if mode.endswith('plus_60') else 0)
        elif mode not in ('next_strategy_bar', 'next_minute'):
            raise ValueError('unknown public-price scenario')

    def needs_detail(self, bar):
        if self.event is not None or bar['volume'] == 0:
            return False
        stop = self.entry if self.be_at < bar['time'] + bar['seconds'] else self.stop
        if self.sign > 0:
            return bar['low'] <= stop or bar['high'] >= self.tp2 or (not self.banked and bar['high'] >= self.tp1)
        return bar['high'] >= stop or bar['low'] <= self.tp2 or (not self.banked and bar['low'] <= self.tp1)

    def consume(self, timestamp, price, *, known_by=None):
        if self.event is not None:
            return
        price = float(price)
        if not math.isfinite(price) or price <= 0:
            raise ValueError('invalid public price')
        self._observe_price(price)
        self.mark, self.mark_known_by = price, timestamp if known_by is None else known_by
        be_active = timestamp >= self.be_at
        stop = self.entry if be_active else self.stop
        if self.sign * (price - stop) <= 0:
            self.event = dict(kind='BE_TRIGGER' if be_active else 'TP1_STOP' if self.banked else 'SL',
                              time=timestamp, price=price, price_kind='first_public_trade_beyond_stop')
            return
        if not self.banked and self.sign * (price - self.tp1) >= 0:
            self.banked = True
            self.public_tp1_at = timestamp
            if self.mode in ('next_strategy_bar', 'next_minute'):
                interval = self.bar_seconds if self.mode == 'next_strategy_bar' else 60
                self.be_at = (math.floor(timestamp / interval) + 1) * interval
        if self.sign * (price - self.tp2) >= 0:
            self.event = dict(kind='TP2', time=timestamp, price=self.tp2,
                              public_crossing_price=price, price_kind='target_capped_at_trigger')

    def _observe_price(self, price):
        value = self.sign * (price - self.entry) / self.risk
        self.max_favorable_r = max(self.max_favorable_r, value)
        self.max_adverse_r = min(self.max_adverse_r, value)

    def consume_untouched_bar(self, bar):
        """Use extrema only in complete post-publication bars without an exit.

        A touched bar requires ordered trades: its high may follow its stop.
        Boundary bars must be clipped by the caller before reaching this path.
        """
        if self.event is not None or not bar['volume']:
            return
        if self.needs_detail(bar):
            raise ValueError('level-touch bar requires ordered public trades')
        for key in ('low', 'high'):
            price = float(bar[key])
            if not math.isfinite(price) or price <= 0:
                raise ValueError('invalid public range')
            self._observe_price(price)
        end = bar['time'] + bar['seconds']
        self.consume(end - 1e-6, bar['close'], known_by=end)

    def finish(self):
        event = self.event or dict(kind='MODEL_CLOSE_MARK' if self.mark is not None else 'NO_POST_PUBLICATION_TRADE',
                                  time=self.mark_known_by, price=self.mark, price_kind='last_public_trade_mark')
        runner = self.sign * (event['price'] - self.entry) / self.risk if event['price'] is not None else None
        first_r = self.sign * (self.tp1 - self.entry) / self.risk
        return dict(event, public_tp1_at=self.public_tp1_at,
            be_activation_scenario_at=self.be_at if math.isfinite(self.be_at) else None,
            tp1_before_exit=self.banked,
            whole_position_r=runner,
            nominal_five_percent_r=(.05 * first_r + .95 * runner if self.banked else runner) if runner is not None else None,
            max_favorable_r_before_exit=self.max_favorable_r,
            max_adverse_r_through_exit=self.max_adverse_r,
            private_amendment_and_fill_unverified=True)
