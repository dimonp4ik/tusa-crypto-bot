"""Reconcile public trade prints and resolve two causal stop-amendment clocks."""
from bisect import bisect_right
from dataclasses import dataclass
import csv
import hashlib
import io
import math
from pathlib import Path
import zipfile

from src.okx_trader import round_to_tick


@dataclass(frozen=True, order=True)
class Trade:
    ms: int
    identity: int
    price: float
    contracts: float


def load_trade_tape(item):
    if not item['complete']:
        raise ValueError('trade catalogue is incomplete')
    records = {}
    for day in item['days'].values():
        path = Path(day['path'])
        if hashlib.sha256(path.read_bytes()).hexdigest() != day['sha256']:
            raise ValueError('trade ZIP digest mismatch')
        with zipfile.ZipFile(path) as archive:
            members = archive.infolist()
            if len(members) != 1 or not 0 < members[0].file_size <= 100*1024**2:
                raise ValueError('unexpected trade file members')
            with archive.open(members[0]) as handle:
                rows = csv.DictReader(io.TextIOWrapper(handle, encoding='utf-8-sig'))
                if not {'instrument_name','trade_id','price','size','created_time'} <= set(rows.fieldnames or []):
                    raise ValueError('unexpected trade schema')
                for row in rows:
                    if row['instrument_name'] != item['instrument']:
                        continue  # daily files contain all expiries of the family
                    t = Trade(int(row['created_time']), int(row['trade_id']), float(row['price']), float(row['size']))
                    if not day['utc_start_ms'] <= t.ms < day['utc_start_ms'] + 86_400_000:
                        raise ValueError('trade timestamp outside declared UTC+8 day')
                    if not all(math.isfinite(v) and v>0 for v in (t.price,t.contracts)):
                        raise ValueError('invalid tape price or contract size')
                    if t.identity in records and records[t.identity] != t:
                        raise ValueError('conflicting trade identity')
                    records[t.identity] = t
    tape = sorted(records.values())
    if not tape or any(b.identity<=a.identity for a,b in zip(tape,tape[1:])):
        raise ValueError('exact-contract trade sequence unavailable or out of order')
    return tape


def reconcile(tape, frame, *, start, end):
    """Check count-independent trade completeness using summed lots and OHLC."""
    bars = {}
    for trade in tape:
        at = trade.ms//900_000*900
        if not start <= at < end:
            continue
        seen = bars.setdefault(at,[trade.price,trade.price,trade.price,trade.price,[]])
        seen[1] = max(seen[1],trade.price)
        seen[2] = min(seen[2],trade.price)
        seen[3] = trade.price
        seen[4].append(trade.contracts)
    discrepancies = {}
    checked = set()
    for i, ts in enumerate(frame['time']):
        if not start <= ts < end:
            continue
        checked.add(ts)
        observed = bars.get(ts)
        volume = math.fsum(observed[4]) if observed else 0.
        if not math.isclose(volume,frame['volume_contracts'][i],rel_tol=1e-8,abs_tol=1e-8):
            discrepancies[ts] = 'volume'
        elif observed and any(not math.isclose(p,frame[k][i],rel_tol=1e-10,abs_tol=1e-10)
                              for k,p in zip(('open','high','low','close'),observed)):
            discrepancies[ts] = 'ohlc'
    if checked != set(range(start,end,900)):
        raise ValueError('15m benchmark does not cover requested tape interval')
    return discrepancies


def resolve_ticks(tape, times, *, signal, entered_at, entry_px, tick_size, amend_seconds,
                  runner_policy="be_tp1"):
    """Model exchange-side triggers, leaving order matching to the L2 replay."""
    if amend_seconds not in (60,900):
        raise ValueError('unknown stop amendment clock')
    if runner_policy not in ("be_tp1", "be_1r", "half_stop_then_1r"):
        raise ValueError('unknown runner protection policy')
    sign = 1 if signal['direction']=='LONG' else -1
    stock = signal['bot']=='stocks'
    original_stop = round_to_tick(signal['stop'],tick_size)
    limit = round_to_tick(signal['target'],tick_size)
    tp1 = round_to_tick(signal['entry']+.5*(signal['entry']-signal['stop']),tick_size)
    one_r = round_to_tick(2*signal['entry']-signal['stop'],tick_size)
    breakeven = round_to_tick(entry_px,tick_size)
    half_risk_stop = round_to_tick(entry_px-.5*(signal['entry']-signal['stop']),tick_size)
    if sign*(half_risk_stop-original_stop)<0:
        half_risk_stop=original_stop
    expiry = int(signal['epoch']*1000)+(64 if stock else 48)*900_000
    if not sign*original_stop < sign*entry_px < sign*limit or not entered_at < expiry:
        return dict(known=False,reason='invalid_entry_relative_to_bracket')
    if not stock and sign*entry_px >= sign*tp1:
        return dict(known=False,reason='entry_is_already_beyond_tp1')
    opened_tick = bisect_right(times, entered_at)
    expires_tick = bisect_right(times, expiry-1)
    bank = None
    activation = math.inf
    half_activation = math.inf
    period = amend_seconds*1000
    for index in range(opened_tick,expires_tick):
        event = tape[index]
        active = event.ms>=activation
        half_active = event.ms>=half_activation
        stop = breakeven if active else half_risk_stop if half_active else original_stop
        if sign*(event.price-stop)<=0:
            code = 'BE' if active else 'TP1_HALF_STOP' if half_active else 'SL_AFTER_TP1' if bank else 'SL'
            return dict(known=True,code=code,
                        end_ms=event.ms,trigger_price=event.price,bank=bank,be_level=breakeven)
        if sign*(event.price-limit)>=0:
            # A first print across both levels creates concurrent triggers.
            # Price the entire remainder as one market exit in this case.
            return dict(known=True,code='TARGET',end_ms=event.ms,trigger_price=event.price,
                        bank=bank,be_level=breakeven,
                        simultaneous_targets=(not stock and bank is None))
        if not stock and bank is None and sign*(event.price-tp1)>=0:
            bank = dict(ms=event.ms,price=event.price)
            if runner_policy == "be_tp1":
                activation = (event.ms//period+1)*period
            elif runner_policy == "half_stop_then_1r":
                half_activation = (event.ms//period+1)*period
        if not stock and runner_policy != "be_tp1" and activation == math.inf and sign*(event.price-one_r)>=0:
            activation = (event.ms//period+1)*period
    return dict(known=True,code='TIME',end_ms=expiry,trigger_price=None,bank=bank,be_level=breakeven)
