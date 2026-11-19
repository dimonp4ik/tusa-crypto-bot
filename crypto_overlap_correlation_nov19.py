"""Compare fixed concurrent correlation caps on the same checked native entries."""
import argparse
from collections import Counter, deque
from datetime import datetime, timezone
from functools import lru_cache
import hashlib
import heapq
import json
from pathlib import Path

import numpy as np

from account_cash_guard_nov17_audit import runtime_gate
from crypto_active_entry_runner_nov19_audit import activity_gate
from crypto_minimum_bank_runner_nov19_audit import checked_row, cash_result, CLIENTS
from crypto_native_price_pool_nov19 import load_pool, ROOT, MANIFEST, META
from liquidity_trap_joint_nov19_audit import load_frame
from account_cash_guard_nov17 import stamp

START = stamp('2026-08-01')
SPLIT = stamp('2026-10-01')
HORIZON = 48 * 900
POLICIES = ('baseline', 'one_direction', 'corr_065', 'corr_080')


class ClosedCorrelations:
    def __init__(self, candles):
        self.data = {}
        for symbol, bars in candles.items():
            times = np.asarray(bars['time'], dtype=np.int64)
            if (np.diff(times) != 900).any():
                raise ValueError('correlation input requires continuous candles')
            prices = np.asarray(bars['close'], dtype=float)
            volume = np.asarray(bars['volume'], dtype=float)
            changes = np.r_[np.nan, np.diff(np.log(prices))]
            active = (volume > 0) & np.r_[False, volume[:-1] > 0]
            changes[~active] = np.nan
            self.data[symbol] = (int(times[0]), changes)

    @lru_cache(maxsize=8192)
    def get(self, a, b, decision):
        if decision % 900:
            raise ValueError('correlation decision must be on the 15m clock')
        values = []
        for symbol in (a, b):
            start, changes = self.data[symbol]
            end = (decision-start)//900
            if end < 96 or end > len(changes):
                return None
            values.append(changes[end-96:end])
        x, y = values
        valid = np.isfinite(x) & np.isfinite(y)
        if int(valid.sum()) < 80:
            return None
        x, y = x[valid], y[valid]
        xc, yc = x-x.mean(), y-y.mean()
        scale = np.linalg.norm(xc) * np.linalg.norm(yc)
        return float(np.clip(np.dot(xc, yc) / scale, -1., 1.)) if scale > 0 else None


def replay(rows, correlations, policy):
    pending = []
    opened = {}
    last = {}
    accepted_times = deque()
    output = []
    blocks = Counter()
    day = None
    streak = 0
    for row in sorted(rows, key=lambda r: (r['entry_time'], r['source_priority'], r['symbol'], r['module_priority'])):
        now = int(row['entry_time'])
        day_key = now // 86400
        if day_key != day:
            day, streak = day_key, 0
        while pending and pending[0][0] < now:
            _, _, done = heapq.heappop(pending)
            opened.pop(done['symbol'])
            if int(done['exit_time'])//86400 == day:
                streak = streak + 1 if done['outcome']=='SL' else 0
        while accepted_times and accepted_times[0] <= now-3600:
            accepted_times.popleft()
        symbol, direction = row['symbol'], row['direction']
        key = symbol, direction
        same_dir = sum(r['direction']==direction for r in opened.values())
        if (streak>=3 or symbol in opened or len(accepted_times)>=2
                or now-last.get(key, -10**12)<10800 or same_dir>=3):
            continue
        if policy == 'one_direction' and same_dir:
            blocks['one_direction'] += 1
            continue
        if policy in ('corr_065', 'corr_080'):
            threshold = .65 if policy == 'corr_065' else .80
            blocked = False
            for other, active in opened.items():
                value = correlations.get(*sorted((symbol, other)), now)
                if value is None:
                    blocks['missing_pair_observations'] += 1
                    continue
                signed = value if direction == active['direction'] else -value
                if signed >= threshold:
                    blocked = True
                    break
            if blocked:
                blocks['correlated'] += 1
                continue
        if not row['known']:
            return None, dict(blocks), (row['symbol'], row['entry_time'], row['unknown_reason'])
        opened[symbol] = row
        last[key] = now
        accepted_times.append(now)
        output.append(row)
        heapq.heappush(pending, (row['exit_time'], len(output), row))
    return output, dict(blocks), None


def clusters(rows):
    stops = deque()
    max_streak = streak = 0
    pairs = 0
    for row in sorted(rows, key=lambda r: (r['exit_time'], r['entry_time'], r['symbol'])):
        time = row['exit_time']
        while stops and stops[0] <= time-3600:
            stops.popleft()
        is_stop = row['outcome']=='SL'
        streak = streak+1 if is_stop else 0
        max_streak = max(max_streak, streak)
        if is_stop:
            pairs += len(stops)
            stops.append(time)
    return dict(paper_stops=sum(r['outcome']=='SL' for r in rows), paper_stop_pairs_within_hour=pairs,
                max_paper_stop_streak=max_streak)


def evaluate(rows, corr, specs, policy, start, end=None):
    cohort = [r for r in rows if r['entry_time'] >= start and
              (end is None or r['entry_time']+HORIZON < end)]
    accepted, blocks, unknown = replay(cohort, corr, policy)
    if unknown:
        return dict(status='UNSCORABLE_ADMITTED_UNKNOWN', first_unknown=unknown, blocks=blocks)
    if policy == 'baseline' and accepted != runtime_gate(cohort, latch_loss_pause=False):
        raise ValueError('correlation control diverged from existing admission replay')
    cases = {client: cash_result(accepted, specs, client) for client in CLIENTS}
    for case in cases.values():
        case['calendar_months'] = {m:v for m,v in case['calendar_months'].items()
                                   if stamp(m+'-01') >= start and (end is None or stamp(m+'-01') < end)}
    return dict(status='RESOLVED_PROXY_CASH', candidates=len(cohort), admitted=len(accepted),
                blocks=blocks, clusters=clusters(accepted), cash=cases)


def qualifies(candidate, baseline):
    if candidate['status'] != 'RESOLVED_PROXY_CASH' or baseline['status'] != 'RESOLVED_PROXY_CASH':
        return False
    for client in CLIENTS:
        case, control = candidate['cash'][client], baseline['cash'][client]
        if case['summary']['cash_drawdown_pct'] >= control['summary']['cash_drawdown_pct']:
            return False
        for month in ('2026-08', '2026-09'):
            current = case['calendar_months'].get(month)
            previous = control['calendar_months'].get(month)
            if not current or not previous or current['closed'] < 20:
                return False
            if current['return_pct'] <= max(0., previous['return_pct']):
                return False
    return candidate['clusters']['paper_stop_pairs_within_hour'] < baseline['clusters']['paper_stop_pairs_within_hour']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--declare-only', action='store_true')
    args = parser.parse_args()
    specs_path = ROOT/'shared_xperp_specs_nov14.json'
    paths = [Path(__file__).resolve(), MANIFEST, META, specs_path,
             Path('account_cash_guard_nov17_audit.py'), Path('account_cash_guard_nov17.py'),
             Path('crypto_native_price_pool_nov19.py'), Path('crypto_active_entry_runner_nov19_audit.py'),
             Path('crypto_minimum_bank_runner_nov19_audit.py'), Path('banked_runner_path_nov19.py'),
             Path('cash_tp1_policy_nov19.py'), Path('src/tp1_plan.py'), Path('src/okx_trader.py')]
    hashes = {str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    protocol = ROOT/'crypto_overlap_correlation_nov19_protocol.json'
    if protocol.exists():
        if json.loads(protocol.read_bytes())['sources'] != hashes:
            raise ValueError('frozen_correlation_protocol_changed')
    else:
        protocol.write_text(json.dumps(dict(sources=hashes, declared_at=datetime.now(timezone.utc).isoformat(),
            policies=POLICIES, correlation='signed Pearson of 96 closed returns; >=80 paired active; missing cannot qualify as correlation',
            selection='August/September with full horizon purge, each month positive and better than control at both TP1 modes, >=20 each month, lower cash DD and fewer stop pairs',
            later='test one selection winner without refit or reuse of selection-period portfolio state',
            cash='one $130 account, 3% margin and 10x, caps 4 crypto/12 shared and 90%, stock remains PAPER',
            limitations='Known-close quotes and checked 5bp exit proxy, no historical bid/ask or private fills. Previously inspected market history.'), indent=2),encoding='utf-8')
    if args.declare_only:
        return
    frame, _ = load_pool()
    manifest = json.loads(MANIFEST.read_bytes())['symbols']
    candles = {s:load_frame(Path(x['path']), x['sha256']) for s,x in manifest.items()}
    indices = {s:{int(t):i for i,t in enumerate(c['time'])} for s,c in candles.items()}
    rows = []
    for row in frame[frame.is_frozen_candidate & frame.entry_time.ge(START)].to_dict('records'):
        s = row['symbol']
        index = indices[s][int(row['entry_time'])]
        if not activity_gate(candles[s],index):
            continue
        checked = checked_row(row,candles[s],index,'be_tp1',.0005)
        rows.append(dict(checked, source_priority=3, module_priority=999,
                         outcome='SL' if checked['sequential_outcome']=='SL' else 'TP'))
    specs = json.loads(specs_path.read_bytes())
    corr = ClosedCorrelations(candles)
    selection = {p:evaluate(rows,corr,specs,p,START,SPLIT) for p in POLICIES}
    eligible = [p for p in POLICIES[1:] if qualifies(selection[p], selection['baseline'])]
    chosen = max(eligible,key=lambda p:min(v['summary']['return_pct'] for v in selection[p]['cash'].values())) if eligible else None
    later = {p:evaluate(rows,corr,specs,p,SPLIT) for p in ('baseline',chosen)} if chosen else None
    output = dict(status='CONCURRENT_CORRELATION_RESEARCH', selection=selection, chosen=chosen, later=later,
                  live_activation_supported=False, objective_verified=False)
    (ROOT/'crypto_overlap_correlation_nov19.json').write_text(json.dumps(output,indent=2,allow_nan=False),encoding='utf-8')
    for p,r in selection.items():
        print(p,r['status'],r.get('admitted'),r.get('blocks'),r.get('clusters'),flush=True)
    print('chosen',chosen,flush=True)


if __name__ == '__main__':
    main()
