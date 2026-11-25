"""Check correlated admissions with one cash balance and actual occupied slots."""

from bisect import bisect_left, bisect_right
from collections import Counter
from functools import lru_cache
from pathlib import Path
from unittest.mock import patch

import numpy as np

import cash_tp1_policy_nov19 as proxy
import closed_acceptance_context_nov25 as inputs
import closed_acceptance_monthly_context_nov25 as monthly_inputs
import okx_xperp_cash_tick_nov19 as exact_cash
from account_cash_guard_nov17 import calendar_months
from crypto_minimum_bank_runner_nov19_audit import CLIENTS
from okx_post_close_cash_nov23 import trade_difference
from okx_post_stop_reclaim_nov23 import with_actual_fill_path
from okx_selective_single_target_nov23 import EARLIER, cash_summary, key
from okx_spot_confirmation_nov22 import load_native
from okx_unbankable_tp1_nov21 import old_inputs, read_json
from okx_xperp_multiday_replay_nov20 import ROOT, file_hash
from stock_liquid_quote_routes_nov21 import write_json


POLICIES = ('cross_bot_pair_065', 'all_pair_080', 'one_btc_loading', 'two_btc_loadings')
REF = 'BTCUSDT'
BAR = 900
WINDOW = 96
MIN_ACTIVE = 32


class PriorPairs:
    def __init__(self, frames):
        self.returns = {}
        for symbol, frame in frames.items():
            times = np.asarray(frame['time'], dtype=np.int64)
            close = np.asarray(frame['close'], dtype=float)
            volume = np.asarray(frame['volume'] if 'volume' in frame else frame['volume_contracts'], dtype=float)
            if (len(times) == 0 or np.any(times % BAR) or np.any(np.diff(times) <= 0)
                    or not (len(times)==len(close)==len(volume)) or np.any(~np.isfinite(close))
                    or np.any(close<=0)):
                raise ValueError(f'invalid native candles for {symbol}')
            active = np.r_[False, (volume[1:]>0) & (volume[:-1]>0) & (np.diff(times)==BAR)]
            rates = np.r_[np.nan, np.diff(np.log(close))]
            rates[~active] = np.nan
            self.returns[symbol] = (times, rates)

    @lru_cache(maxsize=8192)
    def window(self, symbol, epoch):
        times, rates = self.returns[symbol]
        start = epoch-WINDOW*BAR
        lo, hi = np.searchsorted(times, (start,epoch), side='left')
        observed = np.full(WINDOW, np.nan)
        observed[(times[lo:hi]-start)//BAR] = rates[lo:hi]
        return observed

    @lru_cache(maxsize=24576)
    def correlation(self, a, b, epoch):
        if epoch % BAR:
            raise ValueError('statistics require an aligned decision time')
        if a not in self.returns or b not in self.returns:
            return None
        x, y = self.window(a, epoch), self.window(b, epoch)
        paired = np.isfinite(x) & np.isfinite(y)
        if int(np.count_nonzero(paired)) < MIN_ACTIVE:
            return None
        x, y = x[paired], y[paired]
        x, y = x-x.mean(), y-y.mean()
        scale = np.linalg.norm(x) * np.linalg.norm(y)
        return float(np.clip(np.dot(x,y)/scale,-1.,1.)) if scale>0 else None

    def get(self, a, b, epoch):
        return self.correlation(*sorted((a,b)), epoch)


def signed(side):
    return {'LONG':1, 'SHORT':-1}[side]


class Admission:
    """Consume only the held instruments and directions seen by the ledger."""

    def __init__(self, pairs, policy, *, exact):
        self.pairs = pairs
        self.policy = policy
        self.exact = exact
        self.counts = Counter()
        self.decisions = {}
        self.blocks = {}

    def __call__(self, row, held):
        symbol, bot, side = (row[k] for k in ('symbol','bot','direction'))
        clock = row['entry_ms'] if self.exact else round(row['entry_time']*1000)
        epoch = int(row['epoch'] if self.exact else row['entry_time'])
        identity = bot,symbol,clock
        replay_key = identity,tuple(sorted(held.items()))
        if replay_key not in self.decisions:
            self.decisions[replay_key] = self.assess(symbol,bot,side,epoch,held)
        hits = self.decisions[replay_key]
        if not hits or self.policy=='observe':
            return None
        if identity not in self.blocks:
            self.blocks[identity] = hits
            self.counts['refused:'+bot] += 1
        return self.policy

    def assess(self,symbol,bot,side,epoch,held):
        hits = []
        self.counts['eligible_attempts'] += 1
        factor_limit = {'one_btc_loading':1,'two_btc_loadings':2}.get(self.policy)
        if factor_limit is None:
            for other, (owner, direction) in held.items():
                cross = bot!=owner
                if self.policy=='cross_bot_pair_065' and not cross:
                    continue
                prefix = 'cross' if cross else 'same'
                self.counts[prefix+':pairs'] += 1
                corr = self.pairs.get(symbol,other,epoch)
                if corr is None:
                    continue
                self.counts[prefix+':known'] += 1
                strength = signed(side)*signed(direction)*corr
                if strength>=.65:
                    self.counts[prefix+':signed_above_065'] += 1
                threshold = .65 if self.policy=='cross_bot_pair_065' else .80
                if strength>=threshold:
                    hits.append((other,owner,strength))
        else:
            coefficient = self.pairs.get(symbol,REF,epoch)
            if coefficient is None or abs(coefficient)<.65:
                return hits
            self.counts['btc_loaded_attempts'] += 1
            loading = signed(side)*coefficient
            for other,(owner,direction) in held.items():
                peer = self.pairs.get(other,REF,epoch)
                if peer is not None and abs(peer)>=.65 and loading * signed(direction)*peer>0:
                    hits.append((other,owner,signed(direction)*peer))
            if len(hits)<factor_limit:
                hits.clear()
        return hits


def outcome_stats(positions, *, exact):
    exits = {bot:[] for bot in ('crypto','stocks')}
    positive = Counter()
    for row in positions:
        net = row['bank_pnl']+row['runner_pnl']
        if net>1e-12:
            positive[row['bot']] += 1
        elif net < -1e-12:
            exited = row['exit_ms'] if exact else round(row['exit_time']*1000)
            exits[row['bot']].append(exited)
    stock_exits = sorted(exits['stocks'])
    cross_pairs = sum(bisect_left(stock_exits,stamp+3_600_000)-bisect_right(stock_exits,stamp-3_600_000)
                      for stamp in exits['crypto'])
    all_exits = sorted(exits['crypto']+exits['stocks'])
    all_pairs = sum(bisect_left(all_exits,stamp+3_600_000)-i-1 for i,stamp in enumerate(all_exits))
    return dict(winning_exits_by_bot=positive, losing_exits_by_bot={b:len(v) for b,v in exits.items()},
                overlapping_cross_bot_loss_pairs=cross_pairs, overlapping_loss_pairs=all_pairs)


def account_excursions(result, gate, original, *, exact):
    records = result['closed' if exact else 'ledger']
    fields = ('trades','return_pct','cash_drawdown_pct','peak_positions','peak_by_bot',
              'peak_reserved_margin_pct','skipped','return_contribution_pct')
    score = cash_summary(result,{}) if exact else {k:result[k] for k in fields}
    score.update(outcome_stats(records,exact=exact))
    removed = []
    for row in original:
        clock = row['entry_ms'] if exact else round(row['entry_time']*1000)
        blocked_by = gate.blocks.get((row['bot'],row['symbol'],clock))
        if blocked_by:
            removed.append(dict(bot=row['bot'],symbol=row['symbol'],entry_ms=clock,
                                cash_pnl_usd=row['bank_pnl']+row['runner_pnl'],outcome=row.get('code'),
                                blocked_by=blocked_by))
    return dict(score, exposure_counts=gate.counts, rejected_baseline_positions=removed)


def exact(rows, specs, ticks, books, archives, trades, references, expected):
    pairs = PriorPairs(load_native(references))
    repricing = dict(repriced=[],extra_exit_checks={})
    replay = with_actual_fill_path(trades, archives, repricing)
    output = {}
    for lag in exact_cash.CLOSE_LAGS:
        options = dict(clock=60,lag=lag,setting='requested5')
        observe = Admission(pairs,'observe',exact=True)
        base = exact_cash.simulate(rows,specs,ticks,books,**options,entry_admission=observe)
        if any(base.get(k)!=expected[str(lag)].get(k) for k in inputs.entry_data.FIELDS):
            raise ValueError('baseline deviates from frozen public cash')
        outcomes = dict(baseline=account_excursions(base,observe,[],exact=True))
        for policy in POLICIES:
            gate = Admission(pairs,policy,exact=True)
            test = replay(rows,specs,ticks,books,**options,entry_admission=gate)
            result = (dict(account_excursions(test,gate,base['closed'],exact=True),
                           difference=trade_difference(base,test))
                      if 'closed' in test else test)
            outcomes[policy] = result
        output[str(lag)] = outcomes
        print(lag,{k:(r.get('trades'),r.get('return_pct'),r.get('exposure_counts'),r.get('reason'))
                   for k,r in outcomes.items()},flush=True)
    return dict(results=output,repricing=repricing)


def older():
    rows,*material = old_inputs()
    previous = read_json(ROOT/'closed_acceptance_context_older_nov25.json')['cases']
    return {name:exact([r for r in rows if select(r)],*material,
                       {lag:r['baseline'] for lag,r in previous[name]['results'].items()})
            for name,select in EARLIER.items()}


def derived(mode):
    with patch.object(inputs,'exact',exact):
        return {'native_joint':inputs.native_joint,'fresh':inputs.fresh}[mode]()


def monthly():
    crypto, ready, paper, specs, end = monthly_inputs.records()
    frames = monthly_inputs.candidate_context({r['symbol'] for r in crypto+paper},Path('../tusa stocks')).frames
    pairs = PriorPairs(frames)
    data = ROOT/'shared_admission_replay_nov19.json'
    previous = read_json(data)['results']
    options = dict(capital=130.,slots=12,bot_slots={'crypto':4,'stocks':8},simultaneous_writers=1,
                   release_delay=900,tie_key=lambda r:(r['bot']!='crypto',r['symbol']),
                   **CLIENTS['requested5'])
    results = {}
    for name,stocks in (('ready_one_writer',ready),('all_stock_capacity_diagnostic',paper)):
        rows = crypto+stocks
        outcomes = {}
        baseline = None
        for policy in ('observe',*POLICIES):
            gate = Admission(pairs,policy,exact=False)
            sim = proxy.simulate(rows,specs,**options,entry_admission=gate)
            if baseline is None:
                control = previous[name]['requested5']['crypto']
                metrics = ('return_pct','cash_drawdown_pct','peak_positions','peak_by_bot','peak_reserved_margin_pct')
                if any(sim[k]!=control[k] for k in metrics):
                    raise ValueError('shared candle baseline has changed')
                baseline = sim
            score = account_excursions(sim,gate,baseline['ledger'],exact=False)
            score['monthly'] = calendar_months(sim)
            outcomes['baseline' if policy=='observe' else policy] = score
            print(name,policy,round(score['return_pct'],5),round(score['cash_drawdown_pct'],5),
                  len(score['rejected_baseline_positions']),score['overlapping_cross_bot_loss_pairs'],flush=True)
        results[name] = outcomes
    return dict(results=results,end=end,reference_sha256=file_hash(data))


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=('older','native_joint','fresh','monthly'))
    mode = parser.parse_args().mode
    result = {'older':older,'monthly':monthly}.get(mode,lambda:derived(mode))()
    files = (Path(__file__),Path(exact_cash.__file__),Path(proxy.__file__))
    write_json(ROOT/f'shared_correlated_exposure_{mode}_nov25.json',dict(
        cases=result,policies=POLICIES,contact_window_bars=WINDOW,minimum_matched_active_bars=MIN_ACTIVE,
        source_sha256={p.name:file_hash(p) for p in files},live_activation_supported=False,limitations=[
            'Correlations use only closed 15-minute returns. Equal timestamps reuse the same strictly prior window.',
            'A return requires trades in both adjacent periods and observes no information after the decision boundary.',
            'Only held directions already visible in the same simulated cash ledger can block a candidate.',
            'Unrecognized pair exposure is counted as unknown and does not trigger a veto.',
            'The exact cash tests use historical OKX depth. Monthly scenarios are candle proxies and exclude private returns.',
        ]))


if __name__=='__main__':
    main()
