"""Replay shallow breakouts together with the two-position BTC exposure limit."""

from pathlib import Path
from unittest.mock import patch

import closed_acceptance_context_nov25 as acceptance
import okx_post_stop_reclaim_nov23 as reentry
import shared_correlated_exposure_nov25 as exposure
from account_cash_guard_nov17 import calendar_months
from okx_post_close_cash_nov23 import trade_difference
from okx_selective_single_target_nov23 import EARLIER, key
from okx_spot_confirmation_nov22 import load_native
from okx_sweep_context_cash_nov23 import holdout_inputs
from okx_unbankable_tp1_nov21 import old_inputs, read_json
from okx_xperp_multiday_replay_nov20 import ROOT, file_hash
from stock_liquid_quote_routes_nov21 import write_json


BREAKOUT = 'breakout_shallow_cross'
EXPOSURE = 'two_btc_loadings'
PROFILES = {
    'baseline': (False,False),
    'weak_breakout_only': (True,False),
    'shared_exposure_only': (False,True),
    'combined': (True,True),
}


def reject_breakout(row, context, *, executable):
    if row['bot'] != 'crypto':
        return False
    if executable:
        if row['entry_status']!='eligible':
            return False
        row = acceptance.label_family(row, context.frames)
    else:
        row = dict(row, epoch=int(row['entry_time']))
    return acceptance.skip(BREAKOUT, row, context.features(row))


def profile_rows(rows, blocked, *, executable):
    if executable:
        return [dict(r,entry_status=BREAKOUT) if key(r) in blocked else r for r in rows]
    return [r for r in rows if (r['bot'],r['symbol'],round(r['entry_time']*1000)) not in blocked]


def combined_delta(scores, field):
    baseline = scores['baseline'][field]
    isolated = (scores['weak_breakout_only'][field]-baseline) + (scores['shared_exposure_only'][field]-baseline)
    return scores['combined'][field]-baseline-isolated


def held_changes(baseline, trial, breakout, gate, *, exact):
    def ident(row):
        return key(row) if exact else (row['bot'],row['symbol'],round(row['entry_time']*1000))

    before = {ident(r):r for r in baseline}
    after = {ident(r):r for r in trial}
    removed = []
    for identity in sorted(before.keys()-after.keys()):
        row = before[identity]
        reason = BREAKOUT if identity in breakout else EXPOSURE if identity in gate.blocks else 'changed_admission'
        removed.append(dict(bot=row['bot'],symbol=row['symbol'],entry_ms=identity[-1],reason=reason,
                            cash_pnl_usd=row['bank_pnl']+row['runner_pnl'],outcome=row.get('code')))
    return dict(removed=removed, added=len(after.keys()-before.keys()))


def exact(rows,specs,ticks,books,archives,trades,references,expected):
    frames = load_native(references)
    context = acceptance.ClosedContext(frames)
    pairs = exposure.PriorPairs(frames)
    blocked = {key(row) for row in rows if reject_breakout(row,context,executable=True)}
    repricing = dict(repriced=[],extra_exit_checks={})
    replay = reentry.with_actual_fill_path(trades,archives,repricing)
    output = {}
    for lag in exposure.exact_cash.CLOSE_LAGS:
        options = dict(clock=60,lag=lag,setting='requested5')
        # Compounded lot size can change a fill enough to move the exchange's BE tick.
        base = replay(rows,specs,ticks,books,**options)
        mismatched = [k for k in acceptance.entry_data.FIELDS if base.get(k)!=expected[str(lag)].get(k)]
        if mismatched:
            raise ValueError(f'public-book cash baseline changed at {lag} ms: {mismatched}')
        scores = {}
        for name,(weak,shared) in PROFILES.items():
            scope = blocked if weak else set()
            gate = exposure.Admission(pairs,EXPOSURE if shared else 'observe',exact=True)
            trial = base if name=='baseline' else replay(
                profile_rows(rows,scope,executable=True),specs,ticks,books,**options,entry_admission=gate)
            if 'closed' not in trial:
                scores[name] = trial
                continue
            score = exposure.account_excursions(trial,gate,base['closed'],exact=True)
            score.update(attribution=held_changes(base['closed'],trial['closed'],scope,gate,exact=True),
                         book_cash_difference=trade_difference(base,trial), weak_candidates=len(scope))
            scores[name] = score
        print(lag,{k:(v.get('trades'),v.get('return_pct'),v.get('reason')) for k,v in scores.items()},flush=True)
        output[str(lag)] = dict(cases=scores, interaction_pct_points=(
            combined_delta(scores,'return_pct') if all('return_pct' in r for r in scores.values()) else None))
    return dict(results=output,repricing=repricing)


def older():
    rows,*material = old_inputs()
    acceptance.mark_old_family(rows)
    reference = read_json(ROOT/'shared_correlated_exposure_older_nov25.json')['cases']
    return {n:exact([r for r in rows if select(r)],*material,
                    {lag:v['baseline'] for lag,v in reference[n]['results'].items()})
            for n,select in EARLIER.items()}


def captured(mode):
    with patch.object(acceptance,'exact',exact):
        return {'native_joint':acceptance.native_joint,'fresh':acceptance.fresh}[mode]()


def alternate20():
    source = ROOT/'okx_sweep_context_cash_nov23.json'
    name = 'nov20_four_hours'
    reference = read_json(source)['exact_cash_cases'][name]
    return {name:dict(exact(*holdout_inputs(),{lag:v['baseline'] for lag,v in reference.items()}),
                      reference_sha256=file_hash(source))}


def alternate21():
    source = ROOT/'okx_post_stop_reclaim_nov23.json'
    name = 'nov21_crypto_forward'
    reference = read_json(source)['forward']['outcomes'][name]
    controls = {lag:v['baseline'] for lag,v in reference.items()}

    def evaluate(material,*_args,**_kwargs):
        return {name:exact(*material,controls)}

    with patch.object(reentry,'evaluate',evaluate):
        output = reentry.forward()
    return {name:dict(output[name], reference_sha256=file_hash(source),
                      entry_window_utc=output['entry_window_utc'])}


def candle_cash(rows,specs,context,pairs,capital,expected):
    blocked = {
        (row['bot'],row['symbol'],round(row['entry_time']*1000)) for row in rows
        if reject_breakout(row,context,executable=False)
    }
    options = dict(capital=capital,slots=12,bot_slots={'crypto':4,'stocks':8},simultaneous_writers=1,
                   release_delay=900,tie_key=lambda r:(r['bot']!='crypto',r['symbol']),
                   **exposure.CLIENTS['requested5'])
    baseline = exposure.proxy.simulate(rows,specs,**options)
    if capital==130 and any(baseline[k]!=expected[k] for k in ('trades','return_pct','cash_drawdown_pct','peak_positions')):
        raise ValueError('monthly candle cash baseline changed')
    scores = {}
    for name,(weak,shared) in PROFILES.items():
        scope = blocked if weak else set()
        gate = exposure.Admission(pairs,EXPOSURE if shared else 'observe',exact=False)
        trial = baseline if name=='baseline' else exposure.proxy.simulate(
            profile_rows(rows,scope,executable=False),specs,**options,entry_admission=gate)
        score = exposure.account_excursions(trial,gate,baseline['ledger'],exact=False)
        score.update(monthly=calendar_months(trial),weak_candidates=len(scope),
                     attribution=held_changes(baseline['ledger'],trial['ledger'],scope,gate,exact=False))
        scores[name] = score
    return dict(cases=scores,interaction_pct_points=combined_delta(scores,'return_pct'))


def monthly():
    material = exposure.monthly_inputs
    crypto,ready,paper,specs,end = material.records()
    reference = ROOT/'shared_correlated_exposure_monthly_nov25.json'
    expected = read_json(reference)['cases']['results']
    context = material.candidate_context({r['symbol'] for r in crypto+paper},Path('../tusa stocks'))
    pairs = exposure.PriorPairs(context.frames)
    output = {}
    for label,stocks in (('ready_one_writer',ready),('all_stock_capacity_diagnostic',paper)):
        output[label] = {}
        capitals = (100,130,160) if label=='ready_one_writer' else (130,)
        for capital in capitals:
            result = candle_cash(crypto+stocks,specs,context,pairs,capital,expected[label]['baseline'])
            output[label][str(capital)] = result
            print(label,capital,{k:(round(v['return_pct'],4),round(v['cash_drawdown_pct'],4))
                for k,v in result['cases'].items()},flush=True)
    return dict(scenarios=output,reference_sha256=file_hash(reference),end=end)


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    functions = dict(older=older,monthly=monthly,alternate20=alternate20,alternate21=alternate21,
                     native_joint=lambda:captured('native_joint'),fresh=lambda:captured('fresh'))
    parser.add_argument('part',choices=functions)
    part = parser.parse_args().part
    result = functions[part]()
    sources = (Path(__file__),Path(exposure.__file__),Path(acceptance.__file__))
    write_json(ROOT/f'combined_breakout_exposure_{part}_nov25.json',dict(
        cases=result,source_sha256={p.name:file_hash(p) for p in sources},
        profiles=PROFILES,live_activation_supported=False,limitations=[
            'Original publication, market quotes, brackets and risk geometry are frozen.',
            'The entry filter acts before the re-evaluated portfolio exposure. The cash ledger is rerun for each scenario.',
            'The exact cohorts price historic public depth. The longer sample uses candle proxies and fixed 3% client margin.',
            'Alternative capital values are risk-sensitivity scenarios. Private fills and monthly live returns are not scored.',
        ]))


if __name__=='__main__':
    main()
