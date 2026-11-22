"""Reprice subsequent admissions after ordinary and swing-level closes."""

import json

from crypto_minimum_bank_runner_nov19_audit import admitted_records, cash_result
from okx_delayed_trailing_cash_nov23 import paired_dollars, stock_sources
from okx_selective_single_target_nov23 import EARLIER, JOINT, cash_summary, key
from okx_swing_target_cash_nov23 import historical_brackets, prepare_variants
from okx_sweep_context_cash_nov23 import holdout_inputs
from okx_unbankable_tp1_nov21 import old_inputs, recent_inputs
from okx_xperp_cash_tick_nov19 import CAPITAL, CLOSE_LAGS, FRACTION, LEVERAGE, RELEASE_MS, simulate
from okx_xperp_multiday_replay_nov20 import ROOT, exit_books, file_hash
from post_close_replay_nov23 import PostCloseRule


RULES = {
    'no_extra_delay': None,
    'instrument_45m': PostCloseRule(instrument_seconds=2700),
    'same_bot_direction_30m_after_loss': PostCloseRule(direction_loss_seconds=1800),
    'same_bot_direction_60m_after_loss': PostCloseRule(direction_loss_seconds=3600),
    'instrument_45m_or_same_bot_direction_30m_after_loss': PostCloseRule(2700,1800),
    'stock_direction_60m_after_loss': PostCloseRule(direction_loss_seconds=3600,direction_loss_bot='stocks'),
    'same_bot_direction_60m_after_loss_over_2pct_margin': PostCloseRule(
        direction_loss_seconds=3600,loss_margin_floor=.02),
}
EXITS = ('baseline','swing_target')
PREFIX = 'okx_post_close_cash_nov23'
FAST_RELEASE = 'nov17_19_joint_cap12_release_2m'


def trade_difference(control, trial):
    notes = paired_dollars(control,trial)
    passes = {key(row):row for row in trial['closed']}
    originals = {key(row):row for row in control['closed']}
    for label, side, other in (('added_positions',passes,originals),('removed_positions',originals,passes)):
        positions = []
        for identity in sorted(side.keys()-other.keys()):
            trade = side[identity]
            positions.append(dict(bot=trade['bot'],symbol=trade['symbol'],entry_ms=trade['entry_ms'],
                                  exit=trade['code'],cash_pnl_usd=trade['bank_pnl']+trade['runner_pnl']))
        notes[label] = positions
    return notes


def evaluate(material, cohorts, conventions, source, *, joint=False, release_overrides=None):
    rows,specs,ticks,books,archive,trades,references = material
    if joint:
        trades,references=stock_sources(trades,references)
    variants,paths,checks=prepare_variants(rows,specs,ticks,trades,references)
    extra,book_checks=exit_books(paths,{'books':archive},specs,books)
    cases = {}
    for name,selector in cohorts.items():
        cohort = [row for row in rows if selector(row)]
        cases[name] = {}
        for lag in CLOSE_LAGS:
            cases[name][str(lag)] = {}
            for method in EXITS:
                changed = variants[method]
                ordered = [changed.get(key(row),row) for row in cohort]
                release = (release_overrides or {}).get(name,RELEASE_MS)
                params = dict(clock=60,lag=lag,setting='requested5',
                              max_positions=conventions[name],release_delay_ms=release)
                reference = simulate(ordered,specs,ticks,books,**params)
                if name in source:
                    expected = source[name][str(lag)][method]
                    for field in ('trades','return_pct','drawdown_pct','contribution_pct'):
                        if reference[field] != expected[field]:
                            raise ValueError(f'published control disagrees: {name}, {method}, {field}')
                outcomes = {}
                for label,rule in RULES.items():
                    result = (reference if rule is None else
                              simulate(ordered,specs,ticks,books,**params,post_close_rule=rule))
                    summary = cash_summary(result,changed)
                    if 'closed' in result:
                        summary['cash_change_usd'] = summary['cash_pnl_usd']-sum(
                            r['bank_pnl']+r['runner_pnl'] for r in reference['closed'])
                        summary['comparison'] = trade_difference(reference,result)
                    outcomes[label] = summary
                if method == 'baseline':
                    unchanged = reference
                else:
                    outcomes['no_extra_delay']['effect_of_replacing_target'] = trade_difference(unchanged,reference)
                cases[name][str(lag)][method] = outcomes
                print(name,lag,method,{k:(round(v['cash_pnl_usd'],3),v['trades']) if 'cash_pnl_usd' in v
                                       else v.get('reason') for k,v in outcomes.items()},flush=True)
    return cases,dict(paths=checks,book_checks=book_checks,archives={k:v['sha256'] for k,v in extra.items()})


def history(expected):
    brackets,specs,spec_file,universe,metadata,*_ = historical_brackets()
    results = {}
    for method in EXITS:
        admitted, unknown = admitted_records(brackets[method])
        if unknown:
            raise ValueError(f'historical {method} control unresolved')
        variants = {}
        for label,rule in RULES.items():
            score = cash_result(admitted,specs,'requested5',post_close_rule=rule)
            if rule is None and score != expected[method]['cash']:
                raise ValueError(f'published historical {method} control changed')
            variants[label]=score
        results[method]=variants
        print('jul_oct',method,{k:(v['summary']['return_pct'],v['summary']['cash_drawdown_pct'],v['summary']['trades'])
                                 for k,v in variants.items()},flush=True)
    return dict(cases=results,source_sha256={p.name:file_hash(p) for p in (spec_file,universe)},
                candidate_pool_sha256=metadata['sha256'])


def stock_followup():
    from stock_liquid_quote_routes_nov21 import limit_bursts, load_replay, stressed_exit_penalty
    from stock_quote_continuation_holdout_nov21 import DATA, STEM

    plan, _, _, specs, ticks, books, tape, stored_tapes, _ = load_replay(DATA,keep_tapes=True)
    source = ROOT/f'{STEM}_exit_stress.json'
    control = json.loads(source.read_bytes())['results']
    cases = {}
    for name,rows in plan['cases'].items():
        chosen = list(limit_bursts((r for r in rows if r['route']=='aligned_continuation'),max_same_side=2))
        cases[name] = {}
        for bps in (5,15,30,50):
            params = dict(clock=60,lag=200,setting='requested5',
                          exit_price=stressed_exit_penalty(stored_tapes,specs,bps))
            baseline = simulate(chosen,specs,ticks,books,**params)
            expected = control[name][f'aligned_continuation|stock_cadence|stress_{bps}bps']
            fields = ('trades','return_pct','drawdown_pct','contribution_pct') if 'closed' in baseline else ('symbol','reason')
            for field in fields:
                if baseline[field]!=expected[field]:
                    raise ValueError(f'published independent stock baseline disagrees: {name}, {bps}, {field}')
            outcomes = {}
            for label,rule in RULES.items():
                trial = baseline if rule is None else simulate(chosen,specs,ticks,books,**params,post_close_rule=rule)
                summary = cash_summary(trial,{})
                if 'closed' in trial:
                    summary['status'] = 'SCORED_PUBLIC_TRADE_EXIT_STRESS'
                    if 'closed' in baseline:
                        summary['comparison'] = trade_difference(baseline,trial)
                outcomes[label] = summary
            cases[name][str(bps)] = outcomes
            print('stock_followup',name,bps,{k:(v.get('trades'),v.get('cash_pnl_usd'),v.get('reason'))
                                            for k,v in outcomes.items()},flush=True)
    return dict(cases=cases,source_sha256=file_hash(source),tape=tape,
                limitation='Separate later stock continuation sample with observed entry depth and stressed trade-print exits.')


def main():
    path=ROOT/'okx_swing_target_cash_nov23.json'
    source=json.loads(path.read_bytes())
    control=source['cash_cases']
    older,older_checks=evaluate(old_inputs(),EARLIER,{name:12 for name in EARLIER},control)
    cap12='nov17_19_joint_cap12'
    joint,joint_checks=evaluate(recent_inputs(),JOINT|{cap12:lambda _:True,FAST_RELEASE:lambda _:True},
                                {'nov17_19_joint':6,cap12:12,FAST_RELEASE:12},control,joint=True,
                                release_overrides={FAST_RELEASE:120_000})
    latest='nov20_four_hours'
    holdout,holdout_checks=evaluate(holdout_inputs(),{latest:lambda _:True},{latest:12},control)
    report=dict(initial_capital_usd=CAPITAL,margin=FRACTION,leverage=LEVERAGE,
                source_sha256=file_hash(path),rules={k:vars(v) if v else None for k,v in RULES.items()},
                matched_cash_cases=older|joint|holdout,
                modeled_release_ms={k:120_000 if k==FAST_RELEASE else RELEASE_MS for k in older|joint|holdout},
                long_history_crypto_proxy=history(source['monthly_candle_proxy']['cases']),
                later_stock_continuation_exit_stress=stock_followup(),
                archive_checks=dict(older=older_checks,joint=joint_checks,holdout=holdout_checks),
                live_activation_supported=False, limitations=[
                    'The delay starts at the realized close time and becomes eligible for evaluation only after confirmation and release.',
                    'Two-minute release is a stress case and does not assume a guaranteed production release time.',
                    'The new direction rule is bot-local and uses negative aggregate bank-plus-runner cash, never a pending result.',
                    'The stock-only variant and 2% margin loss floor were added after examining the broad direction rule.',
                    'Initial proposals, leverage and client margin remain the same. Existing publication rules were applied upstream.',
                    'Stock continuation in the shared study is experimental. Native stock data did not provide enough fills for this paired test.',
                    'The July–October crypto sample uses conservative native-candle exits instead of public order books.',
                    'The November holdout is a short disjoint four-hour crypto decision window.',
                ])
    (ROOT/f'{PREFIX}.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8',newline='\n')


if __name__=='__main__':
    main()
