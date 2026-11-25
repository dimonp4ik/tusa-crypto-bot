"""Compare fixed shadow-outcome policies on longer native candle cohorts."""

import argparse
from pathlib import Path
import sys

# The stock run imports the stock repository's own runtime modules.
sys.path.insert(0, str(Path.cwd()))
from settled_outcome_gate_nov25 import POLICIES, filter_events


RELEASE_MS = 900_000
MONTHS = [f'2026-{month:02d}-01' for month in range(7, 13)]


def identify(row):
    return row['symbol'], row['direction'], int(row['entry_time'])


def entrance(row):
    return int(row['entry_time']) * 1000


def group_for(row, partition):
    side = row['direction']
    if partition == 'side':
        return side,
    trend = row['signed_market_direction']
    return side, 'opposed' if trend < 0 else 'aligned'


def evaluate(rows, observations, measure):
    if len({identify(r) for r in rows}) != len(rows):
        raise ValueError('duplicate native candidate')
    results = dict(baseline=measure(rows))
    for name, policy in POLICIES.items():
        rejected = filter_events(rows, observations, policy, identify=identify, entrance=entrance,
                                 group=group_for)
        selected = [r for r in rows if identify(r) not in rejected]
        results[name] = dict(measure(selected), rejected=len(rejected))
    return results


def crypto():
    from crypto_minimum_bank_runner_nov19_audit import admitted_records, cash_result
    from okx_swing_target_cash_nov23 import historical_brackets
    from okx_unbankable_tp1_nov21 import read_json
    from okx_xperp_multiday_replay_nov20 import ROOT, file_hash
    from src.okx_trader import calc_contracts
    from src.tp1_plan import plan_tp1_close

    brackets, specs, _, _, metadata, *_ = historical_brackets()
    admitted, unknown = admitted_records(brackets['baseline'])
    if unknown:
        raise ValueError('unresolved historical native trade')
    source = ROOT/'okx_macro_reaction_cash_nov23.json'
    control = read_json(source)['crypto_older_candle_proxy']['cases']['baseline']['cash']
    if cash_result(admitted, specs, 'requested5') != control:
        raise ValueError('crypto candle control changed')
    rows = [dict(r, signed_market_direction=r['feature__BTC_return_16']) for r in admitted]
    evidence = {}
    for row in rows:
        spec = specs.get(row['symbol'])
        if spec is None:
            continue
        quantity = calc_contracts(130*.03, 10, row['entry'], spec)
        if quantity <= 0:
            continue
        tp1 = plan_tp1_close(quantity, 5, spec)['size'] if row['sequential_banked'] else 0.
        payout = (tp1*row['bank_price_return']+(quantity-tp1)*row['runner_price_return'])/quantity
        evidence[identify(row)] = int(row['exit_time']*1000)+RELEASE_MS, payout/row['risk_pct']

    def measure(frame):
        score = cash_result(frame, specs, 'requested5')
        return dict(calendar=score['calendar_months'], drawdown_pct=score['summary']['cash_drawdown_pct'])

    return ROOT, dict(cases=evaluate(rows, evidence, measure), observations=len(evidence), rows=len(rows),
                      source_sha256=file_hash(source), candle_pool_sha256=metadata['sha256'])


def stocks():
    import json
    import pandas as pd
    import stock_xperp_nov7_cluster_diagnostic as route
    from config import NATIVE_XPERP_DAILY_LOSS_R_CAP
    from stock_module_cluster_audit import portfolio_gate
    from stock_momentum_phase_nov14_audit import metrics, window

    root = Path('reports/audit_2026_09_08')
    src = root/'stock_xperp_nov14_preportfolio.pkl'
    manifest = json.loads((root/'stock_xperp_forward_nov14_merged/manifest.json').read_bytes())
    mature = min(v['end'] for v in manifest['symbols'].values() if v.get('end'))-63*900
    all_rows = pd.read_pickle(src)
    guarded = route.long_guard.apply_guard(route.premarket.apply_guard(all_rows[all_rows.entry_time.le(mature)].copy()))
    sample = portfolio_gate(guarded.to_dict('records'), None, rolling_hour_cap=4, rolling_direction_hour_cap=2,
        rank_key=route.quality.ranker('z_high'), loss_streak_cap=2, direction_cap=4,
        daily_loss_r_cap=NATIVE_XPERP_DAILY_LOSS_R_CAP)
    expected = pd.read_pickle(root/'stock_factor_nov14_baseline.pkl').to_dict('records')
    if set(map(identify, expected)) != set(map(identify, sample)):
        raise ValueError('stock candle control changed')
    rows = [dict(r, signed_market_direction=r['direction_n']*r['basket_trend_16']) for r in sample]
    evidence = {identify(r):(int(r['exit_time']*1000)+RELEASE_MS, r['signed_return']/(r['stop_pct']/100))
                for r in rows if pd.notna(r['exit_time']) and pd.notna(r['signed_return'])}

    def measure(selected):
        frame = pd.DataFrame(selected)
        return dict(calendar={a:metrics(window(frame, a, b)) for a,b in zip(MONTHS[:-1],MONTHS[1:],strict=True)})

    from hashlib import sha256
    return Path('../telegram bot')/root, dict(cases=evaluate(rows, evidence, measure), observations=len(evidence),
        rows=len(rows), source_sha256=sha256(src.read_bytes()).hexdigest())


def main():
    import json
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('bot', choices=('crypto', 'stocks'))
    bot = parser.parse_args().bot
    root, result = {'crypto':crypto, 'stocks':stocks}[bot]()
    for mode, case in result['cases'].items():
        print(bot, mode, case.get('rejected'), {k:round(v['return_pct'],3) for k,v in case['calendar'].items()},
              flush=True)
    result.update(live_activation_supported=False, limitations=[
        'July and November are partial. Native price series are candle proxies, separate from executable public-book tests.',
        'The original public signal admission is held fixed. Filtered proposals still settle in the gate’s virtual ledger.',
        'Crypto applies current contract increments. Stock uses the existing nominal 3% margin at 10x model.',
    ])
    path = root/f'settled_health_history_{bot}_nov25.json'
    path.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n', encoding='utf-8', newline='\n')


if __name__ == '__main__':
    main()
