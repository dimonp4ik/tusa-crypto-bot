"""Check loss-pause policies on the pre-November crypto candle cohort."""

from pathlib import Path

from crypto_minimum_bank_runner_nov19_audit import admitted_records, cash_result
from okx_swing_target_cash_nov23 import historical_brackets
from okx_unbankable_tp1_nov21 import read_json
from okx_xperp_multiday_replay_nov20 import ROOT, file_hash
from stock_liquid_quote_routes_nov21 import write_json


def main():
    brackets, specs, specs_path, universe, metadata, *_ = historical_brackets()
    admitted, unknown = admitted_records(brackets['baseline'])
    if unknown:
        raise ValueError('Published historical positions are unresolved')
    source = ROOT / 'crypto_nearer_target_long_history_nov22.json'
    expected = read_json(source)['cases']['control_2r']['clients']['requested5']
    results = {}
    for policy in (None, 'daily', 'bot_hour', 'direction_hour', 'account_hour'):
        score = cash_result(admitted, specs, 'requested5', loss_pause_policy=policy)
        if policy is None and score != expected:
            raise ValueError('Published monthly control changed')
        results[policy or 'baseline'] = score
        print(policy, score['summary']['trades'], score['summary']['return_pct'], score['summary']['cash_drawdown_pct'],
              {month: item['return_pct'] for month, item in score['calendar_months'].items()}, flush=True)
    if results['account_hour'] != results['bot_hour']:
        raise ValueError('A one-bot account must match the local hour rule')
    files = (source, specs_path, universe, Path('cash_tp1_policy_nov19.py'), Path('loss_pause_policy_nov24.py'))
    write_json(ROOT/'cash_loss_pause_history_nov24.json', dict(
        results=results, candle_pool_sha256=metadata['sha256'],
        sources={file.name:file_hash(file) for file in files}, live_activation_supported=False, limitations=[
            'July to October uses conservative candle exits and current contract increments.',
            'The frozen upstream candidate admission was applied before the additional portfolio policy.',
            'This is a one-bot account without stocks, so the local and shared hour rules are equivalent.',
        ]))


if __name__ == '__main__':
    main()
