"""Client TP1 cash comparison on frozen signals; no entry/exit parameter fitting."""
import hashlib
import json
import math
from pathlib import Path
import pandas as pd
from account_cash_guard_nov17 import calendar_months,stamp
from account_cash_guard_nov17_audit import signal_portfolio
from cash_tp1_policy_nov19 import simulate
from crypto_native_price_pool_nov19 import load_pool,ROOT
from shared_factor_account_nov14_audit import stock_ready_rows


def main():
    frame,meta=load_pool()
    crypto=pd.DataFrame(signal_portfolio(frame[frame.is_frozen_candidate].to_dict('records'))).assign(bot='crypto')
    stock_path=Path('../tusa stocks/reports/audit_2026_09_08/stock_close_anchor_nov16_causal_candidates_known_close.pkl')
    stock=stock_ready_rows(pd.read_pickle(stock_path)).assign(bot='stocks',sequential_banked=False,near_exit_time=math.nan)
    stock['sequential_runner_r']=stock.net_r
    spec_path=ROOT/'shared_xperp_specs_nov14.json';specs=json.loads(spec_path.read_text(encoding='utf-8'))
    policies={f'requested{p}':dict(tp1_pct=p,minimum_close=False) for p in (5,25,50,75)}
    policies['half_minimum']=dict(tp1_pct=50,minimum_close=True);results={}
    for scenario in ('crypto_only','with_frozen_stock_reference'):
        feed=crypto if scenario=='crypto_only' else pd.concat([crypto,stock],ignore_index=True)
        rows=feed[feed.entry_time.ge(stamp('2026-07-01'))].to_dict('records');results[scenario]={}
        for name,policy in policies.items():
            value=simulate(rows,specs,capital=130.,slots=12,bot_slots=dict(crypto=4,stocks=8),
                simultaneous_writers=2,release_delay=900,tie_key=lambda r:(r['bot']!='crypto',r['symbol']),**policy)
            months=calendar_months(value)
            for month,item in months.items():
                start,end=stamp(month+'-01'),stamp(f'2026-{int(month[-2:])+1:02}-01')
                closed=[r for r in value['ledger'] if r['bot']=='crypto' and start<=r['exit_time']<end]
                pnl=[r['bank_pnl']+r['runner_pnl'] for r in closed]
                item.update(crypto_closed=len(closed),crypto_winrate_pct=100*sum(p>1e-12 for p in pnl)/len(pnl) if pnl else None,
                    crypto_tp1_bank_usd=sum(r['bank_pnl'] for r in closed),
                    crypto_tp1_hits_without_bank=sum(r['tp1_hit'] and r['planned_partial']==0 for r in closed))
            results[scenario][name]=dict(summary={k:v for k,v in value.items() if k not in ('ledger','cash_events','guard_rejections')},
                calendar_months=months)
    paths=(Path(__file__),Path('cash_tp1_policy_nov19.py'),Path('src/tp1_plan.py'),Path('src/okx_trader.py'),
        Path('account_cash_guard_nov17_audit.py'),stock_path,spec_path)
    report=dict(status='CLIENT_TP1_VOLUME_CASH_COMPARISON',objective_verified=False,live_activation_supported=False,
        frozen_candidate_path_parity=meta['frozen_candidate_path_parity'],policies=policies,results=results,
        sources={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        limitations=['Fixed public reference paths; comparison is not client private fills or current account performance.',
            'One continuous $130 account,3% client margin,10x; no monthly resets.',
            'Crypto-only represents no stock entries; optional stock scenario is an old frozen reference, not current PAPER strategy.',
            'Earlier full private TP1 releases cash/margin; paper signal admission retains its original lifetime.',
            'No full-equity guard,floating PnL,minute recovery,funding,fees,historical quotes or latency.',
            'Current specs assumed historically; November partial. Better banking frequency does not prove higher profit.'])
    (ROOT/'crypto_tp1_bank_size_nov19.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    for name,item in results['crypto_only'].items():
        print('tp1_cash',name,'DD',item['summary']['cash_drawdown_pct'],'unbanked',item['summary']['tp1_hits_without_bank'],
            {m:round(v['return_pct'],3) for m,v in item['calendar_months'].items()},flush=True)


if __name__=='__main__':main()
