"""Shared contract-rounded cash for fixed mark/trade trigger hypotheses."""
import argparse
import hashlib
import json
import math
import pickle
from pathlib import Path
import pandas as pd

from account_cash_guard_nov17 import calendar_months,stamp
from account_cash_guard_nov17_audit import signal_portfolio
from crypto_runner_exit_path_nov16 import replay
from lot_aware_tp1_audit import simulate

ROOT=Path('reports/audit_2026_09_08')
STOCK=Path('../tusa stocks/reports/audit_2026_09_08')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stock-report',type=Path,default=STOCK/'stock_mark_hypotheses_nov18.json')
    p.add_argument('--output',type=Path,default=ROOT/'stock_mark_shared_nov18.json')
    p.add_argument('--start-utc',default='2026-07-01',
        help='Start one continuous account at this UTC timestamp; no monthly resets')
    args=p.parse_args()
    start=pd.Timestamp(args.start_utc)
    start_time=(start.tz_localize('UTC') if start.tzinfo is None else start.tz_convert('UTC')).timestamp()
    if not math.isfinite(start_time):raise ValueError('invalid_cash_start')
    source=ROOT/'crypto_peer_breadth_nov17.json';frozen=json.loads(source.read_text(encoding='utf-8'))
    manifest=ROOT/'xperp_15m_nov14_merged/manifest.json';meta=json.loads(manifest.read_text(encoding='utf-8'))
    candles={};hashes={}
    for symbol,item in meta['symbols'].items():
        if not item.get('path'):continue
        path=Path(item['path']);raw=path.read_bytes();digest=hashlib.sha256(raw).hexdigest()
        if digest!=item['sha256']:raise ValueError('crypto source hash mismatch')
        candles[symbol]=pickle.loads(raw);hashes[str(path)]=digest
    lookups={s:{int(t):i for i,t in enumerate(c['time'])} for s,c in candles.items()}
    records=[]
    for row in frozen['candidate_diagnostics']:
        c=candles[row['symbol']];i=lookups[row['symbol']][int(row['entry_time'])]
        entry=float(c['close'][i-1]);sign=1 if row['direction']=='LONG' else -1
        value=replay(c,i,direction=row['direction'],entry=entry,stop=entry-sign*entry*row['risk_pct'])
        net=.025+.95*value['runner_r'] if value['banked'] else value['runner_r']
        if abs(net-row['model_r'])>1e-9 or value['exit_time']!=row['exit_time']:
            raise ValueError('crypto baseline path parity failed')
        records.append(dict(row,entry=entry,net_r=net,sequential_outcome=value['outcome'],
            sequential_runner_r=value['runner_r'],sequential_banked=value['banked'],
            near_exit_time=value['near_exit_time'] if value['near_exit_time'] is not None else math.nan))
    crypto=pd.DataFrame(signal_portfolio(records)).assign(bot='crypto')
    baseline_specs=ROOT/'shared_xperp_specs_nov14.json';specs=json.loads(baseline_specs.read_text(encoding='utf-8'))
    mark_manifest=STOCK/'stock_mark_matched_nov18/manifest.json'
    matching=json.loads(mark_manifest.read_text(encoding='utf-8'))
    for symbol,item in matching['symbols'].items():
        if item.get('status')!='OK' or item.get('spec',{}).get('instId')!=item.get('instrument'):
            raise ValueError('stock specification/contract not matched')
        specs[symbol]=item['spec']
    alternatives=json.loads(args.stock_report.read_text(encoding='utf-8'))
    report=dict(status='EXPLORATORY_MARK_SIGNAL_SHARED_CASH',objective_verified=False,
        live_activation_supported=False,crypto_path_parity=len(records),cash_start_utc=args.start_utc,
        cash_start_timestamp=start_time,results={},limitations=[
            'One continuous hypothetical $130 account,3% client margin,10x; no monthly resets.',
            '12 global slots,4crypto/8stocks,90% margin,two-writer headroom,15m release.',
            'Mark-trigger scenarios differ from current LIVE last-trigger orders; neither is promoted.',
            'Stock modeled single-target exits; real client partial settings never change.',
            'Positive last20 R and price readiness is recomputed and required for stock cash entries.',
            'Closed known references and hypothetical price stresses are not actual market execution prices.',
            'Current full-equity guard and minute pause recovery, floating PnL,fees,funding,transfers and latency absent.',
            'Current matched contract specs assumed across history; no historical bid/ask or private fills.',
            'Previously inspected market eras; July and November partial, no 30% monthly forecast.'])
    for name,item in alternatives['results'].items():
        if item.get('status')!='RESOLVED_REFERENCE_PATHS':
            report['results'][name]=dict(status=item.get('status'));continue
        path=Path('../tusa stocks')/item['trades_path'];raw=path.read_bytes()
        if hashlib.sha256(raw).hexdigest()!=item['trades_sha256']:raise ValueError('stock path hash mismatch')
        hashes[str(path)]=item['trades_sha256'];frame=pickle.loads(raw)
        ready=frame[frame.forward_r_ready & frame.forward_price_ready].assign(bot='stocks',
            sequential_banked=False,near_exit_time=math.nan)
        ready['sequential_runner_r']=ready.net_r
        feed=pd.concat([crypto,ready],ignore_index=True)
        feed=feed[feed.entry_time.ge(start_time)].to_dict('records')
        results={}
        for priority in ('crypto','stocks'):
            value=simulate(feed,specs,capital=130.,slots=12,bot_slots=dict(crypto=4,stocks=8),
                simultaneous_writers=2,release_delay=900,capture_events=True,
                tie_key=lambda r,p=priority:(r['bot']!=p,r['symbol']))
            months=calendar_months(value)
            months={label:month for label,month in months.items()
                if stamp(f'2026-{int(label[-2:])+1:02}-01')>start_time}
            for label,month in months.items():
                start,end=stamp(label+'-01'),stamp(f'2026-{int(label[-2:])+1:02}-01')
                closed=[r for r in value['ledger'] if r['bot']=='stocks' and start<=r['exit_time']<end]
                pnl=[r['bank_pnl']+r['runner_pnl'] for r in closed]
                month.update(stock_closed=len(pnl),stock_winrate_pct=100*sum(x>1e-12 for x in pnl)/len(pnl) if pnl else None)
            assert value['peak_positions']<=12 and value['peak_reserved_margin_pct']<=90+1e-9
            assert value['peak_by_bot'].get('crypto',0)<=4 and value['peak_by_bot'].get('stocks',0)<=8
            assert abs(sum(value['return_contribution_pct'].values())-value['return_pct'])<1e-4
            for key in ('ledger','cash_events','guard_rejections'):value.pop(key)
            results[priority]=dict(summary=value,calendar_months=months)
        report['results'][name]=dict(status='SHARED_REFERENCE_CASH',cash=results)
        print(name,{m:round(v['return_contribution_pct']['stocks'],3)
            for m,v in results['crypto']['calendar_months'].items()},flush=True)
    for path in (source,manifest,baseline_specs,mark_manifest,args.stock_report,Path(__file__),
        Path('account_cash_guard_nov17.py'),Path('account_cash_guard_nov17_audit.py'),
        Path('crypto_runner_exit_path_nov16.py'),Path('lot_aware_tp1_audit.py')):
        hashes[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
    report['sources']=hashes
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')


if __name__=='__main__':main()
