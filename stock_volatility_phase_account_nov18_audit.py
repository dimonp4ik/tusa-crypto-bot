"""Contract-rounded shared cash for the closed, normalized stock hypotheses."""
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
    source=ROOT/'crypto_peer_breadth_nov17.json'
    frozen=json.loads(source.read_text(encoding='utf-8'))
    manifest=ROOT/'xperp_15m_nov14_merged/manifest.json'
    meta=json.loads(manifest.read_text(encoding='utf-8'));candles={};hashes={}
    for symbol,item in meta['symbols'].items():
        if not item.get('path'):continue
        p=Path(item['path']);raw=p.read_bytes();digest=hashlib.sha256(raw).hexdigest()
        if digest!=item['sha256']:raise ValueError('crypto source hash mismatch')
        candles[symbol]=pickle.loads(raw);hashes[str(p)]=digest
    lookups={s:{int(t):i for i,t in enumerate(c['time'])} for s,c in candles.items()}
    rows=[]
    for r in frozen['candidate_diagnostics']:
        c=candles[r['symbol']];i=lookups[r['symbol']][int(r['entry_time'])]
        entry=float(c['close'][i-1]);sign=1 if r['direction']=='LONG' else -1
        p=replay(c,i,direction=r['direction'],entry=entry,stop=entry-sign*entry*r['risk_pct'])
        net=.025+.95*p['runner_r'] if p['banked'] else p['runner_r']
        if abs(net-r['model_r'])>1e-9 or p['exit_time']!=r['exit_time']:
            raise ValueError('crypto baseline path parity failed')
        rows.append(dict(r,entry=entry,net_r=net,sequential_outcome=p['outcome'],
            sequential_runner_r=p['runner_r'],sequential_banked=p['banked'],
            near_exit_time=p['near_exit_time'] if p['near_exit_time'] is not None else math.nan))
    crypto=pd.DataFrame(signal_portfolio(rows)).assign(bot='crypto')
    stock_report=STOCK/'stock_volatility_phase_nov18.json'
    alternatives=json.loads(stock_report.read_text(encoding='utf-8'))
    spec_path=ROOT/'shared_xperp_specs_nov14.json'
    specs=json.loads(spec_path.read_text(encoding='utf-8'))
    report=dict(status='EXPLORATORY_SHARED_CASH_NORMALIZED_STOCK_SHOCKS',objective_verified=False,
        live_activation_supported=False,crypto_path_parity=len(rows),results={},limitations=[
            'Previously inspected native history and new exploratory hypotheses, not untouched validation.',
            'One continuous hypothetical $130 account from July 1; no monthly reset or inherited positions.',
            'Client 3% margin,10x,5% crypto TP1 with contract rounding; stock hypothetical full single-target exits.',
            'Stock readiness requires positive R and price sums of last20 completed shadow outcomes before entry.',
            'Shared12 slots, crypto4/stocks8,90% margin,two-writer headroom,15m slot release.',
            'Known close references and 15m stop-first paths are not real bid/ask market fills.',
            'Current full-equity risk checks and timed pause recovery are absent; these are unguarded cash sensitivities.',
            'No private fills, floating equity, spreads, latency, financing, transfers or funding reconstruction.',
            'Calendar cash months include cross-month position exits; July and November partial.',
            'Neither guaranteed profitability nor permission to enable stock real orders.'])
    for name,item in alternatives['results'].items():
        path=Path('../tusa stocks')/item['trades_path'];raw=path.read_bytes()
        digest=hashlib.sha256(raw).hexdigest()
        if digest!=item['trades_sha256']:raise ValueError('stock outcome hash mismatch')
        hashes[str(path)]=digest;frame=pickle.loads(raw)
        ready=frame[frame.forward_r_ready & frame.forward_price_ready].assign(bot='stocks',
            sequential_banked=False,near_exit_time=math.nan)
        ready['sequential_runner_r']=ready.net_r
        feed=pd.concat([crypto,ready],ignore_index=True)
        feed=feed[feed.entry_time.ge(stamp('2026-07-01'))].to_dict('records')
        result={}
        for priority in ('crypto','stocks'):
            value=simulate(feed,specs,capital=130.,slots=12,bot_slots=dict(crypto=4,stocks=8),
                simultaneous_writers=2,release_delay=900,capture_events=True,
                tie_key=lambda r,p=priority:(r['bot']!=p,r['symbol']))
            months=calendar_months(value)
            for label,month in months.items():
                start,end=stamp(label+'-01'),stamp(f'2026-{int(label[-2:])+1:02}-01')
                closed=[r for r in value['ledger'] if r['bot']=='stocks' and start<=r['exit_time']<end]
                pnls=[r['bank_pnl']+r['runner_pnl'] for r in closed]
                month.update(stock_closed=len(closed),stock_profitable=sum(p>1e-12 for p in pnls),
                    stock_winrate_pct=100*sum(p>1e-12 for p in pnls)/len(pnls) if pnls else None)
            assert value['peak_positions']<=12 and value['peak_reserved_margin_pct']<=90+1e-9
            assert value['peak_by_bot'].get('crypto',0)<=4 and value['peak_by_bot'].get('stocks',0)<=8
            for key in ('ledger','cash_events','guard_rejections'):value.pop(key)
            result[priority]=dict(summary=value,calendar_months=months)
        report['results'][name]=result
        print(name,{m:round(v['return_contribution_pct']['stocks'],3)
            for m,v in result['crypto']['calendar_months'].items()},flush=True)
    for p in (source,manifest,stock_report,spec_path,Path(__file__),Path('account_cash_guard_nov17.py'),
        Path('account_cash_guard_nov17_audit.py'),Path('crypto_runner_exit_path_nov16.py'),Path('lot_aware_tp1_audit.py')):
        hashes[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
    report['sources']=hashes
    (ROOT/'stock_volatility_phase_account_nov18.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')


if __name__=='__main__':main()
