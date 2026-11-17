"""Joint continuous cash for structural stock hypotheses; no live policy change."""
import hashlib
import json
import math
import pickle
from pathlib import Path
import pandas as pd

from account_cash_guard_nov17 import RealizedCashGuard,calendar_months,stamp
from account_cash_guard_nov17_audit import signal_portfolio
from crypto_runner_exit_path_nov16 import replay
from lot_aware_tp1_audit import simulate

ROOT=Path('reports/audit_2026_09_08')
STOCK=Path('../tusa stocks/reports/audit_2026_09_08')


def main():
    frozen_path=ROOT/'crypto_peer_breadth_nov17.json'
    frozen=json.loads(frozen_path.read_text(encoding='utf-8'))
    manifest=ROOT/'xperp_15m_nov14_merged/manifest.json'
    meta=json.loads(manifest.read_text(encoding='utf-8'));candles={};hashes={}
    for symbol,item in meta['symbols'].items():
        if not item.get('path'):continue
        p=Path(item['path']);raw=p.read_bytes();digest=hashlib.sha256(raw).hexdigest()
        if digest!=item['sha256']:raise ValueError('native crypto candle hash mismatch')
        candles[symbol]=pickle.loads(raw);hashes[str(p)]=digest
    indices={s:{int(t):i for i,t in enumerate(c['time'])} for s,c in candles.items()}
    records=[]
    for r in frozen['candidate_diagnostics']:
        c=candles[r['symbol']];i=indices[r['symbol']][int(r['entry_time'])]
        entry=float(c['close'][i-1]);sign=1 if r['direction']=='LONG' else -1
        p=replay(c,i,direction=r['direction'],entry=entry,stop=entry-sign*entry*r['risk_pct'])
        net=.025+.95*p['runner_r'] if p['banked'] else p['runner_r']
        if abs(net-r['model_r'])>1e-9 or p['exit_time']!=r['exit_time']:raise ValueError('crypto path parity failed')
        records.append(dict(r,entry=entry,net_r=net,sequential_outcome=p['outcome'],
            sequential_runner_r=p['runner_r'],sequential_banked=p['banked'],
            near_exit_time=p['near_exit_time'] if p['near_exit_time'] is not None else math.nan))
    crypto=pd.DataFrame(signal_portfolio(records)).assign(bot='crypto')
    stock_report_path=STOCK/'stock_structure_geometry_nov17.json'
    stock_report=json.loads(stock_report_path.read_text(encoding='utf-8'))
    spec_path=ROOT/'shared_xperp_specs_nov14.json'
    specs=json.loads(spec_path.read_text(encoding='utf-8'))
    reference_path=ROOT/'account_cash_guard_nov17.json'
    reference=json.loads(reference_path.read_text(encoding='utf-8'))
    report=dict(status='STRUCTURAL_STOCK_SHARED_CASH_RESEARCH',objective_verified=False,
        live_activation_supported=False,crypto_path_parity=len(records),results={},
        limitations=[
            'Previously inspected native data and exploratory bracket rules; no independent holdout.',
            'One flat $130 account from July 1,3% margin/10x/5% TP1; continuous states across months.',
            'Stock brackets are full single-target exits; client margin percentage never changes by signal.',
            'R-only readiness is counterfactual; the stricter case requires positive modeled price sum as well.',
            'Known-close references and 15m stop-first paths are not verified market quotes or private fills.',
            'Account guard sees realized cash, not OKX floating equity; transfers/funding/fees/latency are not reconstructed.',
            '12 global positions,4/8 local,90% margin,two-writer headroom and 15m release delay.',
            'July and November are partial; no forecast or proof of 30% monthly for either bot.'])
    for policy,meta in stock_report['results'].items():
        path=Path('../tusa stocks')/meta['trades_path']
        raw=path.read_bytes();digest=hashlib.sha256(raw).hexdigest()
        if digest!=meta['trades_sha256']:raise ValueError('stock path hash mismatch')
        hashes[str(path)]=digest;frame=pickle.loads(raw)
        result={}
        for readiness in ('r_only','r_and_price'):
            mask=frame.forward_r_ready if readiness=='r_only' else frame.forward_r_ready & frame.forward_price_ready
            stock=frame[mask].assign(bot='stocks',sequential_banked=False,near_exit_time=math.nan)
            stock['sequential_runner_r']=stock.net_r
            feed=pd.concat([crypto,stock],ignore_index=True)
            feed=feed[feed.entry_time.ge(stamp('2026-07-01'))].to_dict('records')
            result[readiness]={}
            for priority in ('crypto','stocks'):
                result[readiness][priority]={}
                for mode in ('no_account_guard','realized_cash_guard'):
                    guard=RealizedCashGuard(capital=130.) if mode=='realized_cash_guard' else None
                    value=simulate(feed,specs,capital=130.,slots=12,bot_slots=dict(crypto=4,stocks=8),
                        simultaneous_writers=2,release_delay=900,tie_key=lambda r,p=priority:(r['bot']!=p,r['symbol']),
                        entry_guard=guard,capture_events=True)
                    assert value['peak_positions']<=12 and value['peak_reserved_margin_pct']<=90+1e-9
                    monthly=calendar_months(value)
                    if policy=='baseline':
                        expected=reference['results']['baseline']['continuous'][priority][mode]['summary']
                        for k in ('ending_capital','return_pct','cash_drawdown_pct','trades'):
                            if abs(value[k]-expected[k])>1e-6:raise ValueError('shared baseline parity failed: '+k)
                    value.pop('ledger');value.pop('cash_events');value.pop('guard_rejections')
                    result[readiness][priority][mode]=dict(summary=value,calendar_months=monthly,
                        first_drawdown_blocks=guard.first_drawdown_blocks if guard else {},
                        guard_states=guard.states if guard else {})
            m=result[readiness]['crypto']['realized_cash_guard']['calendar_months']
            print(policy,readiness,'guarded_months',[(k,round(v['return_pct'],3),v['entries']) for k,v in m.items()],flush=True)
        report['results'][policy]=result
    report['baseline_cash_parity']=True
    for path in (frozen_path,manifest,spec_path,stock_report_path,reference_path,Path(__file__),
        Path('account_cash_guard_nov17.py'),Path('account_cash_guard_nov17_audit.py'),
        Path('crypto_runner_exit_path_nov16.py'),Path('lot_aware_tp1_audit.py')):
        hashes[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
    report['sources']=hashes
    (ROOT/'stock_structure_account_nov17.json').write_text(json.dumps(report,indent=2),encoding='utf-8')


if __name__=='__main__':main()
