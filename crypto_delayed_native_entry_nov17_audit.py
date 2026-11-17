"""Ten fixed confirmation hypotheses on previously inspected native history."""
import collections
import hashlib
import json
import math
import pickle
from pathlib import Path

import pandas as pd

from account_cash_guard_nov17 import calendar_months,stamp
from account_cash_guard_nov17_audit import signal_portfolio
from crypto_delayed_native_entry_nov17 import confirmation,RULES,opportunity_key,rotate
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
        if digest!=item['sha256']:raise ValueError('native candle hash mismatch')
        candles[symbol]=pickle.loads(raw);hashes[str(p)]=digest
    lookups={s:{int(t):i for i,t in enumerate(c['time'])} for s,c in candles.items()}
    originals=[]
    for r in frozen['candidate_diagnostics']:
        c=candles[r['symbol']];i=lookups[r['symbol']][int(r['entry_time'])]
        entry=float(c['close'][i-1]);sign=1 if r['direction']=='LONG' else -1
        stop=entry-sign*entry*r['risk_pct']
        path=replay(c,i,direction=r['direction'],entry=entry,stop=stop)
        net=.025+.95*path['runner_r'] if path['banked'] else path['runner_r']
        if abs(net-r['model_r'])>1e-9 or path['exit_time']!=r['exit_time']:
            raise ValueError('original native candidate parity failed')
        originals.append(dict(r,entry=entry,sl=stop))
    stock_report=STOCK/'stock_structure_geometry_nov17.json'
    stock_meta=json.loads(stock_report.read_text(encoding='utf-8'))['results']['baseline']
    stock_path=Path('../tusa stocks')/stock_meta['trades_path']
    raw=stock_path.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=stock_meta['trades_sha256']:
        raise ValueError('stock baseline hash mismatch')
    stock=pickle.loads(raw)
    stock=stock[stock.forward_r_ready & stock.forward_price_ready].assign(bot='stocks',
        sequential_banked=False,near_exit_time=math.nan)
    stock['sequential_runner_r']=stock.net_r
    specs_path=ROOT/'shared_xperp_specs_nov14.json'
    specs=json.loads(specs_path.read_text(encoding='utf-8'))
    policies=[('baseline',0,None)]+[(f'wait{delay}_{rule}',delay,rule) for delay in (1,2) for rule in RULES]
    report=dict(status='EXPLORATORY_NATIVE_CONFIRMATION_NOT_LIVE',objective_verified=False,
        live_change_supported=False,candidate_parity=len(originals),results={},
        limitations=[
            'All historical months were inspected previously; ten new hypotheses are exploratory, with no holdout.',
            'Frozen original classifier selects first; no model retraining, replacement families or hindsight labels.',
            'Waiting consumes one/two fully closed 15m bars; no position exists before confirmation.',
            'References are known closes, not executable contemporaneous bid/ask or private fills.',
            'Original structural stop retained; new 0.5R/2R targets and 1-2% risk gate use delayed reference.',
            'Original 48-bar deadline retained; target exhausted or structural stop touched before entry invalidates.',
            '15m stop-first exits with next-bar breakeven; no private amendment/latency reconstruction.',
            'Joint cash starts flat July 1 at $130, client 3% margin/10x/5% TP1 with lot rounding.',
            'Shared 12 slots, crypto4/stocks8,90% margin,two-writer headroom,15m release proxy.',
            'Stock baseline requires positive prior20 R and price-weighted sums, not authorization for LIVE.',
            'Current equity guard, floating equity, fees,funding, transfers and public spread rejection not replayed.',
            'July and November are partial; returns do not forecast live monthly profitability.'])
    variants={}
    if len({opportunity_key(r) for r in originals})!=len(originals):raise ValueError('duplicate original candidate')
    for name,delay,rule in policies+ [('daily_mature_rotation',None,None)]:
        rows=[];rejections=collections.Counter()
        for original in (originals if delay is not None else []):
            c=candles[original['symbol']];i=lookups[original['symbol']][int(original['entry_time'])]
            update=confirmation(c,i,original,delay=delay,rule=rule) if delay else dict(reason=None)
            if update['reason']:
                rejections[update['reason']]+=1;continue
            row=dict(original,**update,setup_time=original['entry_time'])
            path=replay(c,i+delay,direction=row['direction'],entry=row['entry'],stop=row['sl'],bars=48-delay)
            rows.append(dict(row,exit_time=path['exit_time'],sequential_outcome=path['outcome'],
                sequential_runner_r=path['runner_r'],sequential_banked=path['banked'],
                near_exit_time=path['near_exit_time'] if path['near_exit_time'] is not None else math.nan,
                net_r=.025+.95*path['runner_r'] if path['banked'] else path['runner_r']))
        if delay is None:
            rows,decisions=rotate(originals,variants)
            report['rotation_decisions']=decisions
        else:variants[name]={opportunity_key(r):r for r in rows}
        admitted=signal_portfolio(rows)
        crypto=pd.DataFrame(admitted).assign(bot='crypto')
        feed=pd.concat([crypto,stock],ignore_index=True)
        feed=feed[feed.entry_time.ge(stamp('2026-07-01'))].to_dict('records')
        result=dict(candidates=len(rows),signal_admitted=len(admitted),rejected=dict(rejections),cash={})
        for priority in ('crypto','stocks'):
            value=simulate(feed,specs,capital=130.,slots=12,bot_slots=dict(crypto=4,stocks=8),
                simultaneous_writers=2,release_delay=900,capture_events=True,
                tie_key=lambda r,p=priority:(r['bot']!=p,r['symbol']))
            months=calendar_months(value)
            for label,month in months.items():
                start,end=stamp(label+'-01'),stamp(f'2026-{int(label[-2:])+1:02}-01')
                closed=[r for r in value['ledger'] if r['bot']=='crypto' and start<=r['exit_time']<end]
                pnls=[r['bank_pnl']+r['runner_pnl'] for r in closed]
                month.update(crypto_closed=len(closed),crypto_profitable=sum(p>1e-12 for p in pnls),
                    crypto_winrate_pct=100*sum(p>1e-12 for p in pnls)/len(pnls) if pnls else None)
            assert value['peak_positions']<=12 and value['peak_reserved_margin_pct']<=90+1e-9
            for k in ('ledger','cash_events','guard_rejections'):value.pop(k)
            result['cash'][priority]=dict(summary=value,calendar_months=months)
        report['results'][name]=result
        print(name,dict(kept=len(rows),admitted=len(admitted),months={k:round(v['return_contribution_pct']['crypto'],3)
            for k,v in result['cash']['crypto']['calendar_months'].items()}),flush=True)
    for p in (source,manifest,stock_report,stock_path,specs_path,Path(__file__),
        Path('crypto_delayed_native_entry_nov17.py'),Path('account_cash_guard_nov17.py'),
        Path('account_cash_guard_nov17_audit.py'),Path('crypto_runner_exit_path_nov16.py'),Path('lot_aware_tp1_audit.py')):
        hashes[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
    report['sources']=hashes
    (ROOT/'crypto_delayed_native_entry_nov17.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')


if __name__=='__main__':main()
