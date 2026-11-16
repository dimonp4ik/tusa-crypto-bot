"""Paired same-signal timing audit, not a new strategy or portfolio claim."""
import hashlib
import json
import math
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from public_candle_window import fetch_window
from live_filter_loss_audit_nov15 import load_snapshot,ROOT,STOCK
from append_candle_snapshot import merge


def replay(data,*,entry,direction,stop,target,end):
    sign=1 if direction=='LONG' else -1
    for i,stamp in enumerate(data['time']):
        if stamp>=end:break
        o,h,l,c=(data[k][i] for k in ('open','high','low','close'))
        if (o<=stop if sign==1 else o>=stop):
            return dict(outcome='SL',price=o,r=sign*(o-entry)/abs(entry-stop),exit_time=stamp)
        if (l<=stop if sign==1 else h>=stop):
            return dict(outcome='SL',price=stop,r=-1.,exit_time=stamp)
        if (h>=target if sign==1 else l<=target):
            return dict(outcome='TP',price=target,r=abs(target-entry)/abs(entry-stop),exit_time=stamp)
    return dict(outcome='TIME',price=data['close'][-1],
        r=sign*(data['close'][-1]-entry)/abs(entry-stop),exit_time=end)


def main():
    manifest=ROOT/'stock_15m/manifest.json';metadata=json.loads(manifest.read_text())
    old_path=STOCK/'reports/audit_2026_09_08/stock_xperp_forward_nov14_merged/manifest.json'
    old,fresh=load_snapshot(old_path),load_snapshot(manifest)
    candles={s:merge(old[s],fresh[s])[0] for s in fresh}
    export=ROOT/'stock_signals.json';signals=json.loads(export.read_text())['signals']
    cache=ROOT/'paired_stock_minutes';cache.mkdir(exist_ok=True)
    def inspect(signal):
        entry_time=int(signal['strategy_entry_bar_ts']);opened=float(signal['opened_at'])
        end=int(signal['strategy_expire_at'] or entry_time+64*900)
        minute_start=int(opened//60)*60
        inst=metadata['symbols'][signal['symbol']]['instrument']
        minutes,mhash=fetch_window(inst,minute_start,end,60,cache)
        seconds,shash=fetch_window(inst,minute_start,minute_start+60,1,cache)
        first=math.ceil(opened)
        # The fractional publication second remains explicitly ambiguous.
        # Full seconds strictly after it and all later minutes are causal.
        path={'time':[t for t in seconds['time'] if t>=first]+
            [t for t in minutes['time'] if t>=minute_start+60]}
        for key in ('open','high','low','close','volume'):
            path[key]=[v for i,v in enumerate(seconds[key]) if seconds['time'][i]>=first]+[
                v for i,v in enumerate(minutes[key]) if minutes['time'][i]>=minute_start+60]
        # Isolate the whole-minute bug from the remaining <1s boundary.
        sign=1 if signal['direction']=='LONG' else -1
        entry=signal['entry_price'];stop=signal['sl'];target=signal['tp1']
        touches=lambda h,l:(l<=stop or h>=target) if sign==1 else (h>=stop or l<=target)
        i=int(math.floor(opened))-minute_start
        boundary_ambiguous=touches(seconds['high'][i],seconds['low'][i])
        whole=replay(minutes,entry=entry,direction=signal['direction'],stop=stop,target=target,end=end)
        causal=replay(path,entry=entry,direction=signal['direction'],stop=stop,target=target,end=end)
        candle=candles[signal['symbol']];j=candle['time'].index(entry_time)
        entry_minutes,emhash=fetch_window(inst,entry_time,entry_time+900,60,cache)
        active=next((t for i,t in enumerate(entry_minutes['time']) if entry_minutes['volume'][i]>0),None)
        first_trade=None;first_trade_price=None;thash=None
        if active is not None:
            trades,thash=fetch_window(inst,active,active+60,1,cache)
            traded=next(i for i,v in enumerate(trades['volume']) if v>0)
            first_trade=trades['time'][traded];first_trade_price=trades['open'][traded]
            if abs(first_trade_price-candle['open'][j])>1e-8*first_trade_price:
                raise ValueError('first traded second disagrees with 15m open')
        coarse={k:v[j:j+64] for k,v in candle.items()}
        model_entry=float(candle['open'][j]);target_fraction=abs(target-entry)/entry
        model=replay(coarse,entry=model_entry,direction=signal['direction'],
            stop=model_entry*(1-sign*.01),target=model_entry*(1+sign*target_fraction),end=end)
        return dict(id=signal['id'],symbol=signal['symbol'],direction=signal['direction'],
            bar_seconds=signal['strategy_bar_seconds'],opened_at=opened,reported_r=signal['realized_r'],
            reported_status=signal['status'],entry=entry,planned_open=model_entry,
            entry_drift_pct=100*(entry/model_entry-1),publication_delay_s=opened-entry_time,
            minute_replay=whole,post_publication_replay=causal,coarse_open_replay=model,
            fractional_second_ambiguous=boundary_ambiguous,
            first_traded_second=first_trade,first_traded_price=first_trade_price,
            first_trade_after_publication=(first_trade>opened if first_trade is not None else None),
            first_trade_delay_s=(first_trade-entry_time if first_trade is not None else None),
            minute_window_sha256=mhash,second_window_sha256=shash,
            entry_minutes_sha256=emhash,first_trade_seconds_sha256=thash)
    rows=[]
    with ThreadPoolExecutor(max_workers=3) as pool:
        for row in pool.map(inspect,sorted(signals,key=lambda r:r['opened_at'])):
            rows.append(row);print(json.dumps(row),flush=True)
    summary=dict(n=len(rows),reported_r=sum(r['reported_r'] or 0 for r in rows),
        whole_minute_r=sum(r['minute_replay']['r'] for r in rows),
        post_publication_r=sum(r['post_publication_replay']['r'] for r in rows),
        coarse_next_open_r=sum(r['coarse_open_replay']['r'] for r in rows),
        fractional_second_ambiguous=sum(r['fractional_second_ambiguous'] for r in rows),
        next_open_first_trade_after_publication=sum(r['first_trade_after_publication'] is True for r in rows),
        changed_by_minute_boundary=sum(abs(r['minute_replay']['r']-r['post_publication_replay']['r'])>1e-7 for r in rows))
    report=dict(status='PAIRED_TIMING_DIAGNOSTIC_NOT_ACCOUNT_PNL',summary=summary,rows=rows,
        limitations=['Same actual signals, no portfolio rerouting or new filter selection.',
            'Native history OHLC is not an executable market fill or resting order guarantee.',
            'Less than one second after publication is omitted; level touches there are flagged.',
            'First-touch time is bounded by the candle duration; stop first if both levels touch.',
            'Historical monitoring/rule versions differ; no blind rewriting of old results.'],
        sources={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (manifest,old_path,export,Path(__file__),Path('public_candle_window.py'))})
    (ROOT/'stock_live_replay_parity_nov16.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
