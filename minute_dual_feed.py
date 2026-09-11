"""Minute-close signal monitoring with separate X-Perp exchange protection."""
import os
os.environ['PYTHON_DOTENV_DISABLED']='1'
os.environ['BOT_STARTUP_ENABLED']='0'
import csv,json,hashlib
from pathlib import Path
import backtest as bt
from config import STOP_CLOSE_CONFIRM,STOP_EXCHANGE_BACKSTOP_R
from minute_zone_entries import fetch_minutes


def simulate(row,source,exchange,fill,end,per_side_cost=.0006):
    side=1 if row['direction']=='LONG' else -1
    planned=float(row['entry']);sl=float(row['sl']);tp1=float(row['tp1']);tp2=float(row['tp2'])
    risk=abs(planned-sl);entry=exchange[fill][0];atr=float(row['atr_at_signal'])
    if not (sl<entry<tp1 if side==1 else tp1<entry<sl):return {'skipped':'entry outside bracket'}
    backstop=entry-side*risk*max(1.,STOP_EXCHANGE_BACKSTOP_R) if STOP_CLOSE_CONFIRM else sl
    best=planned;reached=False;multiple=bt.TRAIL_ATR_MULT;last_group=None;group_h=group_l=None
    def done(t,price,reason):
        weight=float(row['size_mult'])
        return dict(entry_time=fill,exit_time=t,entry=entry,exit_price=price,reason=reason,
             net_r=(side*(price-entry)-per_side_cost*(entry+price))/risk*weight,
             cost_r=per_side_cost*(entry+price)/risk*weight)
    for t in range(fill,end,60):
        xo,xh,xl,xc=exchange[t];so,sh,slo,sc=source[t]
        # Orders already on exchange act before a new minute-close callback.
        if (xl<=backstop if side==1 else xh>=backstop):
            return done(t+60,min(xo,backstop) if side==1 else max(xo,backstop),'exchange_stop')
        if (xh>=tp2 if side==1 else xl<=tp2):return done(t+60,tp2,'exchange_tp2')
        group=t//900
        if group!=last_group:group_h=sh;group_l=slo;last_group=group
        else:group_h=max(group_h,sh);group_l=min(group_l,slo)
        best=max(best,sh) if side==1 else min(best,slo)
        reason=None
        if not reached and ((t+60)%900==0 if STOP_CLOSE_CONFIRM else True) and side*(sc-sl)<=0:
            reason='signal_stop'
        if side*((sh if side==1 else slo)-tp2)>=0:reason=reason or 'signal_tp2'
        if not reached and side*((sh if side==1 else slo)-tp1)>=0:
            reached=True
            multiple=bt._post_tp1_trail_mult_bt(row['direction'],planned,tp1,tp2,group_h,group_l,sc)
        if reached:
            trail=max(planned,best-atr*multiple) if side==1 else min(planned,best+atr*multiple)
            if side*(sc-trail)<=0:reason=reason or 'signal_trail'
        if reason:
            # Observation after close, market fill at next minute open, never at derived stop.
            return done(t+60,exchange[t+60][0],reason)
    return done(end,exchange[end][0],'expiry')


def main():
    folder=Path('reports/audit_2026_09_08');cache=folder/'dual_feed_minute_cache';cache.mkdir(exist_ok=True)
    entries=json.loads((folder/'minute_zone_entry_review.json').read_text())['results']
    with (folder/'venue_swap_tagged_raw.csv').open() as f:rows=list(csv.DictReader(f))
    results=[]
    for e in entries:
        row=rows[e['source_row_index']]
        fill=e['modeled_fill_time'];start=e['signal_bar'];end=start+48*3600
        item={'symbol':row['symbol'],'signal_bar':start,'original_outcome':row['outcome'],'original_net_r':float(row['net_r'])}
        try:
            base=row['symbol'][:-4]
            source,sh=fetch_minutes(base+'-USDT-SWAP',start,end+60,cache)
            exchange,xh=fetch_minutes(base+'-USD_UM_XPERP-310404',start,end+60,cache)
            item.update(simulate(row,source,exchange,fill,end));item.update(source_sha256=sh,exchange_sha256=xh)
        except Exception as exc:item['error']=str(exc)
        results.append(item)
        report={'status':'BOUNDED_EXECUTION_DIAGNOSTIC','results':results,
          'assumptions':['Same previously selected 12 entries; no outcome-based selection or optimization.',
           'Signal observations on SWAP minute closes, actual exchange stop/TP2 on X-Perp OHLC.',
           'One-minute observation discretization; callback fill at next minute open, not executable book.',
           '48h expiry, current configured trail profile, 0% partial TP1. Cost 0.05% taker + 0.01% slippage per side; no funding.',
           'No portfolio replay; aggregate sample PnL is not strategy profitability evidence.'],
          'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
        (folder/'minute_dual_feed_taker_review.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
        print(json.dumps(item),flush=True)
if __name__=='__main__':main()
