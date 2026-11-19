"""Explicit revision: mark-only macro indicators plus first opening impulse."""
from bisect import bisect_right
import math
import numpy as np
from us_cash_session_nov19 import session_bounds
from stock_session_entries_nov19 import (SessionContext,FAMILIES as BASE_FAMILIES,
    candidates as base_candidates,bracket,macro_allowed,HORIZON)

FAMILIES=BASE_FAMILIES+('open_range_impulse',)


class MarkSessionMacro:
    """Reference-only mark context; never supplies an executable entry or VWAP."""
    def __init__(self,context):self.context=context;self.memo={}
    def at(self,known_at):
        c=self.context;trade=c.trade;mark=c.mark
        i=c.lookup.get(int(known_at)-900)
        if i is None:return None
        if i in self.memo:return self.memo[i]
        stamp=int(trade['time'][i]);bounds=session_bounds(stamp)
        if not bounds or not bounds[0]<=stamp<bounds[1]:return None
        history=c.regular[:bisect_right(c.regular,i)]
        if len(history)<15:return None
        recent=history[-15:];mids=[c.mark_lookup.get(trade['time'][k]) for k in recent]
        if any(m is None for m in mids):return None
        tr=[]
        for before,m in zip(mids,mids[1:]):
            previous=mark['close'][before]
            tr.append(max(mark['high'][m]-mark['low'][m],
                abs(mark['high'][m]-previous),abs(mark['low'][m]-previous)))
        atr=float(np.mean(tr))
        if not atr>0:return None
        times=list(range(int(bounds[0]),stamp+900,900));session=[c.mark_lookup.get(t) for t in times]
        if any(m is None for m in session):return None
        m=session[-1];close=float(mark['close'][m]);opening=float(mark['open'][session[0]])
        typical=[sum(float(mark[k][m]) for k in ('high','low','close'))/3 for m in session]
        result=dict(known_at=known_at,close=close,opening=opening,atr=atr,
            vwap=float(np.mean(typical)),session_return_atr=(close-opening)/atr,
            reference_kind='native mark typical15m-bar unweighted current-session mean; not VWAP')
        self.memo[i]=result;return result


def candidates(context,symbol,spec,*,references,diagnostics=None):
    refs=[MarkSessionMacro(c) for c in references]
    diag=diagnostics if diagnostics is not None else {}
    control=base_candidates(context,symbol,spec,references=refs,diagnostics=diag)
    seen={(r['symbol'],r['entry_time']) for r in control};extras=[];used=set()
    mark=context.mark;trade=context.trade
    def note(key):diag[key]=diag.get(key,0)+1
    for i,stamp in enumerate(trade['time']):
        known_at=int(stamp)+900;p=context.at(known_at)
        if p is None:continue
        start=p['start']
        if not start+1800<=known_at<min(start+5*3600,p['end']-1800):continue
        indices=p['indices']
        if len(indices)<3 or any(trade['volume'][k]<=0 for k in indices[:2]):continue
        initial=[context.mark_lookup[trade['time'][k]] for k in indices[:2]]
        or_high=max(float(mark['high'][m]) for m in initial);or_low=min(float(mark['low'][m]) for m in initial)
        if not .002<=(or_high-or_low)/p['close']<=.04:continue
        m=p['mark_index'];previous=context.mark_lookup.get(int(stamp)-900)
        if previous is None:continue
        o,h,l,close=[float(mark[k][m]) for k in ('open','high','low','close')]
        prior=float(mark['close'][previous]);atr=p['atr'];width=h-l
        sign=1 if prior<=or_high and close>or_high+.1*atr else -1 if prior>=or_low and close<or_low-.1*atr else 0
        if not sign or (start,sign) in used:continue
        # First actual closed crossing only, rather than restarting a setup
        # repeatedly while the price is already outside the opening range.
        used.add((start,sign));note('first_impulse_crossing')
        level=or_high if sign>0 else or_low
        location=(close-l if sign>0 else h-close)/width if width>0 else .5
        if (sign*(close-o)<.5*atr or location<.75 or p['volume_ratio']<1.5
                or sign*(close-level)>atr):note('impulse_shape_rejected');continue
        if not macro_allowed(refs,known_at,sign,inverse=symbol=='SOXSUSDT'):
            note('impulse_macro_rejected');continue
        row=bracket(p,mark,trade,symbol,spec,sign,l if sign>0 else h,'open_range_impulse',or_high,or_low)
        if row is None:note('impulse_bracket_rejected');continue
        if (symbol,known_at) in seen:note('impulse_lower_priority_duplicate');continue
        extras.append(row);note('published_open_range_impulse')
    return sorted(control+extras,key=lambda r:(r['entry_time'],FAMILIES.index(r['family'])))
