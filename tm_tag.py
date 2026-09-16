"""No resting orders: bot watches price every second.
Entry: after the signal, if price touches the level within 15 min -> MARKET buy at level*(1+slip)
       (bar opens through the level -> market at open*(1+slip)). Otherwise no trade.
TP: touch -> MARKET exit at TP*(1-slip) (no TP credit inside a touch-entry bar).
Stop: exchange stop-market, gap-through at open, extra slip. Taker both sides: cost 4bps."""
import sys, numpy as np
sys.path.insert(0,'C:/Users/Lenovo/Desktop/Торговля/crypto-bot')
from src import pullback_bank as PB
from mrlib import U, D, PIN
from bank15 import report
b=np.load(D+'BTCUSDT.npz'); bt,ba=b['T'],b['A']
data={s:np.load(D+s+'.npz') for s in U}
rules=PB.RULE_SETS['strict3+short2']
sigs={s:PB.hourly_signals(data[s]['T'],data[s]['A'],bt,ba,rules) for s in U}
def run(slip,cost=0.0004,H=PB.HOLD_BARS):
    out=[]
    for s in U:
        t=data[s]['T']; A=data[s]['A']; sig=sigs[s]; pos=dict(zip(t.tolist(),range(len(t)))); busy=0
        for close in sorted(sig):
            if close<busy: continue
            k=pos.get(close)
            if k is None or k+H>len(t): continue
            ri,lim,atr=sig[close]; r=rules[ri]; lg=r.get('side','LONG')=='LONG'
            o,h,l,c=(A[k:k+H,j] for j in range(4))
            if lg:
                if o[0]<=lim: e=o[0]*(1+slip); touch=False
                elif l[0]<=lim: e=lim*(1+slip); touch=True
                else: continue
                TP=e+r['tp']*atr; SL=e-r['sl']*atr; hs=l<=SL; ht=h>=TP
            else:
                if o[0]>=lim: e=o[0]*(1-slip); touch=False
                elif h[0]>=lim: e=lim*(1-slip); touch=True
                else: continue
                TP=e-r['tp']*atr; SL=e+r['sl']*atr; hs=h>=SL; ht=l<=TP
            if touch: ht=ht.copy(); ht[0]=False
            js=np.argmax(hs) if hs.any() else 10**9; jt=np.argmax(ht) if ht.any() else 10**9
            if js<=jt and js<10**9:
                f=(min(SL,o[js])*(1-slip)) if lg else (max(SL,o[js])*(1+slip)); j=js
            elif jt<10**9:
                f=TP*(1-slip) if lg else TP*(1+slip); j=jt
            else:
                j=H-1; f=c[-1]*(1-slip) if lg else c[-1]*(1+slip)
            p=(f/e-1 if lg else 1-f/e)-cost
            ex=int(t[k+j])+900; out.append((close,ex,p,r['sl']*atr/e,s,'LONG' if lg else 'SHORT',ri,('stop' if js<=jt and js<10**9 else ('take' if jt<10**9 else 'time')))); busy=ex
    return out
