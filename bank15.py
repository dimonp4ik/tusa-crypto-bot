"""Bank = in-trend pullback rules on 15m paths (same engine as pb15, extended output).
Returns per trade: signal t, exit t, pnl (pct notional, net), risk pct (stop distance / entry).
R = pnl / risk (so a full stop ~ -1R). Portfolio: all trades, equal risk per trade."""
import numpy as np, datetime, sys
from pb15 import prep, S, PIN
def run(side,th,tp,sl,H=48,over=0.0,uni=None):
    out=[]
    for s in uni:
        T,A,a,r2,r14,t15,A15,tf=prep(s)
        pos15=dict(zip(t15.tolist(),range(len(t15)))); busy=0
        for i in range(len(T)-1):
            t=int(T[i])
            if t+3600<busy or tf.get(t,0)!=(1 if side=='L' else -1) or np.isnan(a[i]): continue
            if not ((side=='L' and r2[i]<th) or (side=='S' and r2[i]>th)): continue
            k=pos15.get(t+3600)
            if k is None or k+4*H>len(t15) or t15[k+4*H-1]-t15[k]!=(4*H-1)*900: continue
            o,h,l,c=(A15[k:k+4*H,j] for j in range(4))
            e=o[0]*(1.0008 if side=='L' else 0.9992)
            if side=='L': TP=e+tp*a[i]; SL=e-sl*a[i]; hs=l<=SL; ht=h>=TP*(1+over)
            else: TP=e-tp*a[i]; SL=e+sl*a[i]; hs=h>=SL; ht=l<=TP*(1-over)
            js=np.argmax(hs) if hs.any() else 10**9; jt=np.argmax(ht) if ht.any() else 10**9
            if js<=jt and js<10**9:
                f=min(SL,o[js]) if side=='L' else max(SL,o[js]); j=js; p=f/e-1 if side=='L' else 1-f/e
            elif jt<10**9: j=jt; p=tp*a[i]/e
            else: j=4*H-1; p=c[-1]/e-1 if side=='L' else 1-c[-1]/e
            p-=0.0004; ex=int(t15[k+j])+900
            out.append((t+3600,ex,p,sl*a[i]/e,s,side)); busy=ex
    return out
def report(name,tr):
    tr=sorted(tr,key=lambda x:x[1]); p=np.array([x[2] for x in tr]); R=np.array([x[2]/x[3] for x in tr])
    eq=np.cumsum(R); dd=(eq-np.maximum.accumulate(np.r_[0,eq])[1:]).min()
    Y={};M={};Dd={}
    for x,r in zip(tr,R):
        d=datetime.datetime.fromtimestamp(x[1],datetime.UTC)
        Y.setdefault(d.year,[]).append(r); M[d.strftime('%y-%m')]=M.get(d.strftime('%y-%m'),0)+r; Dd[x[1]//86400]=Dd.get(x[1]//86400,0)+r
    H={}
    for x in tr:
        d=datetime.datetime.fromtimestamp(x[0],datetime.UTC); H[(d.year,(d.month-1)//6)]=H.get((d.year,(d.month-1)//6),0)+1
    ev=sorted([(x[0],1) for x in tr]+[(x[1],-1) for x in tr]); cur=mx=0
    for _,dlt in ev: cur+=dlt; mx=max(mx,cur)
    mv=np.array(list(M.values()))
    print(f"{name}: n={len(R)} WR={100*(R>0).mean():.1f}% sumR={R.sum():+.1f} avgR={R.mean():+.3f} sum%notional={100*p.sum():+.0f} maxDD={dd:+.1f}R worst_day={min(Dd.values()):+.1f}R months+={int((mv>0).sum())}/{len(mv)} max_concurrent={mx}")
    print('   years:',' '.join(f"{y}:{len(v)}/{100*np.mean(np.array(v)>0):.0f}%/{sum(v):+.1f}R" for y,v in sorted(Y.items())))
    print('   per half-year trades:',' '.join(f"{y%100}H{h+1}:{n}" for (y,h),n in sorted(H.items())),flush=True)
if __name__=='__main__':
    OTH=[s for s in np.unique(S) if s not in PIN]
    over=float(sys.argv[1]) if len(sys.argv)>1 else 0.0
    for nm,uni in (('PIN',PIN),('OTH',OTH)):
        L=run('L',5,0.4,3,over=over,uni=uni); Sh=run('S',90,0.4,4,over=over,uni=uni)
        report(f'{nm} over={over} LONG  rsi2<5 tp.4 sl3',L)
        report(f'{nm} over={over} SHORT rsi2>90 tp.4 sl4',Sh)
        report(f'{nm} over={over} BANK',L+Sh)
