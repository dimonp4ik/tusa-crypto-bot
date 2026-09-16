"""Independent re-implementation on 15m paths. Rule: live 4h trend holds side X at the close
of a 1h bar, and a 1h pullback condition fires -> enter next 1h open (+8bps adverse),
TP = e +/- tp*ATR14(1h), SL = e -/+ sl*ATR14(1h); walk 15m bars (stop wins ties) for up to
H hours; time exit at close. Stress: TP must be exceeded by `over` (maker queue), stop filled
`slip` worse. One position per symbol. Costs 4bps."""
import numpy as np, sys, datetime, itertools
from mrlib import PIN, D, bars, atr, rsi
import os
# Resolve next to this file, not the working directory: the research scripts
# live in reports/search_*/ and used to fail here purely because of where
# they were started from. EVT still overrides.
_EVT = os.environ.get('EVT') or os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ev3t.npz')
z=np.load(_EVT); S=z['sym']; TT=z['t']; TF4=z['tf4']
cache={}
def prep(s):
    if s in cache: return cache[s]
    T,A=bars(s,3600); a=atr(A); r2=rsi(A[:,3],2); r14=rsi(A[:,3],14)
    q=np.load(D+s+'.npz'); t15=q['T']; A15=q['A']
    m=S==s; tf=dict(zip(TT[m].tolist(),TF4[m].tolist()))
    cache[s]=(T,A,a,r2,r14,t15,A15,tf); return cache[s]
def run(side,cond,tp,sl,H=48,over=0.0,slip=0.0,uni=None):
    out=[]
    for s in (uni or np.unique(S)):
        T,A,a,r2,r14,t15,A15,tf=prep(s)
        pos15=dict(zip(t15.tolist(),range(len(t15))))
        busy=0
        for i in range(len(T)-1):
            t=int(T[i])
            if t+3600<busy: continue
            st=tf.get(t,0)
            if st!=(1 if side=='L' else -1): continue
            if not cond(r2[i],r14[i]) or np.isnan(a[i]): continue
            k=pos15.get(t+3600)
            if k is None or k+4*H>len(t15) or t15[k+4*H-1]-t15[k]!=(4*H-1)*900: continue
            o,h,l,c=A15[k:k+4*H,0],A15[k:k+4*H,1],A15[k:k+4*H,2],A15[k:k+4*H,3]
            e=o[0]*(1.0008 if side=='L' else 0.9992)
            if side=='L': TP=e+tp*a[i]; SL=e-sl*a[i]; hs=l<=SL; ht=h>=TP*(1+over)
            else: TP=e-tp*a[i]; SL=e+sl*a[i]; hs=h>=SL; ht=l<=TP*(1-over)
            js=np.argmax(hs) if hs.any() else 10**9; jt=np.argmax(ht) if ht.any() else 10**9
            if js<=jt and js<10**9:
                fill=min(SL,o[js]) if side=='L' else max(SL,o[js]); fill*= (1-slip) if side=='L' else (1+slip)
                j=js; p=fill/e-1 if side=='L' else 1-fill/e
            elif jt<10**9: j=jt; p=tp*a[i]/e
            else: j=4*H-1; p=c[-1]/e-1 if side=='L' else 1-c[-1]/e
            out.append((t,p-0.0004)); busy=int(t15[k+j])+900
    return out
def rep(tr):
    if not tr: return 'n=0'
    p=np.array([x[1] for x in tr]); o=np.argsort([x[0] for x in tr]); eq=np.cumsum(p[o]); dd=(eq-np.maximum.accumulate(np.r_[0,eq])[1:]).min()
    Y={}
    for t,pp in tr: Y.setdefault(datetime.datetime.fromtimestamp(t,datetime.UTC).year,[]).append(pp)
    ys=' '.join(f"{y%100}:{len(v)}/{100*np.mean(np.array(v)>0):.0f}%/{100*sum(v):+.0f}" for y,v in sorted(Y.items()))
    return f"n={len(p):5d} WR={100*(p>0).mean():5.1f}% sum={100*p.sum():+7.1f}% dd={100*dd:+6.1f}% | {ys}"
if __name__=='__main__':
    OTH=[s for s in np.unique(S) if s not in PIN]
    base=lambda r2,r14: r2<5
    print('BASE L rsi2<5 0.5/3 PIN ',rep(run('L',base,0.5,3,uni=PIN)),flush=True)
    print('BASE L rsi2<5 0.5/3 OTH ',rep(run('L',base,0.5,3,uni=OTH)),flush=True)
    for over,slip in ((0.0005,0),(0.001,0),(0,0.002),(0.001,0.002)):
        print(f'STRESS over={over} slip={slip} PIN',rep(run('L',base,0.5,3,over=over,slip=slip,uni=PIN)),flush=True)
    for th,tp,sl in itertools.product((3,5,7,10),(0.4,0.5,0.6),(2.5,3,4)):
        c=lambda r2,r14,th=th: r2<th
        print(f'GRID rsi2<{th:2d} tp={tp} sl={sl} PIN',rep(run('L',c,tp,sl,uni=PIN)),flush=True)
