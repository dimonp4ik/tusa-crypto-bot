"""Causal bar library for high-WR search. Bars from merged 15m, complete only.
Signal on CLOSED bar i -> entry open[i+1] with 8bps adverse; costs 4bps round trip.
Equal notional per trade: pnl = pct return net. One position per symbol."""
import numpy as np, datetime, bisect
PIN=['BTCUSDT','ETHUSDT','XRPUSDT','SOLUSDT','DOTUSDT','XLMUSDT','LINKUSDT','SUIUSDT','HYPEUSDT','ZECUSDT','SEIUSDT','AAVEUSDT','TAOUSDT','NEARUSDT','BILLUSDT','LABUSDT','ADAUSDT','AVAXUSDT']
U='SOLUSDT,ADAUSDT,ETHUSDT,LINKUSDT,BTCUSDT,XRPUSDT,XLMUSDT,SUIUSDT,AVAXUSDT,DOTUSDT,HYPEUSDT,AAVEUSDT,TAOUSDT,ZECUSDT,SEIUSDT,DOGEUSDT,WLDUSDT,BNBUSDT,LABUSDT,UNIUSDT,LITUSDT,PUMPUSDT,PEPEUSDT,ENAUSDT,ONDOUSDT,NEARUSDT,BILLUSDT,FILUSDT,BICOUSDT,TRUMPUSDT,UBUSDT,ARBUSDT,BEATUSDT,OPUSDT,BCHUSDT,OUSDT,CAPUSDT,SHIBUSDT,XPLUSDT,LTCUSDT'.split(',')
OTHER=[s for s in U if s not in PIN]
# Prepared 15m bars. They used to live in a temp scratchpad, which is wiped
# without warning, so the default is now the copy next to this file. Set
# BARS_DIR to point elsewhere. Rebuild with download_xperp_history.py.
import os as _os
D = _os.getenv('BARS_DIR') or (_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), 'bars') + _os.sep)
ADV=0.0008; COST=0.0004
_c={}
def bars(sym,tf):
    k=(sym,tf)
    if k in _c: return _c[k]
    z=np.load(D+sym+'.npz'); T=z['T']//1000 if z['T'][0]>1e11 else z['T']; A=z['A']
    m=tf//900; key=T-T%tf
    # group
    idx=np.r_[0,np.nonzero(np.diff(key))[0]+1,len(T)]
    out_t=[];out=[]
    for a,b in zip(idx[:-1],idx[1:]):
        if b-a!=m or T[b-1]-T[a]!=(m-1)*900: continue
        out_t.append(key[a]); out.append((A[a,0],A[a:b,1].max(),A[a:b,2].min(),A[b-1,3],A[a:b,4].sum()))
    r=(np.array(out_t,dtype=np.int64),np.array(out,dtype=float)); _c[k]=r; return r
def ema(x,n):
    a=2/(n+1); e=np.empty_like(x); e[0]=x[0]
    for i in range(1,len(x)): e[i]=a*x[i]+(1-a)*e[i-1]
    return e
def sma(x,n):
    o=np.full(len(x),np.nan); c=np.cumsum(np.r_[0,x]); o[n-1:]=(c[n:]-c[:-n])/n; return o
def rsi(c,n):
    d=np.diff(c,prepend=c[0]); up=np.where(d>0,d,0.); dn=np.where(d<0,-d,0.)
    a=1/n; ru=np.empty_like(c); rd=np.empty_like(c); ru[0]=rd[0]=0
    for i in range(1,len(c)): ru[i]=ru[i-1]+a*(up[i]-ru[i-1]); rd[i]=rd[i-1]+a*(dn[i]-rd[i-1])
    return 100-100/(1+ru/np.maximum(rd,1e-12))
def atr(A,n=14):
    h,l,c=A[:,1],A[:,2],A[:,3]; pc=np.r_[c[0],c[:-1]]
    tr=np.maximum(h-l,np.maximum(abs(h-pc),abs(l-pc))); return sma(tr,n)
_btc=None
def btc_mom(ts,days=20):
    """BTC return over `days` using only daily bars closed by ts."""
    global _btc
    if _btc is None:
        T,A=bars('BTCUSDT',86400); _btc=(list(T),A[:,3])
    T,C=_btc; j=bisect.bisect_right(T,ts-86400)-1
    if j<days: return np.nan
    return C[j]/C[j-days]-1
def year(t): return datetime.datetime.fromtimestamp(int(t),datetime.UTC).year
def stats(tr):
    """tr: list of (entry_ts, exit_ts, pnl). pnl pct net."""
    if not tr: return dict(n=0)
    tr=sorted(tr,key=lambda x:x[1]); p=np.array([x[2] for x in tr])
    eq=np.cumsum(p); dd=(eq-np.maximum.accumulate(np.r_[0,eq])[1:]).min()
    Y={}
    for x in tr: Y.setdefault(year(x[0]),[]).append(x[2])
    return dict(n=len(p),wr=100*np.mean(p>0),sum=100*p.sum(),avg=100*p.mean(),dd=100*dd,worst=100*p.min(),
                yrs={y:(len(v),100*np.mean(np.array(v)>0),100*sum(v)) for y,v in sorted(Y.items())})
def fmt(s):
    if not s['n']: return 'n=0'
    y=' '.join(f"{k%100}:{v[1]:.0f}%/{v[2]:+.0f}" for k,v in s['yrs'].items())
    return f"n={s['n']:5d} WR={s['wr']:5.1f}% sum={s['sum']:+7.1f}% avg={s['avg']:+.3f}% dd={s['dd']:+6.1f}% worst={s['worst']:+.1f}% | {y}"
