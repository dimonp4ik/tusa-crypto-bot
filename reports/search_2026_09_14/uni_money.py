"""Twenty more coins: what the account does with them, and whether the venue would allow them.

The extra coins add roughly 2,145 bank signals and 385 rule signals, nearly doubling the book. Their
history starts in 2024 while the original sixteen reach back to 2022, so the comparison runs on the
window all of them cover.

Their fill cost is unmeasured and assumed at the worst value the sampler found (0.001540, XLM's) -
which is deliberately harsh, because XLM's spread is far above the live gate and XLM is refused. If
the new coins are that thin, the bot would refuse them too, so a snapshot of their books is taken
here as well.
"""
import collections, csv, datetime, json, pickle, sys, urllib.request, time
import numpy as np
sys.path.insert(0, 'C:/Users/Lenovo/Desktop/Торговля/crypto-bot')
import struct_params as SP
import gauntlet2 as G
import sim_w
import live_rules_sim as L
import universe as UNI
import exec_model as EM

EXTRA = pickle.load(open('extra_coins.pkl', 'rb'))
BASE16 = [s for s in SP.COINS if s not in ('BILLUSDT', 'AAVEUSDT', 'XLMUSDT')]
LO = int(datetime.datetime(2024, 1, 15, tzinfo=datetime.UTC).timestamp())
YT = EM.YT
print('  опорных монет %d, добавляемых %d, окно с 2024-01' % (len(BASE16), len(EXTRA)), flush=True)

def book(coins):
    sig = {s: EM.signals(s, True) for s in coins}
    saved = EM.SIG_ALL
    EM.SIG_ALL = sig
    rows = EM.book(coins, sig, 'рынок')[0]
    EM.SIG_ALL = saved
    return [r for r in rows if r[0] >= LO]

A = book(BASE16)
B = book(BASE16 + EXTRA)
print('  сделок: опора %d, с расширением %d (+%.0f%%)'
      % (len(A), len(B), 100 * (len(B) / len(A) - 1)), flush=True)

print('', flush=True)
print('  === при фиксированном риске ===', flush=True)
for risk in (0.010, 0.0125, 0.014, 0.0155, 0.017):
    ra, rb = sim_w.simulate(A, risk), sim_w.simulate(B, risk)
    print('    %.2f%%  опора $%6.0f / %4.1f%%   расширение $%6.0f / %4.1f%%   %+5.0f%%'
          % (100 * risk, ra['eq'], 100 * abs(L.dd_of(ra['curve'])),
             rb['eq'], 100 * abs(L.dd_of(rb['curve'])),
             100 * (rb['eq'] / ra['eq'] - 1)), flush=True)

print('', flush=True)
print('  === при равной просадке 12%% ===', flush=True)
for lbl, rows in (('опора 13 монет', A), ('расширение 33 монеты', B)):
    k, m = sim_w.money_at_dd(rows, 0.12)
    v = np.array([x[2] for x in rows])
    print('    %-22s %4d сд. ВР%5.1f%% ср%+.4f риск %.3f%% $%6.0f'
          % (lbl, len(v), 100 * np.mean(v > 0), v.mean(), 100 * k, m), flush=True)

print('', flush=True)
print('  === что дают сами новые монеты ===', flush=True)
new = [r for r in B if r[4] in EXTRA]
old = [r for r in B if r[4] not in EXTRA]
for lbl, v in (('старые 13', old), ('новые 20', new)):
    a = np.array([x[2] for x in v])
    print('    %-10s n%4d ВР%5.1f%% ср%+.4fR сумма%+7.1fR'
          % (lbl, len(a), 100 * np.mean(a > 0), a.mean(), a.sum()), flush=True)
per = collections.defaultdict(list)
for r in new:
    per[r[4]].append(r[2])
print('    по монетам:', flush=True)
for s, v in sorted(per.items(), key=lambda kv: -np.mean(kv[1])):
    print('      %-6s n%4d ВР%5.1f%% ср%+.4f' % (s.replace('USDT', ''), len(v),
                                                 100 * np.mean(np.array(v) > 0), np.mean(v)),
          flush=True)

print('', flush=True)
print('  === снимок стакана новых монет прямо сейчас ===', flush=True)
UA = {'User-Agent': 'Mozilla/5.0'}
req = urllib.request.Request('https://www.okx.com/api/v5/public/instruments?instType=FUTURES',
                             headers=UA)
body = json.loads(urllib.request.urlopen(req, timeout=25).read().decode())
inst = {x['instId'].split('-')[0]: x['instId'] for x in body.get('data', [])
        if '_UM_XPERP-' in x.get('instId', '') and x.get('state') == 'live'}
print('    монета   спред      проходит отсечку 0.05%%?', flush=True)
ok_cnt = 0
for s in EXTRA:
    base = s.replace('USDT', '')
    iid = inst.get(base)
    if not iid:
        continue
    try:
        r = urllib.request.Request('https://www.okx.com/api/v5/market/books?instId=%s&sz=20' % iid,
                                   headers=UA)
        d = json.loads(urllib.request.urlopen(r, timeout=15).read().decode()).get('data') or []
        if not d or not d[0].get('bids'):
            continue
        bid, ask = float(d[0]['bids'][0][0]), float(d[0]['asks'][0][0])
        sp = (ask - bid) / ((ask + bid) / 2)
        ok = sp <= 0.0005
        ok_cnt += int(ok)
        print('    %-7s  %.6f   %s' % (base, sp, 'да' if ok else 'НЕТ'), flush=True)
        time.sleep(0.15)
    except Exception as e:
        print('    %-7s  ошибка %s' % (base, str(e)[:30]), flush=True)
print('    проходят отсечку: %d из %d (один снимок, не медиана)' % (ok_cnt, len(EXTRA)), flush=True)
