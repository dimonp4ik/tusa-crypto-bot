"""Stop searching pairs by hand: let a model search every combination of all 39 features at once.

Gradient-boosted trees on the 166,713 gate-eligible coin-hours (features + side + coin), predicting
each hour's R. Walk-forward: for every half-year from 2023 on, fit only on the past, predict the next
six months. The selection cutoff is a percentile of the TRAINING predictions, so nothing about the
test half-year leaks into which trades are taken. Selected hours then go through one-position-per-
coin using the real exit times, because hours are not trades.

Control: the same pipeline with outcomes shuffled inside each side of the training set. Whatever the
control shows is what the pipeline produces from noise.

Reference points on the same folds: every eligible hour, and the pullback rule.
"""
import datetime, pickle, sys
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

rows = pickle.load(open('wide_cache.pkl', 'rb'))
names = sorted(rows[0][5])
ts = np.array([r[0] for r in rows], dtype=np.int64)
te = np.array([r[1] for r in rows], dtype=np.int64)
coins = sorted({r[2] for r in rows})
cid = {c: i for i, c in enumerate(coins)}
coin = np.array([cid[r[2]] for r in rows])
lg = np.array([r[3] for r in rows], dtype=float)
R = np.array([r[4] for r in rows], dtype=float)
X = np.column_stack([np.array([r[5][n] for r in rows], dtype=float) for n in names] + [lg, coin])
del rows
cat = np.zeros(X.shape[1], dtype=bool); cat[-1] = True
jr48, jr6 = names.index('rsi48'), names.index('rsi6')
pull = (X[:, jr48] >= 56.3761) & (X[:, jr6] <= 33.7947) & (lg == 1)
print('  строк %d, признаков %d (+сторона, +монета), монет %d' % (len(R), len(names), len(coins)), flush=True)


def dedupe(idx):
    idx = idx[np.argsort(ts[idx])]
    busy, keep = {}, []
    for i in idx:
        if busy.get(coin[i], 0) > ts[i]:
            continue
        busy[coin[i]] = te[i]
        keep.append(i)
    return np.array(keep, dtype=int)


def fold_edges():
    out, y, h = [], 2023, 1
    while True:
        a = datetime.datetime(y, 1 if h == 1 else 7, 1, tzinfo=datetime.UTC)
        b = datetime.datetime(y + (h == 2), 7 if h == 1 else 1, 1, tzinfo=datetime.UTC)
        a, b = int(a.timestamp()), int(b.timestamp())
        if a > ts.max():
            break
        out.append((a, b)); y, h = (y, 2) if h == 1 else (y + 1, 1)
    return out


QS = (0.99, 0.97, 0.93)
rng = np.random.default_rng(1)
res = {('model', q): [] for q in QS}
res.update({('null', q): [] for q in QS})
res['all'], res['pull'] = [], []
for a, b in fold_edges():
    tr, tst = ts < a - 2 * 86400, (ts >= a) & (ts < b)
    if tst.sum() < 500:
        continue
    for kind in ('model', 'null'):
        y = R[tr].copy()
        if kind == 'null':
            for s in (0, 1):
                m = lg[tr] == s
                y[m] = rng.permutation(y[m])
        mdl = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
                                            min_samples_leaf=200, l2_regularization=1.0,
                                            categorical_features=cat, random_state=0)
        mdl.fit(X[tr], y)
        ptr, pte = mdl.predict(X[tr]), mdl.predict(X[tst])
        for q in QS:
            cut = np.quantile(ptr, q)
            sel = dedupe(np.flatnonzero(tst)[pte > cut])
            res[(kind, q)].append((a, R[sel]))
    res['all'].append((a, R[dedupe(np.flatnonzero(tst))]))
    res['pull'].append((a, R[dedupe(np.flatnonzero(tst & pull))]))
    print('    полугодие %s готово' % datetime.datetime.fromtimestamp(a, datetime.UTC).strftime('%Y-%m'), flush=True)

months = 6.0
print('', flush=True)
print('  вариант                сделок/год   ВР     ср R     R в месяц   полугодий в плюсе   по полугодиям (ср R)', flush=True)
def show(lbl, lst):
    allv = np.concatenate([v for _, v in lst]) if lst else np.zeros(0)
    if not len(allv):
        print('  %-22s нет сделок' % lbl, flush=True); return
    n_half = len(lst)
    pos = sum(1 for _, v in lst if len(v) and v.mean() > 0)
    print('  %-22s %8.0f    %5.1f%%  %+.4f   %+7.2f        %d/%d          %s'
          % (lbl, len(allv) / (n_half / 2), 100 * np.mean(allv > 0), allv.mean(),
             allv.sum() / (n_half * months), pos, n_half,
             ' '.join('%+.2f' % v.mean() if len(v) else '  -  ' for _, v in lst)), flush=True)
show('все часы гейта', res['all'])
show('правило отката', res['pull'])
for q in QS:
    show('модель, топ %.0f%%' % (100 * (1 - q)), res[('model', q)])
for q in QS:
    show('КОНТРОЛЬ шум, топ %.0f%%' % (100 * (1 - q)), res[('null', q)])
