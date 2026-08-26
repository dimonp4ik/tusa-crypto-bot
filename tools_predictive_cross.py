"""Does the outcome model survive a change of REGIME, not just a change of half?

tools_predictive.py fits on the first half of one window and scores the second.
That answers "is there signal here", but both halves share a market. The
question that decides whether coefficients can be shipped into the live bot is
harder: fit on one window, score on a DIFFERENT one, years apart.

Every parameter re-examined on 2026-08-26 that looked good on a single window
failed this way — per-coin tiers, the counter-structure boost, the bottom score
band. A model is a parameter with twenty knobs, so it deserves the same test,
not an easier one.

Usage: python tools_predictive_cross.py bt_w2023.csv bt_w2024.csv bt_wcur.csv
"""
import sys
import numpy as np
from tools_predictive import load, featurize, fit_logit, predict, auc, NUM, CAT


def aligned_featurize(train_rows, test_rows):
    """Featurise both books against ONE shared column layout.

    featurize() derives its categorical levels from the rows it is given, so
    calling it twice yields two different column orders and silently scores the
    test set with weights that belong to other features. Build the layout from
    the union instead, then split it back.
    """
    n = len(train_rows)
    X, y, names = featurize(train_rows + test_rows)
    return X[:n], y[:n], X[n:], y[n:], names


def report(train_path, test_path):
    tr, te = load(train_path), load(test_path)
    Xtr, ytr, Xte, yte, names = aligned_featurize(tr, te)
    mu, sd = Xtr.mean(0), Xtr.std(0)
    sd[sd == 0] = 1.0
    w = fit_logit((Xtr - mu) / sd, ytr)
    a_in = auc(ytr, predict(w, (Xtr - mu) / sd))
    p = predict(w, (Xte - mu) / sd)
    a_out = auc(yte, p)

    # Money question, not just ranking: size by predicted probability and see
    # whether the book earns more per unit of the SAME risk.
    net = np.array([float(r["net_r"]) for r in te])
    mult = np.clip(p / p.mean(), 0.6, 1.6)
    base, weighted = net.sum(), float((net * mult).sum())
    # normalise to equal deployed size so this cannot win by leverage alone
    weighted_eq = weighted / mult.mean()
    print(f"  {train_path:14s} -> {test_path:14s}  "
          f"AUC вне выборки {a_out:.4f}  (внутри {a_in:.4f})  "
          f"прибыль {base:+.1f}R -> {weighted_eq:+.1f}R при равном размере")
    return a_out, base, weighted_eq


def main():
    paths = sys.argv[1:] or ["bt_w2023.csv", "bt_w2024.csv", "bt_wcur.csv"]
    print("Перекрёстная проверка: обучение на одном окне, счёт на другом\n")
    outs = []
    for a in paths:
        for b in paths:
            if a == b:
                continue
            outs.append(report(a, b))
    aucs = [o[0] for o in outs]
    gains = [(o[2] - o[1]) / abs(o[1]) * 100 for o in outs if o[1]]
    print(f"\nAUC вне выборки: медиана {np.median(aucs):.4f}, "
          f"худший {min(aucs):.4f}, лучший {max(aucs):.4f}")
    print(f"прирост при равном размере: медиана {np.median(gains):+.1f}%, "
          f"худший {min(gains):+.1f}%, лучший {max(gains):+.1f}%")
    print("\n0.50 = монета. Внедрять только если вне выборки устойчиво выше "
          "и прирост не отрицателен ни в одной паре.")


# ── Result, 2026-08-26 ───────────────────────────────────────────────────────
#   train        test        AUC out   profit at equal size
#   2023      -> 2024        0.5053    +161.1R -> +163.6R
#   2024      -> 2023        0.5068     +96.0R ->  +93.2R
#   2023      -> current     0.5364    +369.5R -> +385.6R
#   2024      -> current     0.5383    +369.5R -> +384.9R
#   current   -> 2023        0.5353     +96.0R ->  +99.9R
#   current   -> 2024        0.5445    +161.1R -> +166.8R
#   in-sample AUC 0.62-0.63 throughout
#
# NOT shipped. Three reasons, in order of weight:
#
# 1. The two HISTORICAL-to-HISTORICAL transfers give 0.5053 and 0.5068 — coin
#    flips. Every number above 0.53 has the current window on one side of it,
#    and the current window is where the whole config was tuned, so its trades
#    are selected by rules fitted there. That is a leak, not a transfer.
# 2. One pair of six LOSES money (-2.8%). The bar was set before running:
#    ship only if no pair is negative.
# 3. In-sample 0.62 against out-of-sample 0.51-0.54 is the gap of a model
#    memorising its window.
#
# The earlier +4-4.5% came from splitting ONE window into halves. That is the
# same measurement that made per-coin tiers, the counter-structure boost and
# the bottom score band look good, and all three failed once a second regime
# was asked. A model is a parameter with twenty knobs; it gets the harder test,
# not the easier one.

if __name__ == "__main__":
    main()
