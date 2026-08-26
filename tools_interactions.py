"""Search PAIRS of conditions, keep only what holds in every window.

One-feature sweeps were exhausted on 2026-08-26 and the surviving finding was
an interaction (London edge exists only when volume confirms), so pairs are
where the remaining room is. Pairs also overfit far faster than single
features — the number of candidates explodes while the subsets shrink — so the
filter is deliberately harsh:

  * the subset must clear MIN_N trades in EVERY window, not on average;
  * the lift in R/trade must carry the SAME SIGN in every window;
  * output is ranked by the WEAKEST window, never the mean, so one spectacular
    window cannot carry a pair.

Anything that only works in the current window is exactly what failed today
for per-coin tiers, the counter-structure boost and the outcome model.

Usage: python tools_interactions.py bt_w2023.csv bt_w2024.csv bt_wcur.csv
"""
import csv, sys, itertools, collections

MIN_N = 35
CATS = ["session", "trend_1h", "trend_4h", "direction", "entry_source", "swing_trend"]
NUMS = ["volume_ratio", "mtf_score", "rsi", "eff_ratio", "vol_atr_pct", "extension_atr"]


def load(p):
    return list(csv.DictReader(open(p, encoding="utf-8")))


def num(r, k):
    try:
        return float(r.get(k) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def rmean(g):
    return sum(float(r["net_r"] or 0) for r in g) / max(1, len(g))


def wrate(g):
    return sum(1 for r in g if r["outcome"] in ("TP1", "TP2", "TRAIL")) / max(1, len(g))


def conditions(rows):
    """Build the candidate predicates from one reference book."""
    out = []
    for c in CATS:
        for v in sorted({(r.get(c) or "") for r in rows}):
            if not v:
                continue
            out.append((f"{c}={v}", lambda r, c=c, v=v: (r.get(c) or "") == v))
    for n in NUMS:
        vals = sorted(num(r, n) for r in rows)
        for q, lab in ((0.5, "med"),):
            cut = vals[int(len(vals) * q)]
            out.append((f"{n}>={cut:g}", lambda r, n=n, cut=cut: num(r, n) >= cut))
            out.append((f"{n}<{cut:g}", lambda r, n=n, cut=cut: num(r, n) < cut))
    return out


def main():
    paths = sys.argv[1:] or ["bt_w2023.csv", "bt_w2024.csv", "bt_wcur.csv"]
    books = [(p, load(p)) for p in paths]
    conds = conditions(books[-1][1])
    print(f"окон: {len(books)}, условий: {len(conds)}, "
          f"пар: {len(conds)*(len(conds)-1)//2}, порог выборки: {MIN_N}\n")

    keep = []
    for (n1, f1), (n2, f2) in itertools.combinations(conds, 2):
        if n1.split("=")[0].split(">")[0].split("<")[0] == \
           n2.split("=")[0].split(">")[0].split("<")[0]:
            continue                      # same feature on both sides
        per = []
        ok = True
        for _, rows in books:
            sub = [r for r in rows if f1(r) and f2(r)]
            rest = [r for r in rows if not (f1(r) and f2(r))]
            if len(sub) < MIN_N or len(rest) < MIN_N:
                ok = False
                break
            per.append((len(sub), wrate(sub), rmean(sub) - rmean(rest), wrate(sub) - wrate(rest)))
        if not ok:
            continue
        signs = {1 if p[2] > 0 else -1 for p in per}
        if len(signs) > 1:
            continue                      # sign flips between windows
        weakest = min(abs(p[2]) for p in per) * (1 if per[0][2] > 0 else -1)
        keep.append((weakest, n1, n2, per))

    keep.sort(key=lambda x: -abs(x[0]))
    pos = [k for k in keep if k[0] > 0][:12]
    neg = [k for k in keep if k[0] < 0][:12]
    for title, group in (("ЛУЧШЕ книги во ВСЕХ окнах", pos),
                         ("ХУЖЕ книги во ВСЕХ окнах", neg)):
        print(f"=== {title} (ранг по слабейшему окну) ===")
        for weakest, n1, n2, per in group:
            cells = "  ".join(f"{n:4d}сд {w*100:5.1f}% {d:+.3f}" for n, w, d, _ in per)
            print(f"  {n1:22s} + {n2:22s} слабейший {weakest:+.3f} | {cells}")
        print()


if __name__ == "__main__":
    main()
