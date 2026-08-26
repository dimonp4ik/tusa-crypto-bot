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
# Both spellings: the crypto export calls it extension_atr, the stocks export
# bos_extension_atr. Whichever is absent is dropped by the constant-feature
# guard in conditions() rather than quietly becoming a column of zeros.
NUMS = ["volume_ratio", "mtf_score", "rsi", "eff_ratio", "vol_atr_pct",
        "extension_atr", "bos_extension_atr"]


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
    dropped = []
    for n in NUMS:
        if n not in rows[0]:
            dropped.append(f"{n} (нет колонки)")
            continue
        vals = sorted(num(r, n) for r in rows)
        cut = vals[len(vals) // 2]
        share = sum(1 for v in vals if v >= cut) / len(vals)
        # A missing или constant column silently becomes a column of zeros,
        # whose median is 0, whose ">=0" side is EVERY row. That produced a
        # pair reading "session=OPEN + extension_atr>=0" which was really just
        # "session=OPEN" — a degenerate condition presented as a finding, with
        # no warning. Anything that fails to split the book is dropped aloud.
        if len(set(vals)) < 2 or share > 0.95 or share < 0.05:
            dropped.append(f"{n} (не делит книгу: {share*100:.0f}%)")
            continue
        out.append((f"{n}>={cut:g}", lambda r, n=n, cut=cut: num(r, n) >= cut))
        out.append((f"{n}<{cut:g}", lambda r, n=n, cut=cut: num(r, n) < cut))
    if dropped:
        print("отброшены признаки: " + ", ".join(dropped))
    return out


def split_thirds(rows):
    """Chop one book into three consecutive stretches by entry time.

    For the stocks bot there are no historical windows at all — OKX only
    listed those perps in Feb-Mar 2026 — so cross-regime consistency cannot be
    demanded. Consecutive thirds are the weaker substitute: they still catch a
    pair that lives in one stretch of the window, which is the failure mode
    that killed per-coin tiers and the outcome model. They do NOT establish
    that a pair survives a change of market, and nothing found this way should
    be described as if they did.
    """
    rows = sorted(rows, key=lambda r: float(r.get("entry_time") or 0))
    n = len(rows) // 3
    return [("1/3", rows[:n]), ("2/3", rows[n:2 * n]), ("3/3", rows[2 * n:])]


def main():
    paths = [a for a in sys.argv[1:] if not a.startswith("--")]
    thirds = "--thirds" in sys.argv
    paths = paths or ["bt_w2023.csv", "bt_w2024.csv", "bt_wcur.csv"]
    if thirds:
        if len(paths) != 1:
            print("--thirds принимает ровно один файл")
            return
        books = split_thirds(load(paths[0]))
        print(f"режим третей: {paths[0]} разрезан на три последовательных отрезка")
        print("ВНИМАНИЕ: трети одного окна — НЕ замена трём режимам; "
              "они ловят пару, живущую в одном отрезке, но ничего не говорят "
              "об устойчивости к смене рынка" + chr(10))
    else:
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
        # Two numbers that decided real calls on 2026-08-26 and are NOT
        # implied by same-sign-everywhere:
        #
        # share  — the subset as a fraction of the book. Every sizing rule that
        #   survived end-to-end covered 8-15%; one covering 29% turned out to
        #   be leverage (profit up 18%, risk ratios flat or falling) even
        #   though its per-trade lift was the strongest found. A big subset
        #   cannot be sized selectively, because sizing it IS sizing the book.
        #
        # trend  — whether the lift grows or decays across the windows. "LONG
        #   with RSI<60" was negative in all three stretches and passed the
        #   sign filter, but decayed -0.422 -> -0.381 -> -0.145 while the
        #   subset itself still earned +0.423R. Acting on it would have cut
        #   live money to chase an effect already gone. Same sign everywhere
        #   does not separate a live effect from a dying one.
        share = sum(p[0] for p in per) / sum(len(rows) for _, rows in books)
        # MONOTONE movement toward zero, not merely "last below first". The
        # crude version flagged London-with-volume in the crypto book, whose
        # lift runs +0.463 -> +0.176 -> +0.327: a dip in the middle window
        # and a recovery, which is variation between regimes rather than an
        # effect dying. The stocks pair that genuinely died went -0.422 ->
        # -0.381 -> -0.145, shrinking at every step. Only that shape earns a
        # warning; the crude test condemned both.
        mags = [abs(x[2]) for x in per]
        trend = per[-1][2] - per[0][2]
        decaying = all(b < a for a, b in zip(mags, mags[1:]))
        keep.append((weakest, n1, n2, per, share, decaying, trend))

    keep.sort(key=lambda x: -abs(x[0]))
    pos = [k for k in keep if k[0] > 0][:12]
    neg = [k for k in keep if k[0] < 0][:12]
    for title, group in (("ЛУЧШЕ книги во ВСЕХ окнах", pos),
                         ("ХУЖЕ книги во ВСЕХ окнах", neg)):
        print(f"=== {title} (ранг по слабейшему окну) ===")
        for weakest, n1, n2, per, share, decaying, trend in group:
            cells = "  ".join(f"{n:4d}сд {w*100:5.1f}% {d:+.3f}" for n, w, d, _ in per)
            # flag the two traps rather than silently ranking past them
            warn = ""
            if share > 0.20:
                warn += f"  [ДОЛЯ {share*100:.0f}% — велика для множителя]"
            if decaying:
                warn += f"  [ЗАТУХАЕТ монотонно, {trend:+.3f}]"
            print(f"  {n1:22s} + {n2:22s} слабейший {weakest:+.3f} | {cells}{warn}")
        print()


if __name__ == "__main__":
    main()
