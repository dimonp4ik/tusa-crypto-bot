"""
Filter-variant A/B experiment (2026-07-24).

Question this answers: which filter configuration does Claude work BEST with?
The backtest can't answer it — it never calls Claude at all. So we measure live.

Design (single-verdict replay, not parallel Claude calls):
  Every setup that reaches Claude is judged ONCE. We tag it with the list of
  variant codes whose filter settings would ALSO have admitted it. Later, each
  variant's performance = its own subset of setups, replayed against the SAME
  Claude verdicts and the SAME shadow-tracked outcomes.

Why not run 5-9 separate Claude calls per scan:
  - Claude is non-deterministic — separate calls would judge different setups,
    confounding "variant B is better" with "Claude rolled differently".
  - It would split an already-thin signal stream (~27 calls/week) across arms,
    leaving each arm statistically useless.
  - Cost multiplies for no added information.
Single-verdict replay keeps arms on identical verdicts, so the only difference
between them is the filter rule itself.

STRICTER arms (B/E/G/H/I) are a plain subset of what live already admits, so
they replay directly against real-signal outcomes.

LOOSER arms (D/F) need setups the live filter REJECTED — which Claude
would never see and nothing would ever log. The shadow mechanism (2026-07-25)
closes that gap: signal_filter.analyze_coin_smc() takes include_shadow=True
(live only; the backtests never pass it) and routes selected gates through
_soft_fail(). A soft-failed setup is still built and returned, flagged
_shadow_only with the _shadow_reason naming the gate it missed. run_scan
sends those to Claude as a SEPARATE capped batch (after the real LIGHT call
has already claimed its budget) and logs them with source='shadow' — they
never become a real signal, never enter dedup/autotrade, and every
pre-existing analytic keeps filtering source='live' so the numbers the admin
panel reports are unchanged.

Current soft-failable gates and the arm each one feeds:
    "score"          -> D  (mtf_score in [SHADOW_MIN_SCORE, MTF_MIN_SCORE))
    "ctxmom"         -> F  (the five narrow "context momentum pack" gates)

Slot C is FREE (2026-08-03). It last held the OVERLAP-session unblock, removed
the same day at the user's request. Note what that leaves standing: the
11:00-12:59 UTC block stays in place, justified by 19 trades measured against
a base later found broken, and no data will ever accumulate to confirm or
refute it. Re-add a _soft_fail("overlap") in signal_filter.py to revive it.

**Retired 2026-08-03 after the first read-out (54 setups, 27.07-02.08).** The
arms were judged by the SYMMETRIC DIFFERENCE against control A, not by their
own totals — the arms overlap almost entirely, so raw totals (and bootstrap
percentages built on them) are dominated by shared setups and look far more
decisive than they are:
  - E (eff_ratio>=0.25) removed 22 setups averaging +0.93R while A itself
    averaged +0.85R — it cut BETTER-than-average trades. Retired. Replaced by
    volume>=2.5x, the best-supported strict rule in the seed.
  - C (double-neutral OFF) differed from A by exactly ONE setup all week, and
    I (RSI-midline OFF) by four. Both were measuring nothing; their gates now
    fail hard again and the slots were re-pointed.
Read a STRICT arm by the avg R of what it REMOVES: if that exceeds the control's
own average, the arm destroys value however good its own ratio looks.

A setup that soft-fails TWO different gates is dropped outright: no single
variant would have admitted it, so it belongs to no arm. And every arm that
does NOT relax a given gate must exclude its shadow setups — that's _live_ok();
without it, shadow setups leak into the control arm A and the whole comparison
is measured against a polluted baseline.
"""

# code -> (label, predicate, measurable-under-current-live-config)
# predicate(setup: dict) -> bool : would THIS variant's filters admit the setup?


def _f(setup, key, default=0.0):
    v = setup.get(key)
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _bos_of(setup):
    return "bullish" if setup.get("direction") == "LONG" else "bearish"


def _live_ok(s):
    """True only for setups the LIVE filter actually admitted.

    Shadow setups (soft-failed a live gate, see signal_filter.py _shadow_only)
    would NOT exist under any config at least as strict as live, so every
    variant except the one deliberately relaxing that gate must exclude them —
    otherwise the control arm gets polluted with setups live rejected.
    """
    return not s.get("_shadow_only")


def _relaxes(s, reason):
    """True if s is live-admitted, OR is a shadow setup of exactly this kind."""
    return _live_ok(s) or s.get("_shadow_reason") == reason


def _v_a(s):   # control: everything the live filter already passed
    return _live_ok(s)


def _v_b(s):   # HTF_ALIGNED_LONG_GUARD=1 — cut LONGs where 1h AND 4h already bullish
    if not _live_ok(s):
        return False
    if s.get("direction") != "LONG":
        return True
    return not (s.get("trend_1h") == "bullish" and s.get("trend_4h") == "bullish")


def _v_d(s):   # looser score gate — fed by the score-shadow batch (see module docstring)
    return _relaxes(s, "score") and _f(s, "mtf_score") >= 12


def _v_e(s):   # volume_ratio >= 2.5 (re-pointed 2026-08-03; was eff_ratio >= 0.25,
               # RETIRED — over a week it removed 22 setups averaging +0.93R while
               # the control itself averaged +0.85R, i.e. it cut BETTER-than-average
               # trades, and the seed agrees: eff_ratio's top bucket (>=0.45) is its
               # WORST at +0.398R vs +0.475R for the middle).
               # The replacement is the best-supported strict rule we have: on the
               # 10,300-trade seed volume>=2.5x is 39% of setups at WR 82.1% /
               # +0.493R against 80.5% / +0.441R below it — and volume_ratio was
               # separately confirmed to predict on two 6-month windows (3605
               # trades) while mtf_score did not.
    return _live_ok(s) and _f(s, "volume_ratio") >= 2.5


def _v_f(s):   # "context momentum pack" OFF — the 5 narrow segment gates
               # (rel-weakness / narrow-zone / NY-momentum / SHORT-FVG-momentum /
               # FVG-London-BTC). All validated TOGETHER on one window 2026-06-05,
               # never walk-forward tested; BULL_NEUTRAL_LONG_MAX_ZONE_WIDTH_PCT is
               # tuned to 8 significant digits (0.00173509) — overfit smell.
    return _relaxes(s, "ctxmom")


def _v_g(s):   # skip overheated volatility regime, any direction (live only guards
               # bear+SHORT+hot-vol via BEAR_TREND_HOT_VOL_GUARD — this is the broader form)
    return _live_ok(s) and _f(s, "vol_ratio_regime", 1.0) < 2.0


def _v_h(s):   # "fresh trend": 4h leads, 1h hasn't caught up yet
    if not _live_ok(s):
        return False
    bos = _bos_of(s)
    aligned = int(s.get("trend_1h") == bos) + int(s.get("trend_4h") == bos)
    neutral = int(s.get("trend_1h") == "neutral") + int(s.get("trend_4h") == "neutral")
    return aligned == 1 and neutral == 1


def _v_i(s):   # vol_ratio_regime >= 1.3 (re-pointed 2026-08-03; was RSI-midline OFF,
               # which bound on only 4 setups a week — net -0.6R, unconcludable).
               # Deliberately the OPPOSITE END of arm G: the seed says the signal
               # lives at the BOTTOM of the volatility regime, not the top —
               # <0.8 → WR 76.1%/+0.387R, 0.8-1.3 → 79.5%/+0.425R, but 1.3-2.0 →
               # 81.9%/+0.475R and >2.0 → 82.6%/+0.496R. G cuts the BEST bucket;
               # this cuts the worst. Running both falsifies one of them.
    return _live_ok(s) and _f(s, "vol_ratio_regime", 1.0) >= 1.3


VARIANTS = {
    "A": ("Текущий (контроль)",              _v_a, True),
    "B": ("HTF-гейт LONG вкл",               _v_b, True),
    "D": ("Мягкий score ≥12 (shadow)",       _v_d, True),
    "E": ("Объём ≥2.5x",                     _v_e, True),
    "F": ("Контекст-моментум ВЫКЛ",          _v_f, True),
    "G": ("Vol-regime <2.0x",                _v_g, True),
    "H": ("Свежий тренд (mixed)",            _v_h, True),
    "I": ("Vol-regime ≥1.3x",                _v_i, True),
}


def compute_variants(setup: dict) -> str:
    """Comma-separated codes of variants that would admit this setup."""
    out = []
    for code, (_label, pred, _measurable) in VARIANTS.items():
        try:
            if pred(setup):
                out.append(code)
        except Exception:
            continue
    return ",".join(out)
