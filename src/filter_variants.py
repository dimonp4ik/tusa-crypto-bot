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

There are no STRICTER arms left (2026-08-09): every surviving arm either IS
the live filter (A) or loosens it. The experiment now asks one question only —
does relaxing a gate help — which matches where the evidence has pointed all
along: the strict arms measured so far cut better-than-average trades.

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
    "ctxmom"         -> F  (the five narrow "context momentum pack" gates)

Slot C is FREE (2026-08-03). It last held the OVERLAP-session unblock, removed
the same day at the user's request. Note what that leaves standing: the
11:00-12:59 UTC block stays in place, justified by 19 trades measured against
a base later found broken, and no data will ever accumulate to confirm or
refute it. Re-add a _soft_fail("overlap") in signal_filter.py to revive it.

Slot D is FREE (2026-08-07) — removed on request. It held the soft score gate
(mtf_score in [SHADOW_MIN_SCORE=12, MTF_MIN_SCORE=14)), the only arm fed by the
"score" shadow reason, so removing it also switches that shadow batch off and
stops those near-misses costing Claude budget. What is given up: the live test
of whether the 10->14 score raise (A/B'd on ONE 2026-06-11 window: scores 12-13
= WR ~20%, -6.3R over 20 symbols) still holds now. That raise stays in force
unmeasured. SHADOW_MIN_SCORE remains in config but is inert until the
_soft_fail("score") call in signal_filter.py is restored.

Slots B, E, G, H, I are FREE (2026-08-09) — removed on request in two steps.
LIVE ARMS ARE NOW A (control) and F. What was given up, recorded so it does
not resurface as a surprise:
  - G (vol-regime < 2.0x) cut the STRONGEST bucket on the 10,300-trade seed:
    <0.8 -> 76.1%/+0.387R, 0.8-1.3 -> 79.5%/+0.425R, 1.3-2.0 -> 81.9%/+0.475R,
    >2.0 -> 82.6%/+0.496R. Removing it costs nothing and probably prevents a
    mistake; I, its deliberate falsifier at the opposite end of the same axis,
    went in the previous step.
  - E held volume >= 2.5x, the best-supported strict rule available (39% of
    seed setups, WR 82.1%/+0.493R vs 80.5%/+0.441R below the threshold, and
    volume_ratio predicts on two independent 6-month windows where mtf_score
    does not). It was never given live data to fail on.

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



def _v_f(s):   # "context momentum pack" OFF — the 5 narrow segment gates
               # (rel-weakness / narrow-zone / NY-momentum / SHORT-FVG-momentum /
               # FVG-London-BTC). All validated TOGETHER on one window 2026-06-05,
               # never walk-forward tested; BULL_NEUTRAL_LONG_MAX_ZONE_WIDTH_PCT is
               # tuned to 8 significant digits (0.00173509) — overfit smell.
    return _relaxes(s, "ctxmom")


VARIANTS = {
    "A": ("Текущий (контроль)",              _v_a, True),
    "F": ("Контекст-моментум ВЫКЛ",          _v_f, True),
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
