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

INTERPRETATION LIMIT (mostly): variants can only be a SUBSET of what the live
filter already admits (Claude never saw setups the live filter rejected, so
they're never logged with a verdict). Only variants STRICTER than live are
directly comparable to real-signal outcomes.

Exception — variant D (2026-07-25): near-miss setups (score in
[SHADOW_MIN_SCORE, MTF_MIN_SCORE)) are built + sent to Claude by run_scan as a
SEPARATE shadow-only batch (see main.py), tagged and logged the same way, but
never become a real signal. This makes D a genuine extra population, not a
mirror of A. See config.py SHADOW_MIN_SCORE and signal_filter.py's
_shadow_only flag.
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


def _v_a(s):   # control: everything the live filter already passed
    return True


def _v_b(s):   # HTF_ALIGNED_LONG_GUARD=1 — cut LONGs where 1h AND 4h already bullish
    if s.get("direction") != "LONG":
        return True
    return not (s.get("trend_1h") == "bullish" and s.get("trend_4h") == "bullish")


def _v_c(s):   # stricter score gate
    return _f(s, "mtf_score") >= 16


def _v_d(s):   # looser score gate — measurable via the shadow batch (see module docstring)
    return _f(s, "mtf_score") >= 12


def _v_e(s):   # stricter trend-quality floor (Kaufman eff_ratio) — 2026-07 WF sweep,
               # didn't survive OOS in backtest; live already gates at 0.15, this tests 0.25
    return _f(s, "eff_ratio", 1.0) >= 0.25


def _v_f(s):   # RSI ceiling for LONGs
    if s.get("direction") != "LONG":
        return True
    return _f(s, "rsi", 50.0) <= 65.0


def _v_g(s):   # skip overheated volatility regime, any direction (live only guards
               # bear+SHORT+hot-vol via BEAR_TREND_HOT_VOL_GUARD — this is the broader form)
    return _f(s, "vol_ratio_regime", 1.0) < 2.0


def _v_h(s):   # "fresh trend": 4h leads, 1h hasn't caught up yet
    bos = _bos_of(s)
    aligned = int(s.get("trend_1h") == bos) + int(s.get("trend_4h") == bos)
    neutral = int(s.get("trend_1h") == "neutral") + int(s.get("trend_4h") == "neutral")
    return aligned == 1 and neutral == 1


def _v_i(s):   # stricter BOS staleness (pre-2026-07-18 setting)
    ago = s.get("bos_candles_ago")
    if ago is None:
        return True
    return _f(s, "bos_candles_ago", 0.0) <= 3


VARIANTS = {
    "A": ("Текущий (контроль)",              _v_a, True),
    "B": ("HTF-гейт LONG вкл",               _v_b, True),
    "C": ("Строгий score ≥16",               _v_c, True),
    "D": ("Мягкий score ≥12 (shadow)",       _v_d, True),
    "E": ("Eff.ratio ≥0.25",                 _v_e, True),
    "F": ("RSI≤65 для LONG",                 _v_f, True),
    "G": ("Vol-regime <2.0x",                _v_g, True),
    "H": ("Свежий тренд (mixed)",            _v_h, True),
    "I": ("Строгий BOS ≤3 свечей",           _v_i, True),
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
