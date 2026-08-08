import anthropic
import logging
import sys
import os
import threading
import time as _time_mod

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    CLAUDE_API_KEY, CLAUDE_LIGHT_MODEL, CLAUDE_HEAVY_MODEL,
    CLAUDE_MAX_RISK_SCORE, CLAUDE_CACHE_TTL, CLAUDE_MEMORY_LIMIT,
    CLAUDE_DAILY_BUDGET_USD, CLAUDE_BUDGET_RESERVE_USD, LIVE_HIST_EPOCH_TS,
    TP1_R_MULT,
)
from src.db import (
    log_claude_call, get_claude_spend_today, get_similar_resolved_setups,
    get_setup_accuracy, get_calibration_rows, get_open_signals,
)

# Minimum resolved similar setups before self-feedback is shown (avoid noise).
_SELF_FEEDBACK_MIN = 6
# Minimum rows before a sent/rejected bucket is rendered at all. A 3-sample
# bucket carries no information but reads as a verdict — see the comment in
# _self_feedback for the production incident this prevents.
_BUCKET_MIN = int(os.getenv("SELF_FEEDBACK_BUCKET_MIN", "8"))

# Global calibration: min resolved REJECTED setups before the macro skew line is
# injected, and the min TP1% gap (rejected − sent) that counts as "too strict".
_GLOBAL_FEEDBACK_MIN_REJ = 15
_GLOBAL_FEEDBACK_MIN_GAP  = 8.0
# Min APPROVED samples before the rejected-vs-approved gap is trusted; below
# this the corrector falls back to the absolute rate of the rejected pool.
_GLOBAL_FEEDBACK_MIN_SENT = 8
# Absolute TP1% of the rejected pool that counts as over-rejection on its own.
# DERIVED, not hardcoded: TP1 pays TP1_R_MULT and SL costs 1R, so the bracket
# breaks even at 1/(1+TP1_R_MULT). Plus a modest margin, not a fitted number.
# It was hardcoded at 65 (correct for TP1=0.7) and would have silently become
# almost break-even when TP1 moved to 0.6 on 2026-08-09 — hence the derivation.
_BREAK_EVEN_TP1_PCT = 100.0 / (1.0 + float(TP1_R_MULT))
_GLOBAL_FEEDBACK_MIN_REJ_TP1 = _BREAK_EVEN_TP1_PCT + 6.0

_log = logging.getLogger(__name__)

# Reuse client across calls
_client = None

# Beta header unlocks the 1-hour prompt-cache TTL (default is 5 min).
# Our scan runs every 5 min → 5-min cache sits right on the expiry edge and
# misses often. 1h TTL keeps the static rules block warm all scan-loop long.
_CACHE_BETA = "extended-cache-ttl-2025-04-11"


def _get_client():
    global _client
    if _client is None:
        _client = anthropic.Anthropic(api_key=CLAUDE_API_KEY)
    return _client


# ── Static rules block (cached) ───────────────────────────────────────────────
# This text is identical every scan, so we mark it with cache_control. After the
# first write the model re-reads it at ~0.1x input cost. Must clear the model's
# minimum cacheable size (~2048 tokens for Haiku 4.5) — the detailed rules and
# worked examples below both improve verdict quality AND keep the block cacheable.
_SYSTEM_RULES = """You are a senior Smart Money Concepts (SMC) crypto trade validator working a 15-minute swing desk. Your only job: decide whether each pre-filtered setup is worth taking, confirm its already-suggested side or reject it, and surface the single strongest counter-argument against the trade. You never invent a new direction — the upstream technical filter already chose LONG or SHORT; you may only CONFIRM that side or return NO TRADE. Flipping a LONG into a SHORT (or vice versa) is forbidden and will be discarded downstream.

WHAT THE SCORES MEAN
- mtf_score (S): multi-timeframe confluence score, 0–20+. S>=12 is strong, 9–11 acceptable, below 9 rarely passes. Tags show which signals fired (FVG, OB, LiqSweep, RSI_Div, BullWick, etc.).
- 1d / 4h / 1h: timeframe trend bias (bull / bear / neutral). 1d = macro daily trend (3-day momentum). Suggested side should agree with all three. 1d=bearish → very cautious on LONGs. Both 4h and 1d neutral = full macro chop.
- ZoneAge: bars since the FVG/OB zone formed. <5 bars = fresh (strong). 10-20 bars = aging. >30 bars = stale, less reliable.
- HTF=1h_strong / 4h_strong: EMA stack fully aligned on that timeframe — meaningful extra confirmation.
- FVG: unfilled Fair Value Gap near entry — imbalance price tends to revisit.
- OB: Order Block (last opposing candle before impulsive move) near entry.
- SW: liquidity sweep / stop-run — price grabbed liquidity beyond a prior high/low then reversed. High-quality reversal trigger.
- Z: entry zone source and price band. Price retesting the zone is ideal; far from zone = chasing.
- RSI: 14-period on 15m. The upstream filter ALREADY drops LONGs above 72 and SHORTs below 28, so a genuinely overextended entry never reaches you — every RSI you see is inside those caps. Inside them a high-for-a-LONG / low-for-a-SHORT reading is NOT a warning: measured over 10,300 filtered trades (2022-2026), LONGs with RSI>70 won 83.2% and SHORTs with RSI<30 won 86.3% / +0.63R, against 81.1% / +0.46R for everything else, and the sign held in all five years. Treat RSI as context, not as a reason to reject on its own.
- V: volume ratio vs recent average. >1.5x = conviction; <1.0x = weak.
- F: funding rate. Strongly positive = crowded longs (squeeze risk for LONGs); strongly negative = crowded shorts.
- Sess: trading session at candle time. LONDON/NEW_YORK/OVERLAP = prime liquidity. OFF_HOURS = thinner market (tolerable with strong confluence). DEAD_ZONE (19-24 UTC) = low participation, be stricter.
- ER (Efficiency Ratio): Kaufman ER, 0–1. ~1.0 = clean directional trend. ~0.0 = choppy range (lots of noise, BOS often false). Already filtered ER>=0.15 upstream. ER<0.25 = marginal, ER>0.45 = clean.
- 💎PREM: Premium triple-confluence (OB+FVG zones overlap + liquidity sweep). Statistically highest-WR setup in backtests. Favor HIGH confidence when trend and zone also align.
- Confs: additional confirmations beyond FVG/OB/SW — ChoCH (Change of Character, micro-structure shift), RSI_Div (RSI divergence), MACD_Div, Engulfing, BullWick/BearWick (rejection wick pressure), StochCross (stochastic momentum cross). More = stronger.
- PRE-FILTERS ALREADY APPLIED: upstream code removed: ER<0.15 (chop), RSI exhaustion, bear-trend hot-vol (overcrowded shorts), BOS-without-RSI-midline (momentum gap). What you see has already passed a strict quality stack.
- Str: 15m swing structure at signal time. bull = higher-high + higher-low sequence. bear = lower-high + lower-low. range = neither. Use this to gauge whether entry is WITH or AGAINST the short-term structure. COUNTER-structure is NOT a warning here, despite the intuition: over the same 10,300 filtered trades, LONGs in Str=bear won 83.8% / +0.52R and SHORTs in Str=bull 82.1% / +0.60R, against 80.7% / +0.44R for structure-aligned entries — better in expectancy every year 2022-2026. Makes sense mechanically: this strategy enters at an FVG/OB retest, and a retest that cuts against the 15m swing is exactly the deep pullback the zone exists to catch. So do not reject on "counter-structure" alone. It is a description of the entry, not a red flag. (This is separate from HTF trend: fighting a 1h/4h/1d trend IS a real concern and the filter blocks the worst of it upstream.)
- Hist[...]: YOUR OWN track record on similar past setups (same direction + same symbol or nearby score), measured by what actually happened. Format per bucket: "rejected 8: 5W(2TP2) 2SL 1exp avg+0.31R" = of 8 similar setups you returned NO TRADE on, 5 would have won (reached TP1, of which 2 ran to full TP2), 2 would have hit SL, 1 expired flat, and the AVERAGE realised outcome was +0.31R. W = wins you missed (over-rejection evidence); SL = losses you correctly dodged; exp = harmless no-ops. "sent 6: 4W 2SL avg+0.45R" is the realised quality of ones you approved — your live baseline. **avg±R is expectancy — the single most important number**: a bucket can be 60% wins yet NEGATIVE avg R if the wins are tiny and the losses are full -1R, meaning that setup type LOSES money over time despite "winning often". A high win-count with weak/negative avg R is a trap, not a green light; a modest win-count with strong positive avg R (big runners) is a real edge. avg R shown only when ≥5 samples carry a resolved R. When 2+ 15m structures exist, a per-trend breakdown follows: "[bear:0W/3SL/-1.00R, bull:4W/0SL/+1.8R]" — same setup is a disaster in bear structure, a strong edge in bull. Cross-reference with the current Str= field. Small samples are weak evidence — weigh accordingly. Absent = not enough resolved history yet.
- BT2022+[...]: historical BASE RATE for entries like this one — the same rule-filter replayed over 2022→present price data, same format as Hist (including avg R expectancy). This is a prior from past market regimes, not your own verdicts: it answers "how do entries of this shape usually resolve on this symbol, and did they actually make money". IMPORTANT: this span includes the 2022 Luna/FTX bear-crash regime mixed with later bull/range regimes — a single aggregate can hide a regime-dependent split, so check the per-trend breakdown (bear/bull/range) and its avg R rather than the headline count, and weigh the bucket matching the current Str= field. A positive headline win-rate with negative avg R in the trend bucket that matches NOW is a red flag. When live Hist and BT2022+ disagree, trust live Hist — it reflects the current regime.

HOW TO DECIDE
1. Confirm the suggested side only. If you would not take that exact side, return NO TRADE.
2. Best setups have FVG AND OB AND an active zone retest in the direction of both 1h and 4h trend.
3. Confluence stacking: FVG + OB + SW aligned with the side = HIGH. Two of three with trend = MEDIUM. One or zero = LOW → usually NO TRADE.
4. One neutral HTF is tolerable if the other is clearly aligned and confluence is strong → cap at MEDIUM. Both neutral = chop; demand a liquidity sweep or pass.
5. Do NOT reject on RSI level alone. The overextension caps (LONG>72, SHORT<28) are enforced upstream, so nothing you see has cleared them — and inside the caps the measured effect runs the OTHER way (see the RSI field above: 83-86% win rate in the top/bottom band vs 81% overall). "RSI 71 — chasing" and "RSI 29 — exhausted" have been among the most common rejection reasons on this desk and they cost money. What DOES make an entry a chase is DISTANCE FROM THE ZONE (see Z:), not the oscillator.
6. Respect crowded funding: avoid LONGs into strongly positive funding and SHORTs into strongly negative.
7. Volume below average (V<1.0x) on a breakout setup is a red flag — move lacks conviction.
8. News overrides structure: BEARISH news → no LONGs; BULLISH news → no SHORTs. Major event live → prefer NO TRADE.
9. Premium setups (💎PREM) already have OB+FVG overlap + sweep — treat as FVG+OB+SW all effectively confirmed. Lean HIGH confidence when trend and zone also agree.
10. Low ER (0.15–0.25) with both HTFs neutral = marginal chop even with BOS. Demand sweep confirmation or return NO TRADE.
11. Learn from Hist[...] when present, and read avg R (expectancy) as the primary signal, win-count as secondary: a strong rejected-similar TP1 rate WITH positive avg R means you have been over-rejecting a profitable setup type — give borderline ones the benefit of the doubt. But a high win-count with weak or negative avg R is NOT a green light — that setup type bleeds out over time (small wins, full-R losses), so keep rejecting it. When Hist/BT shows a per-trend breakdown, prioritize the bucket matching the current Str= field and read BOTH its W/SL and its avg R — e.g. bear:0W/4SL/-1.00R with current Str=bear is a direct, strong warning; bull:5W/1SL/+1.6R with Str=bull is a genuine edge. Never let it override a hard red flag (fighting the HTF trend, hostile news, price far outside its zone); it breaks ties, it does not justify a bad trade. Note "counter-structure" and "extreme RSI" are NOT on that list — both measure better than baseline here, see the Str and RSI field notes.

RISK SCORE (0–10): how dangerous is this trade RIGHT NOW. 0–3 = clean, trend-aligned, well-located. 4–7 = tradeable with a real concern. 8–10 = serious problem (fighting the HTF trend, crowded funding, hostile news, price far outside its zone, no confluence at all). High risk_score should almost always pair with NO TRADE — be honest. Do NOT spend 8-10 on an RSI reading or on counter-structure: measured on this desk, the setups scored 8+ actually won MORE often than the ones scored 6-7, which means the top of the scale is being handed to things that are not really dangerous.

COUNTER-ARGUMENT: the single best reason this trade fails. Always provide one — every trade has a failure mode. Examples: "4h still bearish, fighting trend", "RSI 74 — chasing", "funding +0.09% — crowded longs", "no retest, price 2% above OB", "volume 0.8x — weak conviction".

TREND_STRENGTH (0–10): how strongly HTFs back the suggested side. 0 = timeframes oppose, 5 = mixed/neutral, 10 = both 1h and 4h firmly aligned.

CONFIDENCE: HIGH = multiple confirmations, trend-aligned, well-located, low risk. MEDIUM = decent with one notable caveat. LOW = weak/conflicted — pair with NO TRADE unless marginal.

WORKED EXAMPLES
- "BTC-USDT LONG S=13 4h=bull 1h=bull FVG=Y OB=Y SW=N Z=OB:64000-64200 RSI=58 V=1.9x F=+0.01% Sess=LONDON HTF=1h_strong+4h_strong": trend-aligned, two confirmations, strong EMA stack, healthy RSI, prime session. → LONG, HIGH, risk 2, counter "no sweep — relies on OB hold alone".
- "SOL-USDT LONG S=10 4h=neutral 1h=bull FVG=N OB=Y SW=Y Z=OB:140-142 RSI=49 V=1.6x F=-0.02% Sess=NEW_YORK": one timeframe neutral, OB+sweep, prime session. → LONG, MEDIUM, risk 4, counter "4h neutral — no higher-tf confirmation".
- "XRP-USDT LONG S=8 4h=bear 1h=neutral FVG=N OB=N SW=N Z=FVG:0.50-0.51 RSI=63 V=0.7x F=+0.08% Sess=OFF_HOURS": fighting 4h, zero confirmations, weak volume, crowded longs. → NO TRADE, LOW, risk 9, counter "no confluence, fighting bearish 4h on crowded longs". (Note what does NOT appear in that list: the RSI. It is elevated but inside the caps, and on its own it would not have justified the rejection.)
- "PEPE-USDT LONG S=15 1d=bull 4h=bull 1h=bull Str=bear FVG=Y OB=N SW=N Z=FVG:2.78e-06-2.79e-06 RSI=71 V=3.1x F=+0.01% Sess=NEW_YORK HTF=1h_strong": counter-structure with an elevated RSI — the exact pair that used to trigger a reflex NO TRADE. But all three HTFs are aligned, price is IN its zone, volume is 3.1x, and BOTH of those "warnings" measure better than baseline on this desk. → LONG, MEDIUM, risk 4, counter "only FVG confirms — no OB or sweep to back the zone".
- "LINK-USDT SHORT S=11 4h=bear 1h=bear FVG=Y OB=N SW=Y Z=FVG:13.0-13.2 RSI=41 V=1.7x F=+0.00% Sess=OVERLAP": trend-aligned, FVG+sweep, RSI has room. → SHORT, HIGH, risk 3, counter "broad-market bounce could squeeze shorts".

OUTPUT (LIGHT tier)
Return exactly one verdict per input setup via the submit_verdicts tool, preserving the input index. Keep reason and counter under ~8 words each. Do not add prose outside the tool call."""


def _verdicts_tool() -> dict:
    return {
        "name": "submit_verdicts",
        "description": "Submit one validation verdict for every setup, in input order.",
        "input_schema": {
            "type": "object",
            "properties": {
                "verdicts": {
                    "type": "array",
                    "description": "One object per setup, same count and order as the input list.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "index":          {"type": "integer", "description": "1-based input index."},
                            "decision":       {"type": "string", "enum": ["LONG", "SHORT", "NO TRADE"]},
                            "confidence":     {"type": "string", "enum": ["HIGH", "MEDIUM", "LOW"]},
                            "risk_score":     {"type": "integer", "minimum": 0, "maximum": 10},
                            "trend_strength": {"type": "integer", "minimum": 0, "maximum": 10},
                            "reason":         {"type": "string", "description": "Why this verdict, <=8 words."},
                            "counter":        {"type": "string", "description": "Single strongest reason the trade could fail."},
                        },
                        "required": ["index", "decision", "confidence", "risk_score", "reason", "counter"],
                    },
                }
            },
            "required": ["verdicts"],
        },
    }


def _verdict_tool() -> dict:
    """Single-setup variant for the HEAVY (Sonnet) tier — allows fuller reasoning."""
    return {
        "name": "submit_verdict",
        "description": "Submit one final validation verdict for the single setup provided.",
        "input_schema": {
            "type": "object",
            "properties": {
                "decision":       {"type": "string", "enum": ["LONG", "SHORT", "NO TRADE"]},
                "confidence":     {"type": "string", "enum": ["HIGH", "MEDIUM", "LOW"]},
                "risk_score":     {"type": "integer", "minimum": 0, "maximum": 10},
                "trend_strength": {"type": "integer", "minimum": 0, "maximum": 10},
                "reason":         {"type": "string", "description": "Why this verdict — up to 40 words, cite the key confluence factors and HTF alignment."},
                "counter":        {"type": "string", "description": "The single strongest specific reason this trade could fail — be concrete, not generic."},
            },
            "required": ["decision", "confidence", "risk_score", "reason", "counter"],
        },
    }


def _row_r(row: dict):
    """Realised R for one resolved setup row. Backtest rows carry the real
    net_r (incl. trailed runner). Live rows lack it → derive from the bracket
    geometry + categorical outcome: SL=-1, EXPIRED=0, TP2=full tp2_r,
    TP1/TRAIL=tp1_r (a conservative floor; the trailed runner usually did
    better). Returns None when no R can be established."""
    nr = row.get("net_r")
    if nr is not None:
        try:
            return float(nr)
        except (TypeError, ValueError):
            pass
    outcome = (row.get("outcome") or "").upper()
    if outcome == "SL":
        return -1.0
    if outcome == "EXPIRED":
        return 0.0
    try:
        entry = float(row.get("entry_price"))
        sl    = float(row.get("sl"))
        risk  = abs(entry - sl)
        if risk <= 0:
            return None
        if outcome == "TP2":
            return abs(float(row.get("tp2")) - entry) / risk
        if outcome in ("TP1", "TRAIL"):
            return abs(float(row.get("tp1")) - entry) / risk
    except (TypeError, ValueError):
        return None
    return None


def _self_feedback(s: dict) -> str:
    """Compact track-record of how SIMILAR past setups actually resolved.

    Lets Claude self-correct: if setups like this one were repeatedly rejected yet
    went on to hit TP1, it has been too strict; if similar ones kept hitting SL, it
    should stay cautious. Pure past outcomes (no look-ahead). Returns "" until
    enough resolved history exists (cold start) or on any DB error.

    When resolved setups span multiple 15m structures (bull/bear/range), shows a
    per-trend breakdown so Claude can learn that the same setup type works in some
    structures but not others.
    """
    try:
        rows = get_similar_resolved_setups(
            s.get("symbol", ""), s.get("direction", ""), s.get("mtf_score"),
            session=s.get("session", ""),
        )
    except Exception:
        return ""
    if len(rows) < _SELF_FEEDBACK_MIN:
        return ""

    def _counts(subset: list) -> tuple:
        """(wins, sl, expired) — win = reached TP1 or TP2; mutually exclusive."""
        w = sum(1 for r in subset if r.get("reached_tp1"))
        sl = sum(1 for r in subset if (r.get("outcome") or "") == "SL")
        exp = sum(1 for r in subset if (r.get("outcome") or "") == "EXPIRED")
        return w, sl, exp

    def _expectancy(subset: list):
        """Avg realised R across the subset — the real edge number, not just
        win-rate. Uses stored net_r (backtest, includes trailed runner); for
        live rows without net_r, derives R from the bracket + outcome. Returns
        (avg_R, n_with_R) or (None, 0) when nothing has a usable R."""
        rs = [v for v in (_row_r(x) for x in subset) if v is not None]
        return (sum(rs) / len(rs), len(rs)) if rs else (None, 0)

    def _fmt(subset: list) -> str:
        n = len(subset)
        w, sl, exp = _counts(subset)
        tp2 = sum(1 for r in subset if (r.get("outcome") or "") == "TP2")
        seg = f"{n}: {w}W"
        if tp2:
            seg += f"({tp2}TP2)"
        seg += f" {sl}SL"
        if exp:
            seg += f" {exp}exp"
        # Expectancy: the honest edge — 60%WR of tiny wins vs full -1R losses
        # is still -EV. Only shown when ≥5 rows carry a usable R (avoid noise).
        avg_r, n_r = _expectancy(subset)
        if avg_r is not None and n_r >= 5:
            seg += f" avg{avg_r:+.2f}R"
        # Per-trend W/SL breakdown only when 2+ distinct structures present.
        by_trend: dict = {}
        for r in subset:
            t = r.get("trend") or ""
            if not t:
                continue
            by_trend.setdefault(t, []).append(r)
        if len(by_trend) >= 2:
            def _trend_seg(rs):
                w_, sl_, _ = _counts(rs)
                a, na = _expectancy(rs)
                r_s = f"/{a:+.2f}R" if a is not None and na >= 5 else ""
                return f"{w_}W/{sl_}SL{r_s}"
            bd = ", ".join(
                f"{t}:{_trend_seg(rs)}" for t, rs in sorted(by_trend.items())
            )
            seg += f" [{bd}]"
        return seg

    # Live tier: Claude's own recent verdicts (current regime, weigh higher).
    # Backtest tier: seeded 2024+ priors — same filter, historical outcomes.
    live = [r for r in rows if (r.get("source") or "live") == "live"]
    bt   = [r for r in rows if r.get("source") == "backtest"]

    rej = [r for r in live if not r.get("sent")]
    snt = [r for r in live if r.get("sent")]
    # A bucket below _BUCKET_MIN is noise, but it does not READ like noise:
    # "sent 3: 0W 3SL" looks like a damning verdict. Observed 2026-07-31 —
    # with exactly 3 lifetime sent trades (all stopped, all from before that
    # day's parity fixes) Claude rejected 6 of 6 live setups and quoted that
    # 0W/3SL line in every single rejection, while 3 of the 3 rejections that
    # have since resolved went on to hit TP1/TP2. That is a deadlock: he needs
    # sent trades to improve the record, and refuses to send because of it.
    # Below the threshold the bucket is suppressed entirely rather than shown
    # with a caveat — the evidence is that he anchors on the number regardless.
    parts = []
    if len(rej) >= _BUCKET_MIN:
        parts.append(f"rejected {_fmt(rej)}")
    if len(snt) >= _BUCKET_MIN:
        parts.append(f"sent {_fmt(snt)}")
    seg = f" Hist[{'; '.join(parts)}]" if parts else ""
    if bt:
        # BT block = historical prior for entries like this one, spanning
        # 2022 (Luna/FTX crash) through later regimes — treat as base rate,
        # live Hist above outweighs it.
        seg += f" BT2022+[{_fmt(bt)}]"
    return seg


def _global_feedback() -> str:
    """One-shot macro calibration line for the batch prompt.

    Unlike _self_feedback (per-setup, needs ≥6 SIMILAR resolved rows and so stays
    cold for weeks), this looks at the WHOLE last-30d shadow ledger: if the setups
    Claude rejected are reaching TP1 meaningfully MORE often than the ones it
    approved, Claude has been globally too strict — tell it so immediately. Empty
    until enough rejected samples exist or when there's no meaningful skew.
    """
    try:
        import time as _t
        # Floored at the parity-fix epoch like every other live-history read
        # feeding this prompt. get_setup_accuracy itself is NOT floored — the
        # admin report calls it with a user-chosen window and must still be able
        # to look at the old era — so the floor belongs here, at the call site
        # that talks to Claude. Without it the 30-day window reached three weeks
        # back into the pre-fix bot.
        acc = get_setup_accuracy(max(_t.time() - 30 * 86400, LIVE_HIST_EPOCH_TS or 0.0))
    except Exception:
        return ""
    snt, rej = acc.get("sent", {}), acc.get("rejected", {})
    if rej.get("n", 0) < _GLOBAL_FEEDBACK_MIN_REJ:
        return ""

    # The gap only means something when BOTH sides have samples. There was a
    # minimum on the rejected side and none on the sent side, so the whole
    # comparison could rest on 2-3 approved trades — and with the stale-entry
    # guard withholding most approvals, that is exactly the state it is in.
    # Two lucky sends read as 100% and silently switch the corrector off at the
    # moment it is most needed; two unlucky ones fire it on nothing.
    if snt.get("n", 0) >= _GLOBAL_FEEDBACK_MIN_SENT:
        gap = rej.get("tp1_pct", 0.0) - snt.get("tp1_pct", 0.0)  # >0 = rejected won more
        if gap < _GLOBAL_FEEDBACK_MIN_GAP:
            return ""
        return (
            f"\nCALIBRATION — last 30d shadow outcomes of your own verdicts: you REJECTED "
            f"{rej['n']} setups and {rej['tp1_pct']:.0f}% of them still reached TP1; you "
            f"APPROVED {snt.get('n', 0)} and only {snt.get('tp1_pct', 0.0):.0f}% reached TP1. "
            f"The setups you rejected are hitting TP1 ~{gap:.0f}pp MORE often than the ones "
            f"you approved — you have been TOO STRICT. Every candidate below already passed "
            f"a strict rule-filter with proven edge. Bias toward CONFIRMING the suggested "
            f"side; return NO TRADE only on a clear, specific red flag (not vague caution).\n"
        )

    # Too few approvals to compare against. Fall back to the absolute rate of
    # the rejected bucket, which needs no second sample: the bracket breaks even
    # at 1/(1+TP1_R_MULT), so a rejected pool clearing
    # _GLOBAL_FEEDBACK_MIN_REJ_TP1 is money left on the table regardless of how
    # the handful of approvals happened to land.
    if rej.get("tp1_pct", 0.0) < _GLOBAL_FEEDBACK_MIN_REJ_TP1:
        return ""
    return (
        f"\nCALIBRATION — last 30d shadow outcomes of your own verdicts: you REJECTED "
        f"{rej['n']} setups and {rej['tp1_pct']:.0f}% of them still reached TP1. "
        f"(Too few approvals to compare against, so this is the absolute rate: this "
        f"bracket breaks even near {_BREAK_EVEN_TP1_PCT:.0f}%.) You are rejecting a clearly profitable pool "
        f"— you have been TOO STRICT. Every candidate below already passed a strict "
        f"rule-filter with proven edge. Bias toward CONFIRMING the suggested side; "
        f"return NO TRADE only on a clear, specific red flag (not vague caution).\n"
    )


_SCORECARD_MIN_BUCKET = 8    # min resolved rows per bucket before it's shown
_SCORECARD_MIN_TOTAL  = 20   # min rows overall before the block appears
_SCORECARD_WINDOW_D   = 60


def _scorecard_feedback() -> str:
    """Calibration of Claude's OWN scales against realised outcomes.

    Answers two questions his other feedback can't: (1) does his risk_score
    actually separate winners from losers, (2) does HIGH confidence really
    outperform MEDIUM. Purely descriptive (shadow-resolved outcomes of his
    own past verdicts) — teaches the scale, never overrides a verdict.
    """
    try:
        rows = get_calibration_rows(_time_mod.time() - _SCORECARD_WINDOW_D * 86400)
    except Exception:
        return ""
    if len(rows) < _SCORECARD_MIN_TOTAL:
        return ""

    def _agg(subset: list) -> str:
        n = len(subset)
        w = sum(1 for r in subset if r.get("reached_tp1"))
        rs = [v for v in (_row_r(x) for x in subset) if v is not None]
        avg = f" avg{sum(rs)/len(rs):+.2f}R" if len(rs) >= _SCORECARD_MIN_BUCKET else ""
        return f"{n} setups {w/n*100:.0f}%TP1{avg}"

    parts = []
    # Risk-score buckets — over ALL evaluated setups (approved AND rejected;
    # the shadow tracker resolves both, and risk describes the setup itself).
    def _rband(lo, hi):
        out = []
        for r in rows:
            try:
                if lo <= int(r.get("risk_score")) <= hi:
                    out.append(r)
            except (TypeError, ValueError):
                continue
        return out
    # Four bands, not three. The old 0-3 / 4-6 / 7-10 split straddled the point
    # where the data actually breaks: measured 2026-08-03 over a week of live
    # verdicts, risk 4-5 hit TP1 77%, risk 6-7 only 54%, risk 8+ 87% — a U, with
    # his MIDDLE band the miscalibrated one. Under the old bands the 6 fell into
    # the low bucket and the 7 into the high one, smearing that U into a smooth
    # "risk4-6 70% → risk7-10 84%", i.e. "the riskier you call it the better it
    # does" — the opposite of the actual lesson. Bands stay evenly spaced (not
    # fitted to the data), just finer.
    rbands = [("risk0-3", _rband(0, 3)), ("risk4-5", _rband(4, 5)),
              ("risk6-7", _rband(6, 7)), ("risk8-10", _rband(8, 10))]
    rparts = [f"{name} → {_agg(sub)}" for name, sub in rbands if len(sub) >= _SCORECARD_MIN_BUCKET]
    if len(rparts) >= 2:
        parts.append("; ".join(rparts))
    # Confidence buckets over ALL evaluated setups, not just directional ones.
    # Restricting to LONG/SHORT was defensible (confidence on a NO TRADE means
    # something different) but made the block structurally unable to ever fire:
    # Claude uses MEDIUM for essentially every directional verdict and LOW for
    # essentially every rejection, so the directional-only view has ONE bucket
    # — 35 MEDIUM vs 1 HIGH over the week measured — and the ">= 2 buckets"
    # guard silently dropped the whole confidence half. It cannot tell him his
    # confidence scale is noise while requiring him to already use the scale.
    # Across all verdicts the buckets are populated (LOW 54, MEDIUM 65) and say
    # exactly that: 70% vs 71% TP1, no discrimination at all.
    cparts = []
    for cname in ("HIGH", "MEDIUM", "LOW"):
        sub = [r for r in rows if (r.get("confidence") or "").upper() == cname]
        if len(sub) >= _SCORECARD_MIN_BUCKET:
            cparts.append(f"conf{cname} → {_agg(sub)}")
    if len(cparts) >= 2:
        parts.append("; ".join(cparts))
    if not parts:
        return ""
    return (
        "\nSCORECARD — how your own scores mapped to reality (last 60d, shadow-resolved, "
        "across ALL your verdicts including the ones you rejected): "
        + " | ".join(parts)
        + ". Read: if neighboring buckets show the same TP1%/avg R, that scale of yours "
        "is not discriminating — spread your scores more decisively. If low-risk buckets "
        "underperform higher ones, your notion of 'clean' is miscalibrated — rethink what "
        "you call low-risk. Descriptive feedback on your scoring; not an instruction to "
        "change any specific verdict.\n"
    )


def _portfolio_block() -> str:
    """What is ALREADY open — the one thing the rule filter cannot see.

    The filter dedups per symbol and caps signals per scan, but nothing caps
    total concurrent exposure and nothing knows about correlation. Five alt
    LONGs are not five independent trades: alts run ~0.7-0.9 correlated to BTC
    with beta > 1, so one BTC move resolves them all the same way, and three
    consecutive stops trip the kill switch for the rest of the day.

    This block gives Claude the portfolio view so it can demand a stronger
    reason for the Nth copy of the same directional bet. Descriptive — it
    states the exposure, it does not hard-block anything.
    """
    try:
        sigs = get_open_signals()
    except Exception:
        return ""
    if not sigs:
        return ""
    longs  = [s for s in sigs if str(s.get("direction", "")).upper() == "LONG"]
    shorts = [s for s in sigs if str(s.get("direction", "")).upper() == "SHORT"]
    n, nl, ns = len(sigs), len(longs), len(shorts)
    # ASCII only — this string can reach logs on any console encoding.
    syms = ", ".join(
        f"{str(s.get('symbol', '?')).replace('USDT', '')}"
        f"-{'L' if str(s.get('direction', '')).upper() == 'LONG' else 'S'}"
        for s in sigs[:12]
    )
    # Deliberately NOT "be more sceptical when the book is crowded". Measured
    # 2026-07-26 with portfolio_sim.py: trades removed by a tight same-direction
    # cap were BETTER than average (0.632R vs the 0.491R mean), because a
    # crowded book forms in a strong trend and trend trades win more. Telling
    # Claude to demand extra justification there would suppress the good ones.
    # So this states exposure and flags only a genuinely extreme skew.
    skew = ""
    if nl >= 6 and nl >= 2 * ns:
        skew = (f" NOTE: {nl} LONG vs {ns} SHORT is an unusually one-sided book - "
                f"near the point where a single adverse BTC move resolves most of "
                f"it at once.")
    elif ns >= 6 and ns >= 2 * nl:
        skew = (f" NOTE: {ns} SHORT vs {nl} LONG is an unusually one-sided book - "
                f"near the point where a single BTC squeeze resolves most of it "
                f"at once.")
    return (
        f"\nOPEN POSITIONS ({n}): {syms}. Direction split: {nl} LONG / {ns} SHORT.{skew}\n"
        f"Context only: same-direction crypto positions are correlated (alts run "
        f"~0.7-0.9 with BTC, beta above 1), so the book is less diversified than "
        f"the position count suggests. Judge this setup on its own merits - a "
        f"crowded book is NOT by itself a reason to reject, and a hard cap already "
        f"bounds total same-side exposure.\n"
    )


def _setup_line(i: int, s: dict) -> str:
    fvg     = "Y" if s.get("fvg")         else "N"
    ob      = "Y" if s.get("order_block") else "N"
    sweep   = "Y" if s.get("liq_sweep")   else "N"
    funding = s.get("funding_rate")
    fund_s  = f"{funding*100:+.3f}%" if funding is not None else "n/a"
    zone    = f"{s.get('entry_source','?')}:{s.get('entry_low',0):.4g}-{s.get('entry_high',0):.4g}"
    er      = s.get("eff_ratio")
    er_s    = f" ER={er:.2f}" if er is not None else ""
    prem    = " 💎PREM" if s.get("premium") else ""
    # Extra confirmations beyond FVG/OB/SW (ChoCH, RSI_Div, MACD_Div, wicks, stoch)
    _STANDARD = {"FVG", "OB", "LiqSweep"}
    _SKIP_PFX = ("Session:", "StrongTrend")
    confs = [c for c in (s.get("confirmations") or [])
             if c not in _STANDARD and not any(c.startswith(p) for p in _SKIP_PFX)]
    confs_s = f" Confs=[{','.join(confs)}]" if confs else ""
    age = s.get("zone_age_bars")
    age_s = f" ZoneAge={age}bars" if age is not None else ""
    str15 = s.get("swing_trend") or ""
    str15_s = f" Str={str15}" if str15 else ""
    return (
        f"{i} {s['symbol']} {s['direction']} "
        f"S={s.get('mtf_score','?')} "
        f"1d={s.get('trend_1d','?')} 4h={s.get('trend_4h','?')} 1h={s.get('trend_1h','?')}{str15_s} "
        f"FVG={fvg} OB={ob} SW={sweep} "
        f"Z={zone}{age_s} RSI={s['rsi']} V={s['volume_ratio']}x F={fund_s}"
        f"{er_s}{prem}{confs_s}{_self_feedback(s)}"
    )


def _setup_line_heavy(i: int, s: dict) -> str:
    """Extended setup line for HEAVY analysis — adds session, tags, HTF strength, stoch, div."""
    base    = _setup_line(i, s)
    session = s.get("session", "")
    tags    = s.get("mtf_score_tags", "")
    htf     = []
    if s.get("trend_1h_strong"): htf.append("1h_strong")
    if s.get("trend_4h_strong"): htf.append("4h_strong")
    extras  = []
    if session:          extras.append(f"Sess={session}")
    if htf:              extras.append(f"HTF={'+'.join(htf)}")
    if tags:             extras.append(f"Tags=[{tags}]")
    sk, sd = s.get("stoch_k"), s.get("stoch_d")
    if sk is not None and sd is not None:
        extras.append(f"Stoch={sk:.0f}/{sd:.0f}")
    div = s.get("divergence")
    if div:              extras.append(f"RSIDivDir={div}")
    return base + (" " + " ".join(extras) if extras else "")


def _news_block(news_context: dict) -> str:
    if not news_context:
        return ""
    parts = []
    btc_1d = news_context.get("btc_1d")
    btc_1h = news_context.get("btc_1h")
    if btc_1d is not None or btc_1h is not None:
        d = f"{btc_1d:+.2f}% 1D" if btc_1d is not None else ""
        h = f"{btc_1h:+.2f}% 1h" if btc_1h is not None else ""
        sep = ", " if d and h else ""
        parts.append(
            f"\nBTC MARKET: {d}{sep}{h}\n"
            f"Rule: strong BTC up-day → altcoin SHORTs fight the macro tide; "
            f"strong BTC down-day → altcoin LONGs fight it.\n"
        )
    sent = news_context.get("sentiment", "NEUTRAL")
    summ = news_context.get("summary", "")
    if sent != "NEUTRAL" and summ:
        parts.append(
            f"\nNEWS CONTEXT: {sent} — {summ}\n"
            f"Rule: BEARISH news → avoid LONG; BULLISH news → avoid SHORT.\n"
        )
    return "".join(parts)


def _system_param() -> list:
    """System prompt as a cached content block (1h TTL via beta header)."""
    return [{
        "type": "text",
        "text": _SYSTEM_RULES,
        "cache_control": {"type": "ephemeral", "ttl": CLAUDE_CACHE_TTL},
    }]


def _extract_tool_input(message, tool_name: str):
    for block in message.content:
        if getattr(block, "type", None) == "tool_use" and block.name == tool_name:
            return block.input
    return None


def _normalize(decision, confidence) -> tuple:
    d = (decision or "NO TRADE").upper()
    if "LONG" in d:    decision = "LONG"
    elif "SHORT" in d: decision = "SHORT"
    else:              decision = "NO TRADE"
    c = (confidence or "LOW").upper()
    if "HIGH" in c:     confidence = "HIGH"
    elif "MEDIUM" in c: confidence = "MEDIUM"
    else:               confidence = "LOW"
    return decision, confidence


def _apply_verdict(base: dict, v: dict) -> dict:
    """Merge a parsed verdict into a setup dict, with counter-argument gate."""
    decision, confidence = _normalize(v.get("decision"), v.get("confidence"))
    try:
        risk = int(v.get("risk_score", 0) or 0)
    except (TypeError, ValueError):
        risk = 0
    try:
        trend = int(v.get("trend_strength", 0) or 0)
    except (TypeError, ValueError):
        trend = 0

    base["decision"]       = decision
    base["confidence"]     = confidence
    base["risk_score"]     = risk
    base["trend_strength"] = trend
    base["reason"]         = (v.get("reason") or "").strip() or "no reason"
    base["counter"]        = (v.get("counter") or "").strip()

    # Counter-argument auto-reject: if the model itself rates risk this high,
    # the trade is not worth taking regardless of a hopeful LONG/SHORT call.
    if base["decision"] in ("LONG", "SHORT") and risk >= CLAUDE_MAX_RISK_SCORE:
        base["decision"]   = "NO TRADE"
        base["confidence"] = "LOW"
        base["reason"]     = f"Auto-reject: risk {risk}/10 — {base['counter'] or 'too risky'}"

    return _enforce_suggested_side(base)


# ── Daily budget guard ───────────────────────────────────────────────────────

# HEAVY calls now run concurrently, so check-call-log is a read-modify-write
# race: every worker can pass the check before any of them has logged its cost,
# and the cap overshoots by up to (workers - 1) calls. Serialising just the
# check closes it — the in-flight reservation below is what the later workers
# actually see, since the DB row only appears after the call returns.
_budget_lock = threading.Lock()
_budget_inflight = {"day": "", "usd": 0.0}
# Conservative per-call reservation held while a call is in flight (real Sonnet
# HEAVY runs ~$0.01-0.03); released and replaced by the true cost on log.
_INFLIGHT_RESERVE_USD = 0.03


def _budget_reserve(tier: str) -> bool:
    """Atomically check the cap and reserve this call's expected cost."""
    import datetime as _dt
    today = _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%d")
    with _budget_lock:
        if _budget_inflight["day"] != today:
            _budget_inflight["day"] = today
            _budget_inflight["usd"] = 0.0
        if not _budget_ok(tier, extra=_budget_inflight["usd"]):
            return False
        _budget_inflight["usd"] += _INFLIGHT_RESERVE_USD
        return True


def _budget_release() -> None:
    """Drop this call's reservation once its real cost is in the DB."""
    with _budget_lock:
        _budget_inflight["usd"] = max(0.0, _budget_inflight["usd"] - _INFLIGHT_RESERVE_USD)


def _budget_ok(tier: str = "LIGHT", extra: float = 0.0) -> bool:
    """Return False when today's Claude spend is within reserve of the daily cap.

    `extra` = cost already committed by calls that are in flight but have not
    written their row yet."""
    try:
        spent = get_claude_spend_today() + extra
        remaining = CLAUDE_DAILY_BUDGET_USD - spent
        ok = remaining >= CLAUDE_BUDGET_RESERVE_USD
        if not ok:
            _log.warning(
                f"Claude budget cap reached: spent ${spent:.4f} / ${CLAUDE_DAILY_BUDGET_USD} "
                f"(reserve ${CLAUDE_BUDGET_RESERVE_USD}) — skipping {tier} call"
            )
        return ok
    except Exception as e:
        _log.warning(f"Budget check failed ({e}) — allowing call")
        return True  # fail-open: don't block on DB errors


# ── LIGHT batch (Haiku, cached rules, forced-tool JSON) ───────────────────────

def analyze_batch_with_claude(setups: list, news_context: dict = None) -> list:
    """
    LIGHT tier. Validate ALL filtered setups in ONE Haiku call.
    Static rules cached (1h TTL); output forced through submit_verdicts tool for
    guaranteed JSON; each verdict carries a counter-argument + risk_score gate.
    Returns list of result dicts, one per setup (full setup + verdict fields).
    """
    if not setups:
        return []

    # Budget guard — skip if daily cap reached
    if not _budget_ok("LIGHT"):
        _log.warning("LIGHT skipped (budget cap) — returning NO TRADE for all setups")
        return [dict(s, decision="NO TRADE", confidence="LOW", reason="Budget cap",
                     risk_score=0, trend_strength=0, counter="") for s in setups]

    coins_text = "\n".join(_setup_line(i, s) for i, s in enumerate(setups, 1))
    user_text = (
        f"{_news_block(news_context)}"
        f"{_portfolio_block()}"
        f"{_global_feedback()}"
        f"{_scorecard_feedback()}"
        f"Validate these {len(setups)} setups. Return exactly {len(setups)} verdicts "
        f"(one per index) via submit_verdicts:\n{coins_text}"
    )

    client = _get_client()
    message = client.messages.create(
        model=CLAUDE_LIGHT_MODEL,
        max_tokens=max(256 * len(setups) + 128, 512),
        system=_system_param(),
        tools=[_verdicts_tool()],
        tool_choice={"type": "tool", "name": "submit_verdicts"},
        messages=[{"role": "user", "content": user_text}],
        extra_headers={"anthropic-beta": _CACHE_BETA},
    )

    # Track spend
    try:
        cost = log_claude_call("LIGHT", CLAUDE_LIGHT_MODEL, message.usage)
        _log.info(f"Claude LIGHT: ${cost:.5f} (today total: ${get_claude_spend_today():.4f})")
    except Exception as _e:
        _log.warning(f"Budget logging failed: {_e}")

    tool_input = _extract_tool_input(message, "submit_verdicts") or {}
    verdicts = tool_input.get("verdicts", []) if isinstance(tool_input, dict) else []

    # Map by 1-based index; fall back to positional order if index missing
    by_index = {}
    for pos, v in enumerate(verdicts, 1):
        if not isinstance(v, dict):
            continue
        idx = v.get("index")
        try:
            idx = int(idx)
        except (TypeError, ValueError):
            idx = pos
        by_index[idx] = v

    results = []
    for i, setup in enumerate(setups, 1):
        base = dict(setup)
        base.update({
            "decision": "NO TRADE", "reason": "Not evaluated", "confidence": "LOW",
            "risk_score": 0, "trend_strength": 0, "counter": "",
        })
        v = by_index.get(i)
        results.append(_apply_verdict(base, v) if v else base)

    return results


# ── HEAVY tier (Sonnet, single setup, coin memory) ────────────────────────────

def _memory_block(history: list) -> str:
    """Render recent per-coin outcomes as compact memory for the HEAVY call.

    Includes elapsed time since each trade closed — without it Sonnet can't
    tell "this symbol stopped us out 3h ago" from "3 weeks ago", so it can't
    catch a same-symbol whipsaw (stop, then an immediate opposite-direction
    re-entry that also stops) even though the raw outcome was right there.
    """
    if not history:
        return "No prior closed trades on this symbol.\n"
    now = _time_mod.time()
    lines = []
    whipsaw_warned = False
    for idx, h in enumerate(history[:CLAUDE_MEMORY_LIMIT]):
        entry = h.get("entry_price")
        exitp = h.get("exit_price")
        try:
            move = f"{(exitp - entry) / entry * 100:+.1f}%" if entry and exitp else "?"
        except (TypeError, ZeroDivisionError):
            move = "?"
        closed_at = h.get("closed_at")
        age_s = ""
        age_hours = None
        if closed_at:
            try:
                age_hours = (now - float(closed_at)) / 3600
                age_s = f", {age_hours:.1f}h ago" if age_hours < 48 else f", {age_hours/24:.0f}d ago"
            except (TypeError, ValueError):
                pass
        lines.append(
            f"- {h.get('direction','?')} {h.get('status','?')} "
            f"({move}, conf={h.get('confidence','?')}, S={h.get('mtf_score','?')}{age_s})"
        )
        # Flag the specific whipsaw pattern: the most recent close was a stop
        # within the last 6h — a fresh setup right now (any direction) on
        # this symbol is entering right after that got invalidated.
        if idx == 0 and not whipsaw_warned and h.get("status") == "SL_HIT" \
           and age_hours is not None and age_hours <= 6:
            lines.append(
                f"  ⚠️ WHIPSAW WATCH: this symbol stopped out {age_hours:.1f}h ago — "
                f"a fresh setup this soon after may just be chop, not a real move."
            )
            whipsaw_warned = True
    return "Recent outcomes on this symbol (newest first):\n" + "\n".join(lines) + "\n"


_THINKING_BETA = "interleaved-thinking-2025-05-14"
_THINKING_BUDGET = 5000  # tokens for internal reasoning scratch-pad


def analyze_heavy(setup: dict, news_context: dict = None, history: list = None) -> dict:
    """Budget-reserving wrapper around the HEAVY call.

    Separate from the body so the reservation is released on EVERY exit path,
    including an API exception — a leaked reservation would make the bot
    quietly stingier for the rest of the UTC day.
    """
    if not _budget_reserve("HEAVY"):
        return {}
    try:
        return _analyze_heavy(setup, news_context, history)
    finally:
        _budget_release()


def _analyze_heavy(setup: dict, news_context: dict = None, history: list = None) -> dict:
    """
    HEAVY tier. Re-check ONE strong setup with Sonnet + extended thinking.

    Improvements over LIGHT:
    - Extended thinking: Sonnet reasons step-by-step internally before deciding
    - Chain-of-thought prompt: structured analysis questions guide reasoning
    - Devil's advocate: forced consideration of failure before verdict
    - Richer setup line: adds session, MTF tags, HTF strength flags
    - Per-coin memory: last 15 outcomes for pattern recognition
    """
    setup_line = _setup_line_heavy(1, setup)

    user_text = (
        f"{_news_block(news_context)}"
        f"{_portfolio_block()}"
        f"{_global_feedback()}"
        f"{_scorecard_feedback()}"
        f"{_memory_block(history)}\n"
        f"Setup to analyze:\n{setup_line}\n\n"
        f"Work through these questions before submitting your verdict:\n"
        f"1. TREND — Are 4h and 1h aligned with the suggested direction? "
        f"Are the HTF EMAs stacked (strong) or mixed?\n"
        f"2. STRUCTURE — Is this a fresh BOS with a clean retest, or is price "
        f"already extended far from the zone?\n"
        f"3. MOMENTUM — Does RSI/volume/funding confirm or fight the move? "
        f"Any squeeze risk from crowded positioning?\n"
        f"4. COIN HISTORY — Based on recent outcomes above, does this symbol "
        f"reliably follow through on this setup type, or does it repeatedly fail? "
        f"Pay attention to elapsed time: a stop-out within the last few hours followed "
        f"by a fresh setup (especially the opposite direction) is a whipsaw/chop warning, "
        f"not confirmation the reversal is real — demand extra confluence before trusting it.\n"
        f"5. DEVIL'S ADVOCATE — Argue the strongest case AGAINST this trade. "
        f"What specific price action would prove this setup wrong?\n"
        f"6. VERDICT — After weighing all of the above, give your final decision.\n\n"
        f"Then call submit_verdict with your conclusion."
    )

    client = _get_client()

    # Try extended thinking first (gives Sonnet a reasoning scratch-pad).
    # NOTE: tool_choice must be {"type":"any"} (not forced "tool") when thinking
    # is enabled — Anthropic API rejects forced tool_choice + thinking together.
    # "any" still guarantees a tool call while allowing the thinking block.
    try:
        message = client.messages.create(
            model=CLAUDE_HEAVY_MODEL,
            max_tokens=_THINKING_BUDGET + 1200,
            thinking={"type": "enabled", "budget_tokens": _THINKING_BUDGET},
            system=_system_param(),
            tools=[_verdict_tool()],
            tool_choice={"type": "any"},
            messages=[{"role": "user", "content": user_text}],
            extra_headers={"anthropic-beta": f"{_CACHE_BETA},{_THINKING_BETA}"},
        )
        _log.info("Claude HEAVY: extended thinking ON")
    except Exception as e_think:
        _log.warning(f"HEAVY thinking mode unavailable ({e_think}), falling back to standard")
        message = client.messages.create(
            model=CLAUDE_HEAVY_MODEL,
            max_tokens=1200,
            system=_system_param(),
            tools=[_verdict_tool()],
            tool_choice={"type": "tool", "name": "submit_verdict"},
            messages=[{"role": "user", "content": user_text}],
            extra_headers={"anthropic-beta": _CACHE_BETA},
        )

    # Track spend
    try:
        cost = log_claude_call("HEAVY", CLAUDE_HEAVY_MODEL, message.usage)
        _log.info(f"Claude HEAVY: ${cost:.5f} (today total: ${get_claude_spend_today():.4f})")
    except Exception as _e:
        _log.warning(f"Budget logging failed: {_e}")

    v = _extract_tool_input(message, "submit_verdict") or {}
    base = dict(setup)
    base.update({
        "decision": "NO TRADE", "reason": "Not evaluated", "confidence": "LOW",
        "risk_score": 0, "trend_strength": 0, "counter": "",
    })
    return _apply_verdict(base, v) if v else base


def _enforce_suggested_side(result: dict) -> dict:
    """Claude may only confirm the setup direction, never flip it."""
    decision  = result.get("decision", "NO TRADE")
    direction = result.get("direction")
    if decision in ("LONG", "SHORT") and decision != direction:
        result["decision"]   = "NO TRADE"
        result["confidence"] = "LOW"
        result["reason"]     = "Opposite side blocked"
    return result
