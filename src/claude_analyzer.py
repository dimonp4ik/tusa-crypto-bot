import anthropic
import logging
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    CLAUDE_API_KEY, CLAUDE_LIGHT_MODEL, CLAUDE_HEAVY_MODEL,
    CLAUDE_MAX_RISK_SCORE, CLAUDE_CACHE_TTL, CLAUDE_MEMORY_LIMIT,
    CLAUDE_DAILY_BUDGET_USD, CLAUDE_BUDGET_RESERVE_USD,
)
from src.db import log_claude_call, get_claude_spend_today

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
- 4h / 1h: higher-timeframe trend bias (bull / bear / neutral). Suggested side should agree. Both neutral = chop, treat with suspicion.
- HTF=1h_strong / 4h_strong: EMA stack fully aligned on that timeframe — meaningful extra confirmation.
- FVG: unfilled Fair Value Gap near entry — imbalance price tends to revisit.
- OB: Order Block (last opposing candle before impulsive move) near entry.
- SW: liquidity sweep / stop-run — price grabbed liquidity beyond a prior high/low then reversed. High-quality reversal trigger.
- Z: entry zone source and price band. Price retesting the zone is ideal; far from zone = chasing.
- RSI: 14-period on 15m. >72 weakens LONGs; <28 weakens SHORTs.
- V: volume ratio vs recent average. >1.5x = conviction; <1.0x = weak.
- F: funding rate. Strongly positive = crowded longs (squeeze risk for LONGs); strongly negative = crowded shorts.
- Sess: trading session at candle time. LONDON/NEW_YORK/OVERLAP = prime liquidity. OFF_HOURS = thinner market (tolerable with strong confluence). DEAD_ZONE (19-24 UTC) = low participation, be stricter.

HOW TO DECIDE
1. Confirm the suggested side only. If you would not take that exact side, return NO TRADE.
2. Best setups have FVG AND OB AND an active zone retest in the direction of both 1h and 4h trend.
3. Confluence stacking: FVG + OB + SW aligned with the side = HIGH. Two of three with trend = MEDIUM. One or zero = LOW → usually NO TRADE.
4. One neutral HTF is tolerable if the other is clearly aligned and confluence is strong → cap at MEDIUM. Both neutral = chop; demand a liquidity sweep or pass.
5. Reject overextended entries: LONG with RSI>72 or SHORT with RSI<28 is chasing — downgrade hard or NO TRADE unless a fresh sweep justifies it.
6. Respect crowded funding: avoid LONGs into strongly positive funding and SHORTs into strongly negative.
7. Volume below average (V<1.0x) on a breakout setup is a red flag — move lacks conviction.
8. News overrides structure: BEARISH news → no LONGs; BULLISH news → no SHORTs. Major event live → prefer NO TRADE.

RISK SCORE (0–10): how dangerous is this trade RIGHT NOW. 0–3 = clean, trend-aligned, well-located. 4–7 = tradeable with a real concern. 8–10 = serious problem (chasing, fighting trend, crowded funding, hostile news, far from zone). High risk_score should almost always pair with NO TRADE — be honest.

COUNTER-ARGUMENT: the single best reason this trade fails. Always provide one — every trade has a failure mode. Examples: "4h still bearish, fighting trend", "RSI 74 — chasing", "funding +0.09% — crowded longs", "no retest, price 2% above OB", "volume 0.8x — weak conviction".

TREND_STRENGTH (0–10): how strongly HTFs back the suggested side. 0 = timeframes oppose, 5 = mixed/neutral, 10 = both 1h and 4h firmly aligned.

CONFIDENCE: HIGH = multiple confirmations, trend-aligned, well-located, low risk. MEDIUM = decent with one notable caveat. LOW = weak/conflicted — pair with NO TRADE unless marginal.

WORKED EXAMPLES
- "BTC-USDT LONG S=13 4h=bull 1h=bull FVG=Y OB=Y SW=N Z=OB:64000-64200 RSI=58 V=1.9x F=+0.01% Sess=LONDON HTF=1h_strong+4h_strong": trend-aligned, two confirmations, strong EMA stack, healthy RSI, prime session. → LONG, HIGH, risk 2, counter "no sweep — relies on OB hold alone".
- "SOL-USDT LONG S=10 4h=neutral 1h=bull FVG=N OB=Y SW=Y Z=OB:140-142 RSI=49 V=1.6x F=-0.02% Sess=NEW_YORK": one timeframe neutral, OB+sweep, prime session. → LONG, MEDIUM, risk 4, counter "4h neutral — no higher-tf confirmation".
- "XRP-USDT LONG S=8 4h=bear 1h=neutral FVG=N OB=N SW=N Z=FVG:0.50-0.51 RSI=74 V=0.7x F=+0.08% Sess=OFF_HOURS": fighting 4h, no confirmations, overbought, weak volume, crowded longs. → NO TRADE, LOW, risk 9, counter "chasing into bearish 4h on crowded longs".
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


def _setup_line(i: int, s: dict) -> str:
    fvg     = "Y" if s.get("fvg")         else "N"
    ob      = "Y" if s.get("order_block") else "N"
    sweep   = "Y" if s.get("liq_sweep")   else "N"
    funding = s.get("funding_rate")
    fund_s  = f"{funding*100:+.3f}%" if funding is not None else "n/a"
    zone    = f"{s.get('entry_source','?')}:{s.get('entry_low',0):.4g}-{s.get('entry_high',0):.4g}"
    return (
        f"{i} {s['symbol']} {s['direction']} "
        f"S={s.get('mtf_score','?')} "
        f"4h={s.get('trend_4h','?')} 1h={s.get('trend_1h','?')} "
        f"FVG={fvg} OB={ob} SW={sweep} "
        f"Z={zone} RSI={s['rsi']} V={s['volume_ratio']}x F={fund_s}"
    )


def _setup_line_heavy(i: int, s: dict) -> str:
    """Extended setup line for HEAVY analysis — adds session, tags, HTF strength."""
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
    return base + (" " + " ".join(extras) if extras else "")


def _news_block(news_context: dict) -> str:
    if not news_context:
        return ""
    sent = news_context.get("sentiment", "NEUTRAL")
    summ = news_context.get("summary", "")
    if sent != "NEUTRAL" and summ:
        return (
            f"\nNEWS CONTEXT: {sent} — {summ}\n"
            f"Rule: BEARISH news → avoid LONG; BULLISH news → avoid SHORT.\n"
        )
    return ""


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

def _budget_ok(tier: str = "LIGHT") -> bool:
    """Return False when today's Claude spend is within reserve of the daily cap."""
    try:
        spent = get_claude_spend_today()
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
    """Render recent per-coin outcomes as compact memory for the HEAVY call."""
    if not history:
        return "No prior closed trades on this symbol.\n"
    lines = []
    for h in history[:CLAUDE_MEMORY_LIMIT]:
        entry = h.get("entry_price")
        exitp = h.get("exit_price")
        try:
            move = f"{(exitp - entry) / entry * 100:+.1f}%" if entry and exitp else "?"
        except (TypeError, ZeroDivisionError):
            move = "?"
        lines.append(
            f"- {h.get('direction','?')} {h.get('status','?')} "
            f"({move}, conf={h.get('confidence','?')}, S={h.get('mtf_score','?')})"
        )
    return "Recent outcomes on this symbol (newest first):\n" + "\n".join(lines) + "\n"


_THINKING_BETA = "interleaved-thinking-2025-05-14"
_THINKING_BUDGET = 3000  # tokens for internal reasoning scratch-pad


def analyze_heavy(setup: dict, news_context: dict = None, history: list = None) -> dict:
    """
    HEAVY tier. Re-check ONE strong setup with Sonnet + extended thinking.

    Improvements over LIGHT:
    - Extended thinking: Sonnet reasons step-by-step internally before deciding
    - Chain-of-thought prompt: structured analysis questions guide reasoning
    - Devil's advocate: forced consideration of failure before verdict
    - Richer setup line: adds session, MTF tags, HTF strength flags
    - Per-coin memory: last 15 outcomes for pattern recognition
    """
    if not _budget_ok("HEAVY"):
        return {}

    setup_line = _setup_line_heavy(1, setup)

    user_text = (
        f"{_news_block(news_context)}"
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
        f"reliably follow through on this setup type, or does it repeatedly fail?\n"
        f"5. DEVIL'S ADVOCATE — Argue the strongest case AGAINST this trade. "
        f"What specific price action would prove this setup wrong?\n"
        f"6. VERDICT — After weighing all of the above, give your final decision.\n\n"
        f"Then call submit_verdict with your conclusion."
    )

    client = _get_client()

    # Try extended thinking first (gives Sonnet a reasoning scratch-pad).
    # Falls back to standard call if the beta is unavailable.
    try:
        message = client.messages.create(
            model=CLAUDE_HEAVY_MODEL,
            max_tokens=_THINKING_BUDGET + 800,
            thinking={"type": "enabled", "budget_tokens": _THINKING_BUDGET},
            system=_system_param(),
            tools=[_verdict_tool()],
            tool_choice={"type": "tool", "name": "submit_verdict"},
            messages=[{"role": "user", "content": user_text}],
            extra_headers={"anthropic-beta": f"{_CACHE_BETA},{_THINKING_BETA}"},
        )
        _log.info("Claude HEAVY: extended thinking ON")
    except Exception as e_think:
        _log.warning(f"HEAVY thinking mode unavailable ({e_think}), falling back to standard")
        message = client.messages.create(
            model=CLAUDE_HEAVY_MODEL,
            max_tokens=800,
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
