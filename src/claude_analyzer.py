import anthropic
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    CLAUDE_API_KEY, CLAUDE_LIGHT_MODEL, CLAUDE_HEAVY_MODEL,
    CLAUDE_MAX_RISK_SCORE, CLAUDE_CACHE_TTL, CLAUDE_MEMORY_LIMIT,
)

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
- mtf_score (S): multi-timeframe confluence score, roughly 0–15. Higher = more aligned signals across 15m/1h/4h. S>=12 is strong, 9–11 is acceptable, below 9 should rarely pass.
- 4h / 1h: higher-timeframe trend bias (bull / bear / neutral). The suggested side should agree with the dominant higher-timeframe trend. Both neutral = chop, treat with suspicion.
- FVG: an unfilled Fair Value Gap is present near entry (imbalance the market tends to revisit).
- OB: a valid Order Block (last opposing candle before an impulsive move) sits near entry.
- SW: a liquidity sweep / stop-run occurred (price grabbed liquidity beyond a prior high/low then reversed) — a high-quality reversal trigger.
- Z: the entry zone source and price band (OB or FVG and its low-high range). Price currently retesting this zone is ideal.
- RSI: 14-period relative strength on 15m. Overbought (>72) weakens fresh LONGs; oversold (<28) weakens fresh SHORTs.
- V: volume ratio vs recent average. >1.5x confirms genuine participation behind a move; <1.0x is weak.
- F: perpetual funding rate. Strongly positive funding = crowded longs (squeeze risk for new LONGs); strongly negative = crowded shorts (squeeze risk for new SHORTs).

HOW TO DECIDE
1. Confirm the suggested side only. If you would not take that exact side, return NO TRADE.
2. Prefer setups with S>=12. The strongest possible setup has FVG AND OB AND an active retest zone in the direction of both the 1h and 4h trend.
3. Confluence stacking: each of FVG, OB, SW that agrees with the side adds confidence. Two or more confirmations plus trend alignment is a HIGH-confidence trade. One confirmation with trend alignment is MEDIUM. Zero confirmations, or confirmations that fight the trend, is LOW → usually NO TRADE.
4. A neutral higher timeframe is tolerable if the other timeframe is clearly aligned and confirmations are strong → cap confidence at MEDIUM. Both timeframes neutral = chop; demand a clean liquidity sweep or pass.
5. Reject overextended entries: a LONG with RSI>72 or a SHORT with RSI<28 is chasing — downgrade hard or NO TRADE unless a fresh sweep justifies it.
6. Respect crowded funding: avoid new LONGs into strongly positive funding and new SHORTs into strongly negative funding (squeeze risk).
7. Volume below average (V<1.0x) on a breakout-style setup is a red flag — the move lacks conviction.
8. News overrides structure: BEARISH macro news → do not open LONGs; BULLISH macro news → do not open SHORTs. When told the market is paused or a major event is live, prefer NO TRADE.
9. OFF_HOURS or thin-liquidity context = NO TRADE.

RISK SCORE (0–10): rate how dangerous taking this trade is RIGHT NOW. 0–3 = clean, well-located, trend-aligned. 4–7 = tradeable but with a real concern (mild overextension, one timeframe neutral, average volume). 8–10 = serious problem (chasing extended price, fighting the trend, crowded funding into the move, hostile news, far from any retest zone). A high risk_score should almost always pair with NO TRADE — be honest, do not soften it.

COUNTER-ARGUMENT: in a few words, state the single best reason this trade could fail (the strongest bear case for a LONG, or bull case for a SHORT). Always provide one even for good setups — every trade has a failure mode. Example counters: "4h still bearish, fighting trend", "RSI 74 — late entry, chasing", "funding +0.09% — crowded longs", "no retest, price 2% above OB", "volume 0.8x — weak conviction".

TREND_STRENGTH (0–10): how strongly the higher timeframes back the suggested side. 0 = timeframes oppose the side, 5 = mixed/neutral, 10 = both 1h and 4h firmly aligned with the side.

CONFIDENCE: HIGH = multiple confirmations, trend-aligned, well-located, low risk. MEDIUM = decent setup with one notable caveat. LOW = weak/conflicted — pair with NO TRADE unless marginal.

WORKED EXAMPLES
- "BTC-USDT LONG S=13 4h=bull 1h=bull FVG=Y OB=Y SW=N Z=OB:64000-64200 RSI=58 V=1.9x F=+0.01%": trend-aligned, two confirmations, healthy RSI, strong volume, neutral funding. → LONG, HIGH, risk 2, counter "minor: no sweep, relies on OB hold".
- "SOL-USDT LONG S=10 4h=neutral 1h=bull FVG=N OB=Y SW=Y Z=OB:140-142 RSI=49 V=1.6x F=-0.02%": one timeframe neutral but 1h aligned, OB + sweep, good location. → LONG, MEDIUM, risk 4, counter "4h neutral — no higher-tf push".
- "XRP-USDT LONG S=8 4h=bear 1h=neutral FVG=N OB=N SW=N Z=FVG:0.50-0.51 RSI=74 V=0.7x F=+0.08%": fighting 4h trend, no confirmations, overbought, weak volume, crowded longs. → NO TRADE, LOW, risk 9, counter "chasing into bearish 4h on crowded longs".
- "LINK-USDT SHORT S=11 4h=bear 1h=bear FVG=Y OB=N SW=Y Z=FVG:13.0-13.2 RSI=41 V=1.7x F=+0.00%": trend-aligned short, FVG + sweep, RSI has room, solid volume. → SHORT, HIGH, risk 3, counter "broad-market bounce could squeeze".

OUTPUT
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
    """Single-setup variant for the HEAVY (Sonnet) tier."""
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
                "reason":         {"type": "string", "description": "Why this verdict, <=10 words."},
                "counter":        {"type": "string", "description": "Single strongest reason the trade could fail."},
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


def analyze_heavy(setup: dict, news_context: dict = None, history: list = None) -> dict:
    """
    HEAVY tier. Re-check ONE strong setup with the bigger Sonnet model plus
    per-coin memory of recent outcomes. Returns a verdict dict (same fields as
    LIGHT). No extended thinking — kept off for cost/latency on high-confluence
    setups. Caller decides whether to override the LIGHT verdict with this one.
    """
    user_text = (
        f"{_news_block(news_context)}"
        f"{_memory_block(history)}\n"
        f"Deep second-opinion on this single setup. Weigh the symbol's recent "
        f"outcomes above: if this exact side keeps losing here, be stricter. "
        f"Return one verdict via submit_verdict:\n{_setup_line(1, setup)}"
    )

    client = _get_client()
    message = client.messages.create(
        model=CLAUDE_HEAVY_MODEL,
        max_tokens=400,
        system=_system_param(),
        tools=[_verdict_tool()],
        tool_choice={"type": "tool", "name": "submit_verdict"},
        messages=[{"role": "user", "content": user_text}],
        extra_headers={"anthropic-beta": _CACHE_BETA},
    )

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
