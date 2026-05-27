import anthropic
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import CLAUDE_API_KEY

# Reuse client across calls
_client = None


def _get_client():
    global _client
    if _client is None:
        _client = anthropic.Anthropic(api_key=CLAUDE_API_KEY)
    return _client


# ── Batch SMC analysis (main) ─────────────────────────────────────────────────

def analyze_batch_with_claude(setups: list, news_context: dict = None) -> list:
    """
    Send ALL filtered setups to Claude Haiku in ONE call.
    Returns list of result dicts, one per setup.
    """
    if not setups:
        return []

    coins_text = ""
    for i, s in enumerate(setups, 1):
        fvg     = "✓" if s.get("fvg")         else "✗"
        ob      = "✓" if s.get("order_block") else "✗"
        sweep   = "✓" if s.get("liq_sweep")   else "✗"
        funding = s.get("funding_rate")
        fund_s  = f"{funding*100:+.3f}%" if funding is not None else "n/a"
        coins_text += (
            f"{i}. {s['symbol']} → {s['direction']} | "
            f"4h:{s.get('trend_4h','?')} 1h:{s.get('trend_1h','?')} | "
            f"BOS:{s.get('bos','?')} | FVG:{fvg} OB:{ob} Sweep:{sweep} | "
            f"RSI:{s['rsi']} Vol:{s['volume_ratio']}x Funding:{fund_s}\n"
        )

    # Build news context block
    if news_context:
        news_sentiment = news_context.get("sentiment", "NEUTRAL")
        news_summary   = news_context.get("summary", "")
        news_block = (
            f"\nGLOBAL NEWS CONTEXT:\n"
            f"  Market sentiment: {news_sentiment}\n"
            f"  Key event: {news_summary}\n"
            f"  Rule: if news=BEARISH → avoid LONG (lower confidence); "
            f"if news=BULLISH → avoid SHORT (lower confidence)\n"
        )
    else:
        news_block = ""

    prompt = f"""You are a Smart Money Concepts (SMC) crypto trader. Analyze each setup and decide whether to trade.
{news_block}
Rules:
- LONG only if 4h=bullish AND 1h=bullish AND BOS=bullish (strongest)
- SHORT only if 4h=bearish AND 1h=bearish AND BOS=bearish (strongest)
- If one timeframe is neutral — still trade but lower confidence
- Skip (NO TRADE) if RSI > 75 on LONG or RSI < 25 on SHORT (overextended)
- FVG + OB together = highest probability setup
- Volume above 2x = institutional confirmation
- Funding > +0.05% means crowded LONG → prefer SHORT, avoid LONG
- Funding < -0.05% means crowded SHORT → prefer LONG, avoid SHORT

Setups to analyze:
{coins_text}
Reply EXACTLY one line per setup, same numbering:
1. DECISION|REASON (max 8 words)|CONFIDENCE
2. DECISION|REASON (max 8 words)|CONFIDENCE
...

DECISION must be: LONG or SHORT or NO TRADE
CONFIDENCE must be: HIGH or MEDIUM or LOW"""

    client = _get_client()
    message = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=60 * len(setups) + 50,
        messages=[{"role": "user", "content": prompt}],
    )

    raw = message.content[0].text.strip()
    return _parse_batch_response(raw, setups)


def _parse_batch_response(raw: str, setups: list) -> list:
    """Parse Claude's multi-line response into a list of result dicts."""
    # Build index: "1" → line text
    line_map = {}
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        dot = line.find(".")
        if dot > 0:
            key = line[:dot].strip()
            if key.isdigit():
                line_map[key] = line[dot + 1:].strip()

    results = []
    for i, setup in enumerate(setups, 1):
        base = {
            "symbol":        setup["symbol"],
            "direction":     setup["direction"],
            "current_price": setup["current_price"],
            "recent_high":   setup.get("recent_high", 0),
            "recent_low":    setup.get("recent_low", 0),
            "rsi":           setup["rsi"],
            "volume_ratio":  setup["volume_ratio"],
            "signals":       setup["signals"],
            "decision":      "NO TRADE",
            "reason":        "Not evaluated",
            "confidence":    "LOW",
        }

        text = line_map.get(str(i), "")
        if text:
            parts = [p.strip() for p in text.split("|")]
            if parts:
                val = parts[0].upper()
                if "LONG"  in val: base["decision"] = "LONG"
                elif "SHORT" in val: base["decision"] = "SHORT"
                else:                base["decision"] = "NO TRADE"
            if len(parts) >= 2:
                base["reason"] = parts[1]
            if len(parts) >= 3:
                conf = parts[2].upper()
                if "HIGH"   in conf: base["confidence"] = "HIGH"
                elif "MEDIUM" in conf: base["confidence"] = "MEDIUM"
                else:                  base["confidence"] = "LOW"

        results.append(base)

    return results


# ── Legacy single-coin analysis (kept for reference) ─────────────────────────

def analyze_with_claude(setup: dict) -> dict:
    """
    Original single-coin analysis. Not used in main scan anymore.
    Kept for reference / manual testing.
    """
    signals_text = ", ".join(setup["signals"])

    prompt = f"""You are a crypto trading signal validator. Analyze this setup and decide if it is tradeable.

Symbol: {setup['symbol']}
Suggested Direction: {setup['direction']}
Current Price: {setup['current_price']}
RSI: {setup['rsi']}
Volume ratio vs average: {setup['volume_ratio']}x
Technical signals triggered: {signals_text}

Reply in EXACTLY this format (3 lines, no extra text):
DECISION: LONG or SHORT or NO TRADE
REASON: one sentence
CONFIDENCE: HIGH or MEDIUM or LOW"""

    client = _get_client()
    message = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=120,
        messages=[{"role": "user", "content": prompt}],
    )

    raw = message.content[0].text.strip()

    result = {
        "symbol":        setup["symbol"],
        "direction":     setup["direction"],
        "current_price": setup["current_price"],
        "recent_high":   setup.get("recent_high", 0),
        "recent_low":    setup.get("recent_low", 0),
        "rsi":           setup["rsi"],
        "volume_ratio":  setup["volume_ratio"],
        "signals":       setup["signals"],
        "decision":      "NO TRADE",
        "reason":        "Could not parse Claude response.",
        "confidence":    "LOW",
    }

    for line in raw.splitlines():
        line = line.strip()
        if line.startswith("DECISION:"):
            val = line.replace("DECISION:", "").strip().upper()
            if "LONG" in val:   result["decision"] = "LONG"
            elif "SHORT" in val: result["decision"] = "SHORT"
            else:                result["decision"] = "NO TRADE"
        elif line.startswith("REASON:"):
            result["reason"] = line.replace("REASON:", "").strip()
        elif line.startswith("CONFIDENCE:"):
            result["confidence"] = line.replace("CONFIDENCE:", "").strip().upper()

    return result
