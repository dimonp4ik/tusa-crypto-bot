import anthropic
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import CLAUDE_API_KEY

# Reuse client across calls to avoid reconnecting every time
_client = None


def _get_client():
    global _client
    if _client is None:
        _client = anthropic.Anthropic(api_key=CLAUDE_API_KEY)
    return _client


def analyze_with_claude(setup: dict) -> dict:
    """
    Send a pre-filtered setup to Claude Haiku for final confirmation.
    Returns a result dict with 'decision' = LONG | SHORT | NO TRADE.
    """
    signals_text = ", ".join(setup["signals"])

    prompt = f"""You are a crypto trading signal validator. Analyze this setup and decide if it is tradeable.

Symbol: {setup['symbol']}
Suggested Direction: {setup['direction']}
Current Price: {setup['current_price']}
RSI: {setup['rsi']}
EMA9: {setup['ema9']} | EMA21: {setup['ema21']}
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
    return _parse_response(raw, setup)


def _parse_response(raw: str, setup: dict) -> dict:
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
            if "LONG" in val:
                result["decision"] = "LONG"
            elif "SHORT" in val:
                result["decision"] = "SHORT"
            else:
                result["decision"] = "NO TRADE"
        elif line.startswith("REASON:"):
            result["reason"] = line.replace("REASON:", "").strip()
        elif line.startswith("CONFIDENCE:"):
            result["confidence"] = line.replace("CONFIDENCE:", "").strip().upper()

    return result
