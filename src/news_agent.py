"""
Global macro news agent.

Sources (all free, no API key needed):
  - Reuters Business RSS
  - CNBC Markets RSS
  - BBC Business RSS
  - CoinDesk RSS

AI: Groq free tier (llama-3.1-8b-instant) — 14 400 req/day, ~200ms latency.
Register free at https://groq.com → API Keys → Create Key → set GROQ_API_KEY in Render.

Runs once per scan. Returns:
  sentiment  : BULLISH | BEARISH | NEUTRAL
  summary    : one-line key event (max 15 words)
  pause      : True only on extreme events (war, total ban, major crash)
  headlines  : list of fetched titles
"""

import xml.etree.ElementTree as ET
import requests as _req
import sys
import os
from datetime import datetime, timezone, timedelta
from email.utils import parsedate_to_datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import GROQ_API_KEY, NEWS_LOOKBACK_HOURS

# RSS sources — all public, no registration
RSS_FEEDS = [
    ("Reuters",   "https://feeds.reuters.com/reuters/businessNews"),
    ("CNBC",      "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=15839135"),
    ("BBC Biz",   "https://feeds.bbci.co.uk/news/business/rss.xml"),
    ("CoinDesk",  "https://www.coindesk.com/arc/outboundfeeds/rss/"),
    ("Decrypt",   "https://decrypt.co/feed"),
]


def _fetch_rss(url: str, timeout: int = 8) -> list[dict]:
    """Fetch and parse a single RSS feed. Returns list of {title, published_utc}."""
    try:
        resp = _req.get(url, timeout=timeout,
                        headers={"User-Agent": "Mozilla/5.0 CryptoBot/1.0"})
        if resp.status_code != 200:
            return []
        root = ET.fromstring(resp.content)
        items = []
        for item in root.iter("item"):
            title = item.findtext("title", "").strip()
            pub   = item.findtext("pubDate", "")
            if not title:
                continue
            try:
                pub_dt = parsedate_to_datetime(pub).astimezone(timezone.utc) if pub else None
            except Exception:
                pub_dt = None
            items.append({"title": title, "published": pub_dt})
        return items
    except Exception:
        return []


def fetch_recent_headlines(hours: int = NEWS_LOOKBACK_HOURS) -> list[str]:
    """Collect headlines from all RSS sources published in last `hours` hours."""
    cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
    result = []

    for name, url in RSS_FEEDS:
        items = _fetch_rss(url)
        for it in items[:15]:
            pub = it["published"]
            if pub and pub < cutoff:
                continue   # too old
            result.append(f"[{name}] {it['title']}")

    return result[:35]  # cap at 35 headlines to keep prompt small


def analyze_with_groq(headlines: list[str]) -> dict:
    """
    Send headlines to Groq llama-3.1-8b-instant.
    Free tier: 14 400 req/day, ~200ms.
    """
    if not headlines:
        return {"sentiment": "NEUTRAL", "summary": "No recent news", "pause": False}

    if not GROQ_API_KEY:
        return {"sentiment": "NEUTRAL", "summary": "GROQ_API_KEY not set", "pause": False}

    text = "\n".join(f"• {h}" for h in headlines)

    prompt = f"""You are a crypto market analyst. Read these recent global headlines and assess their impact on crypto markets.

{text}

Reply in EXACTLY this format (3 lines):
SENTIMENT: BULLISH or BEARISH or NEUTRAL
PAUSE: YES or NO  (YES only for: exchange hacks >$500M, total regulatory ban, major war start)
SUMMARY: [max 12 words describing the key market-moving event]"""

    try:
        resp = _req.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model":       "llama-3.1-8b-instant",
                "messages":    [{"role": "user", "content": prompt}],
                "max_tokens":  100,
                "temperature": 0.1,
            },
            timeout=12,
        )
        raw = resp.json()["choices"][0]["message"]["content"].strip()
        return _parse_groq(raw)

    except Exception as e:
        return {"sentiment": "NEUTRAL", "summary": f"Groq unavailable: {e}", "pause": False}


def _parse_groq(raw: str) -> dict:
    result = {"sentiment": "NEUTRAL", "summary": "", "pause": False}
    for line in raw.splitlines():
        line = line.strip()
        if line.startswith("SENTIMENT:"):
            val = line.split(":", 1)[1].strip().upper()
            if "BULLISH" in val:   result["sentiment"] = "BULLISH"
            elif "BEARISH" in val: result["sentiment"] = "BEARISH"
        elif line.startswith("PAUSE:"):
            result["pause"] = "YES" in line.upper()
        elif line.startswith("SUMMARY:"):
            result["summary"] = line.split(":", 1)[1].strip()
    return result


def detect_major_events(headlines: list[str]) -> list[dict]:
    """
    Detect HIGH IMPACT macro events from headlines.
    Returns list of {name, direction, level (1-3), explanation} dicts.
    Max 2 events per call.
    """
    if not headlines or not GROQ_API_KEY:
        return []

    text = "\n".join(f"• {h}" for h in headlines)

    prompt = f"""Analyze these headlines for HIGH IMPACT macro events that significantly move crypto markets.

HIGH IMPACT events only: Fed rate decision, ECB decision, US CPI release, NFP jobs report, GDP surprise, major crypto regulatory ban, exchange collapse or hack >$500M.

Headlines:
{text}

If HIGH IMPACT event found, reply one line per event (max 2 events):
EVENT|[event name in Russian, max 8 words]|BULLISH or BEARISH|[1 or 2 or 3]|[market effect in Russian, max 10 words]

Impact scale: 1=moderate, 2=significant, 3=major market mover

If NO high impact events found, reply only: NONE"""

    try:
        resp = _req.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model":       "llama-3.1-8b-instant",
                "messages":    [{"role": "user", "content": prompt}],
                "max_tokens":  150,
                "temperature": 0.1,
            },
            timeout=12,
        )
        raw = resp.json()["choices"][0]["message"]["content"].strip()
        return _parse_events(raw)
    except Exception:
        return []


def _parse_events(raw: str) -> list[dict]:
    """Parse: EVENT|name|direction|level|explanation"""
    events = []
    for line in raw.splitlines():
        line = line.strip()
        if not line or line.upper() == "NONE":
            continue
        if not line.startswith("EVENT|"):
            continue
        parts = line.split("|")
        if len(parts) < 5:
            continue
        try:
            direction = parts[2].strip().upper()
            if direction not in ("BULLISH", "BEARISH"):
                direction = "NEUTRAL"
            level = int(parts[3].strip())
            events.append({
                "name":        parts[1].strip(),
                "direction":   direction,
                "level":       level,
                "explanation": parts[4].strip(),
            })
        except (ValueError, IndexError):
            continue
    return events[:2]


def get_market_news() -> dict:
    """
    Main entry point. Fetch headlines → analyze → return context dict.
    Always succeeds (errors return NEUTRAL).
    """
    headlines = fetch_recent_headlines()
    analysis  = analyze_with_groq(headlines)
    return {
        "sentiment":       analysis["sentiment"],
        "summary":         analysis["summary"],
        "pause":           analysis["pause"],
        "headline_count":  len(headlines),
    }
