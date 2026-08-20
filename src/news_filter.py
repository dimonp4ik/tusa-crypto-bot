"""Per-symbol bad-news gate.

Rebuilt 2026-08-16. The previous version queried CryptoPanic's v1 public API,
which now answers 403 behind Cloudflare without a key — and the function failed
open on any non-200, so every call returned "safe". The gate had been silently
inert, and nothing logged it: a filter that fails open invisibly is worse than
no filter, because it also removes the pressure to notice.

Two things changed:

  * Source is the RSS set news_agent already reads (Cointelegraph, Decrypt,
    CryptoSlate, BBC Business). Verified live on 2026-08-16: 4 of the 7 feeds
    respond, no key needed. Headlines are cached module-side because this is
    called once per candidate setup, and refetching per symbol would hammer the
    feeds and add seconds to every scan.

  * Matching is on word boundaries. The old code did `kw in title`, so "ban"
    matched "banking", "urban" and "Albania", and the keyword list carried
    "sec " with a trailing space as a hand-rolled workaround for "second".

The returned dict gains `checked`: False means the gate could not run. Callers
must not read that as clean.
"""

import re
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import NEWS_BLOCK_KEYWORDS, QUOTE_ASSET

_CACHE_TTL_SEC = 600
_cache: dict = {"ts": 0.0, "titles": []}

# NEWS_LOOKBACK_HOURS is 2, tuned for the macro sentiment agent that runs once a
# scan and wants only fresh tape. A per-coin disaster gate needs a longer memory:
# an exploit announced six hours ago still matters to a position opened now.
# Measured 2026-08-16: at 2h the feeds yielded 4 headlines total, far too thin to
# gate on.
_SYMBOL_LOOKBACK_HOURS = 24

# Headlines name the asset, tickers rarely appear. Only majors need an alias —
# for anything else the bare ticker is the best available handle.
_ALIASES = {
    "BTC": ("bitcoin",), "ETH": ("ethereum", "ether"), "SOL": ("solana",),
    "XRP": ("ripple",), "ADA": ("cardano",), "DOT": ("polkadot",),
    "AVAX": ("avalanche",), "LINK": ("chainlink",), "XLM": ("stellar",),
    "SUI": ("sui",), "NEAR": ("near protocol",), "AAVE": ("aave",),
    "ZEC": ("zcash",), "TAO": ("bittensor",), "SEI": ("sei",),
    "HYPE": ("hyperliquid",), "DOGE": ("dogecoin",), "LTC": ("litecoin",),
}


def _kw_pattern() -> re.Pattern:
    words = [k.strip() for k in NEWS_BLOCK_KEYWORDS if k and k.strip()]
    return re.compile(r"\b(" + "|".join(re.escape(w) for w in words) + r")\b", re.I)


_KW = _kw_pattern()


def _headlines() -> list[str]:
    now = time.time()
    if now - _cache["ts"] < _CACHE_TTL_SEC and _cache["titles"]:
        return _cache["titles"]
    try:
        from src.news_agent import fetch_recent_headlines
        titles = fetch_recent_headlines(hours=_SYMBOL_LOOKBACK_HOURS)
    except Exception:
        titles = []
    if titles:
        _cache["ts"], _cache["titles"] = now, titles
    return titles


def check_news_sentiment(symbol: str) -> dict:
    """{'safe': bool, 'reason': str, 'checked': bool}.

    `checked` is False when no headlines could be fetched — the caller sees a
    permissive `safe` but knows it was never actually verified.
    """
    coin = str(symbol or "").upper().replace("-", "").replace("/", "").replace("_", "")
    if coin.endswith(QUOTE_ASSET):
        coin = coin[:-len(QUOTE_ASSET)]
    if not coin:
        return {"safe": True, "reason": "no symbol", "checked": False}

    titles = _headlines()
    if not titles:
        return {"safe": True, "reason": "news feeds unavailable", "checked": False}

    names = (coin.lower(),) + _ALIASES.get(coin, ())
    name_re = re.compile(r"\b(" + "|".join(re.escape(n) for n in names) + r")\b", re.I)

    for title in titles:
        if not name_re.search(title):
            continue
        hit = _KW.search(title)
        if hit:
            return {
                "safe": False,
                "reason": f"bad news: '{hit.group(0)}' in '{title[:70]}'",
                "checked": True,
            }
    return {"safe": True, "reason": "", "checked": True}
