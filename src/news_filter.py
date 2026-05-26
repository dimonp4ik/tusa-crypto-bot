"""
CryptoPanic news filter — blocks signals if recent bad news.
Free public API endpoint (no key required for basic use).
"""

import requests
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import CRYPTOPANIC_API_KEY, NEWS_BLOCK_KEYWORDS


def check_news_sentiment(symbol: str) -> dict:
    """
    Fetch recent news for the coin and check for bad-news keywords.
    Returns {'safe': bool, 'reason': str}.
    `symbol` is in KuCoin format like 'BTC-USDT' — we strip to 'BTC'.
    """
    coin = symbol.replace("-USDT", "")

    try:
        params = {"currencies": coin, "public": "true", "filter": "hot"}
        if CRYPTOPANIC_API_KEY:
            params["auth_token"] = CRYPTOPANIC_API_KEY

        resp = requests.get(
            "https://cryptopanic.com/api/v1/posts/",
            params=params,
            timeout=8,
        )
        if resp.status_code != 200:
            return {"safe": True, "reason": "news API unavailable"}

        posts = resp.json().get("results", [])[:10]  # last 10 headlines
        for post in posts:
            title = (post.get("title") or "").lower()
            for kw in NEWS_BLOCK_KEYWORDS:
                if kw in title:
                    return {"safe": False, "reason": f"bad news: '{kw}' in '{title[:60]}'"}

        return {"safe": True, "reason": ""}

    except Exception:
        # Fail open — don't block on API errors
        return {"safe": True, "reason": "news check failed"}
