// Cloudflare Worker — proxy for Bybit market data (bypasses US geoblock).
//
// SETUP (free, ~5 min):
//   1. Go to https://dash.cloudflare.com → Workers & Pages → Create → Worker
//   2. Name it e.g. "bybit-proxy", click Deploy
//   3. Click "Edit code", DELETE the default code, PASTE everything below
//   4. Click Deploy
//   5. Copy your worker URL (e.g. https://bybit-proxy.YOURNAME.workers.dev)
//   6. In Render → Environment → add variable:
//        BYBIT_PROXY_BASE = https://bybit-proxy.YOURNAME.workers.dev
//   7. Save → Render redeploys → bot fetches Bybit through the worker.
//
// The worker forwards the request path + query to api.bybit.com unchanged,
// so /v5/market/tickers?category=spot just works.

export default {
  async fetch(request) {
    const url = new URL(request.url);
    const target = "https://api.bybit.com" + url.pathname + url.search;
    try {
      const upstream = await fetch(target, {
        method: "GET",
        headers: { "User-Agent": "Mozilla/5.0" },
        cf: { cacheTtl: 0 },
      });
      return new Response(upstream.body, {
        status: upstream.status,
        headers: {
          "Content-Type": "application/json",
          "Access-Control-Allow-Origin": "*",
        },
      });
    } catch (err) {
      return new Response(JSON.stringify({ error: String(err) }), {
        status: 502,
        headers: { "Content-Type": "application/json" },
      });
    }
  },
};
