"""Realtime quote helpers (Tencent qt.gtimg.cn) shared by API layer and scripts.

A single HTTP request can return up to ~50 symbols, which avoids the
per-symbol rate limits of most free quote APIs.

Tencent payload layout (fields split by '~'):
    [1]=name  [2]=symbol  [3]=price  [4]=pre_close  [5]=open  [6]=volume(lots)
    [31]=change_amount  [32]=change_pct  [33]=high  [34]=low
"""
import logging
from typing import Dict, List, Optional

import httpx

logger = logging.getLogger(__name__)

QUOTE_URL = "https://qt.gtimg.cn/q={symbols}"
QUOTE_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
    "Referer": "https://gu.qq.com/",
}
MAX_BATCH = 50


def to_tencent_symbol(symbol: str) -> Optional[str]:
    """Normalize a symbol to Tencent's market-prefixed form (sh/sz/bj/hk)."""
    sym = str(symbol or "").strip()
    if not sym:
        return None
    sym = sym.split(".")[0]
    low = sym.lower()
    if low.startswith(("sh", "sz", "bj", "hk")):
        return low
    if not sym.isdigit():
        return None
    if len(sym) == 5:  # HK codes are 5 digits (e.g. 00700)
        return f"hk{sym}"
    if sym.startswith(("6", "9")):
        return f"sh{sym}"
    if sym.startswith(("4", "8")):
        return f"bj{sym}"
    return f"sz{sym}"


def parse_tencent_payload(text: str) -> Dict[str, dict]:
    """Parse a Tencent multi-symbol response into {symbol: quote}."""
    quotes: Dict[str, dict] = {}
    for part in text.split(";"):
        part = part.strip()
        if '="' not in part:
            continue
        try:
            payload = part.split('="', 1)[1].rstrip('";\n ')
            fields = payload.split("~")
            if len(fields) < 35:
                continue
            sym = fields[2] or ""
            if not sym:
                continue
            volume_lots = float(fields[6]) if fields[6] else 0.0
            quotes[sym] = {
                "symbol": sym,
                "name": fields[1] or sym,
                "price": float(fields[3]) if fields[3] else 0.0,
                "pre_close": float(fields[4]) if fields[4] else 0.0,
                "open": float(fields[5]) if fields[5] else 0.0,
                "volume": int(volume_lots * 100),
                "change_pct": float(fields[32]) if fields[32] else 0.0,
                "change_amount": float(fields[31]) if fields[31] else 0.0,
                "high": float(fields[33]) if fields[33] else 0.0,
                "low": float(fields[34]) if fields[34] else 0.0,
            }
        except (ValueError, IndexError) as e:
            logger.debug("Tencent parse error: %s", e)
            continue
    return quotes


async def fetch_tencent_quotes(symbols: List[str]) -> Dict[str, dict]:
    """Batch fetch realtime quotes. Returns {symbol: quote} for resolved symbols."""
    tx_syms: List[str] = []
    seen = set()
    for sym in symbols:
        tx = to_tencent_symbol(sym)
        if tx and tx not in seen:
            seen.add(tx)
            tx_syms.append(tx)
    if not tx_syms:
        return {}

    quotes: Dict[str, dict] = {}
    for i in range(0, len(tx_syms), MAX_BATCH):
        chunk = tx_syms[i:i + MAX_BATCH]
        url = QUOTE_URL.format(symbols=",".join(chunk))
        try:
            async with httpx.AsyncClient(timeout=8.0, headers=QUOTE_HEADERS) as client:
                resp = await client.get(url)
                resp.raise_for_status()
                quotes.update(parse_tencent_payload(resp.text))
        except Exception as e:
            logger.warning("Tencent batch fetch failed (%s): %s", ",".join(chunk), e)
    return quotes
