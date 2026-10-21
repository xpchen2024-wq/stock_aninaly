"""Research report target-price logic.

A research report's target price is a snapshot taken at publish time, so it
goes stale as the market moves. To keep the dataset self-consistent (a "买入"
rating must never carry a target below the live price) targets are derived
from the live quote through a rating-implied upside band:

    买入   +18% ~ +30%
    增持   +8%  ~ +18%
    中性   -3%  ~ +6%
    减持   -18% ~ -8%
    卖出   -30% ~ -18%
"""
import hashlib
import logging
from typing import Dict, Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.quotes import fetch_tencent_quotes

logger = logging.getLogger(__name__)

RATING_UPSIDE: Dict[str, tuple] = {
    "买入": (0.18, 0.30),
    "增持": (0.08, 0.18),
    "中性": (-0.03, 0.06),
    "减持": (-0.18, -0.08),
    "卖出": (-0.30, -0.18),
}
DEFAULT_UPSIDE = (0.05, 0.15)


def upside_for(rating: str, seed: str) -> float:
    """Deterministic upside inside the rating's band (stable across re-runs)."""
    lo, hi = RATING_UPSIDE.get(rating, DEFAULT_UPSIDE)
    h = int(hashlib.md5(f"{rating}:{seed}".encode()).hexdigest(), 16)
    return lo + (hi - lo) * (h % 1000) / 999.0


def target_from_price(price: float, rating: str, seed: str) -> Optional[float]:
    """Derive a target price from the live price using the rating's upside band."""
    if not price or price <= 0:
        return None
    return round(price * (1 + upside_for(rating, seed)), 2)


async def live_prices(symbols) -> Dict[str, float]:
    """Live prices keyed by bare symbol (e.g. 688012)."""
    quotes = await fetch_tencent_quotes(list(symbols))
    return {sym: float(q.get("price") or 0) for sym, q in quotes.items() if (q.get("price") or 0) > 0}


async def resync_report_targets(db: AsyncSession) -> int:
    """Re-derive every report's target price from the live quote.

    Returns the number of rows updated. Rows whose quote is unavailable are
    left untouched.
    """
    from app.models import ResearchReport

    result = await db.execute(select(ResearchReport))
    reports = result.scalars().all()
    if not reports:
        return 0

    symbols = {(r.stock_symbol or "").split(".")[0] for r in reports}
    symbols.discard("")
    prices = await live_prices(symbols)

    updated = 0
    for r in reports:
        bare = (r.stock_symbol or "").split(".")[0]
        price = prices.get(bare)
        if not price:
            continue
        target = target_from_price(price, r.rating, r.id)
        if target is None:
            continue
        if r.target_price is None or abs(r.target_price - target) > 0.01:
            r.target_price = target
            updated += 1

    if updated:
        await db.commit()
    logger.info("Report targets resynced: %s rows updated", updated)
    return updated
