# ============================================================================
# AI Stock Analysis Platform - Data API (DS-001 ~ DS-005)
# ============================================================================
from __future__ import annotations

import asyncio
import logging
from typing import Optional, List
from datetime import datetime, timedelta

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, delete

from app.database import get_db
from app.adapters import DataType, FailoverManager, create_default_failover_manager
from app.cache import cache_realtime_quote, get_cached_quote
from app.config import get_settings
from app.models import KlineCache, StockFundamental

logger = logging.getLogger(__name__)

router = APIRouter()
settings = get_settings()

# Global failover manager (initialized lazily)
_failover: Optional[FailoverManager] = None


def get_failover() -> FailoverManager:
    global _failover
    if _failover is None:
        _failover = create_default_failover_manager(tushare_token=settings.TUSHARE_TOKEN)
    return _failover


# -- Schemas --
class KlineResponse(BaseModel):
    symbol: str
    frequency: str
    data: list


class QuoteResponse(BaseModel):
    symbol: str
    name: str
    price: float
    change_pct: float
    volume: int
    high: float
    low: float
    open: float
    pre_close: float


class FundamentalResponse(BaseModel):
    symbol: str
    name: str
    pe_ratio: Optional[float]
    pb_ratio: Optional[float]
    market_cap: Optional[float]
    industry: Optional[str]
    roe: Optional[float]


# -- Routes --
@router.get("/kline", response_model=KlineResponse)
async def get_kline(
    symbol: str = Query(..., description="Stock symbol, e.g. 000001"),
    frequency: str = Query("D", description="D/W/M/5min/15min/30min/60min"),
    days: int = Query(90, ge=1, le=365),
    db: AsyncSession = Depends(get_db),
):
    """Get K-line data. Watchlist stocks read from cache (NF-005)."""
    # 1. Try local cache (watchlist stocks)
    result = await db.execute(
        select(KlineCache)
        .where(KlineCache.symbol == symbol, KlineCache.frequency == frequency)
        .order_by(KlineCache.trade_date.desc())
        .limit(days)
    )
    cached = result.scalars().all()

    if cached:
        data = [
            {
                "date": str(k.trade_date),
                "open": k.open, "high": k.high, "low": k.low, "close": k.close,
                "volume": k.volume, "amount": k.amount, "turnover_rate": k.turnover_rate,
            }
            for k in reversed(cached)
        ]
        return KlineResponse(symbol=symbol, frequency=frequency, data=data)

    # 2. Fetch from data source (DS-005: on-demand fetch)
    manager = get_failover()
    end_date = datetime.now().strftime("%Y%m%d")
    start_date = (datetime.now() - timedelta(days=days)).strftime("%Y%m%d")

    try:
        klines = await manager.fetch_with_failover(
            DataType.KLINE, symbol,
            frequency=frequency, start_date=start_date, end_date=end_date,
        )
        data = [
            {
                "date": k.date, "open": k.open, "high": k.high,
                "low": k.low, "close": k.close,
                "volume": k.volume, "amount": k.amount,
            }
            for k in klines
        ]
        return KlineResponse(symbol=symbol, frequency=frequency, data=data)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"All data sources failed: {e}")


@router.get("/quote", response_model=QuoteResponse)
async def get_realtime_quote(symbol: str = Query(...)):
    """Get real-time quote. Cached in Redis (DS-002).

    Multi-source strategy: Redis cache -> AkShare(East Money) -> Sina -> Yahoo.
    Each source is independent and we fall through on failure.
    """
    from app.adapters import AkShareAdapter, SourceInfo, DataType, SourcePriority

    # 1. Check Redis cache
    cached = await get_cached_quote(symbol)
    if cached:
        return QuoteResponse(**cached)

    # 2. Try Tencent Finance (most reliable, no rate limit, supports both SH/SZ)
    try:
        tx_quote = await _fetch_quote_tencent(symbol)
        if tx_quote:
            data = {
                "symbol": tx_quote["symbol"], "name": tx_quote["name"],
                "price": tx_quote["price"], "change_pct": tx_quote["change_pct"],
                "volume": tx_quote.get("volume", 0),
                "high": tx_quote.get("high", 0.0), "low": tx_quote.get("low", 0.0),
                "open": tx_quote.get("open", 0.0), "pre_close": tx_quote.get("pre_close", 0.0),
            }
            await cache_realtime_quote(symbol, data, ttl=120)
            return QuoteResponse(**data)
    except Exception as e:
        logger.warning(f"Tencent quote failed for {symbol}: {e}")

    # 3. Try East Money direct (fallback)
    try:
        em_quote = await _fetch_quote_eastmoney(symbol)
        if em_quote:
            data = {
                "symbol": em_quote["symbol"], "name": em_quote["name"],
                "price": em_quote["price"], "change_pct": em_quote["change_pct"],
                "volume": em_quote.get("volume", 0),
                "high": em_quote.get("high", 0.0), "low": em_quote.get("low", 0.0),
                "open": em_quote.get("open", 0.0), "pre_close": em_quote.get("pre_close", 0.0),
            }
            await cache_realtime_quote(symbol, data, ttl=120)
            return QuoteResponse(**data)
    except Exception as e:
        logger.warning(f"East Money direct quote failed for {symbol}: {e}")

    # 3. Try AkShare adapter (uses stock_zh_a_spot_em)
    try:
        adapter = AkShareAdapter(SourceInfo(
            name="AkShareQuoteFallback",
            priority=SourcePriority.P1,
            group="cn_stock",
            reliability="medium",
            supported_types=[DataType.REALTIME_QUOTE],
        ))
        quote = await asyncio.wait_for(
            adapter.fetch_realtime_quote(symbol), timeout=8.0
        )
        if quote:
            data = {
                "symbol": quote.symbol, "name": quote.name,
                "price": quote.price, "change_pct": quote.change_pct,
                "volume": quote.volume, "high": quote.high, "low": quote.low,
                "open": quote.open, "pre_close": quote.pre_close,
            }
            await cache_realtime_quote(symbol, data, ttl=settings.REDIS_CACHE_TTL)
            return QuoteResponse(**data)
    except Exception as e:
        logger.warning(f"AkShare quote failed for {symbol}: {e}")

    # 4. Last resort: fail gracefully with hint instead of 502
    raise HTTPException(
        status_code=502,
        detail=f"Quote temporarily unavailable (rate limit). Please retry in 30s."
    )


async def _fetch_quote_eastmoney(symbol: str) -> Optional[dict]:
    """Fetch real-time quote via East Money's lightweight push API.

    This endpoint returns full market snapshot for all stocks in one call,
    avoiding per-symbol rate limits. We then filter for the requested symbol.
    """
    import asyncio
    import httpx

    # Normalize symbol: 6 digits
    sym = symbol.replace(".SH", "").replace(".SZ", "").strip()
    if sym.startswith(("6", "9")):
        secid = f"1.{sym}"  # 1=SH
    else:
        secid = f"0.{sym}"  # 0=SZ

    url = "https://push2.eastmoney.com/api/qt/stock/get"
    params = {
        "secid": secid,
        "fields": "f43,f44,f45,f46,f47,f48,f60,f169,f170,f171,f168",
        "invt": "2",
        "fltt": "2",
    }
    # Field mapping: f43=price(÷100), f44=high, f45=low, f46=open, f47=volume,
    # f48=amount, f60=pre_close, f169=change_amount, f170=change_pct(÷100), f171=total_market_cap

    async with httpx.AsyncClient(
        timeout=8.0,
        headers={
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
            "Referer": "https://quote.eastmoney.com/",
        },
    ) as client:
        resp = await client.get(url, params=params)
        resp.raise_for_status()
        data = resp.json()

    d = data.get("data") or {}
    if not d or d.get("f43") is None:
        return None
    logger.info(f"East Money raw quote for {sym}: f43={d.get('f43')} f44={d.get('f44')} f60={d.get('f60')} f170={d.get('f170')} f58={d.get('f58')}")

    price = d.get("f43", 0) / 100
    pre_close = d.get("f60", 0) / 100
    high = d.get("f44", 0) / 100
    low = d.get("f45", 0) / 100
    open_p = d.get("f46", 0) / 100
    change_pct = d.get("f170", 0) / 100
    volume = int(d.get("f47", 0))

    return {
        "symbol": sym,
        "name": str(d.get("f58", "")) if d.get("f58") else sym,
        "price": price,
        "change_pct": change_pct,
        "volume": volume,
        "high": high,
        "low": low,
        "open": open_p,
        "pre_close": pre_close,
    }


async def _fetch_quote_tencent(symbol: str) -> Optional[dict]:
    """Fetch real-time quote via Tencent Finance's qt.gtimg.cn API.

    Format: v_sh000001="1~name~symbol~price~pre_close~open~volume~...~change_amount~change_pct~high~low~..."
    Field indices (split by '~'):
      [1]=name, [2]=symbol, [3]=current_price, [4]=pre_close, [5]=open,
      [6]=volume (in 手/lots, 1 lot = 100 shares),
      [30]=change_pct, [33]=high, [34]=low,
      [32]=change_amount, [31]=high_52w?, [45]=market_cap(亿)
    """
    import httpx

    sym = symbol.replace(".SH", "").replace(".SZ", "").strip()
    if sym.startswith(("6", "9")):
        tx_sym = f"sh{sym}"
    else:
        tx_sym = f"sz{sym}"

    url = f"https://qt.gtimg.cn/q={tx_sym}"
    async with httpx.AsyncClient(
        timeout=6.0,
        headers={
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
            "Referer": "https://gu.qq.com/",
        },
    ) as client:
        resp = await client.get(url)
        resp.raise_for_status()
        text = resp.text

    # Parse: v_sh000001="1~name~..."
    if "=" not in text or "~\"" not in text and '~"' not in text:
        return None
    try:
        payload = text.split('="', 1)[1].rstrip('";\n')
        fields = payload.split("~")
        if len(fields) < 35:
            return None
        # volume is in 手 (lots) - convert to shares (×100)
        volume_lots = float(fields[6]) if fields[6] else 0.0
        volume_shares = int(volume_lots * 100)
        return {
            "symbol": fields[2] or sym,
            "name": fields[1] or sym,
            "price": float(fields[3]) if fields[3] else 0.0,
            "pre_close": float(fields[4]) if fields[4] else 0.0,
            "open": float(fields[5]) if fields[5] else 0.0,
            "volume": volume_shares,
            "change_pct": float(fields[32]) if fields[32] else 0.0,
            "change_amount": float(fields[31]) if fields[31] else 0.0,
            "high": float(fields[33]) if fields[33] else 0.0,
            "low": float(fields[34]) if fields[34] else 0.0,
        }
    except (ValueError, IndexError) as e:
        logger.debug(f"Tencent parse error for {symbol}: {e}, payload={payload[:200] if 'payload' in dir() else text[:200]}")
        return None


@router.get("/quote/batch")
async def get_realtime_quote_batch(
    symbols: str = Query(..., description="Comma-separated symbols, e.g. 000001,600519"),
):
    """Batch fetch real-time quotes via single Tencent API call.

    Returns dict keyed by symbol. Avoids N separate rate-limited calls.
    """
    import asyncio
    import httpx

    sym_list = [s.strip() for s in symbols.split(",") if s.strip()][:50]
    if not sym_list:
        return {}

    results = {}
    # Check cache first for all
    for sym in sym_list:
        cached = await get_cached_quote(sym)
        if cached:
            results[sym] = cached

    missing = [s for s in sym_list if s not in results]
    if not missing:
        return results

    # Build Tencent symbol list (one call returns all)
    tx_syms = []
    for sym in missing:
        if sym.startswith(("6", "9")):
            tx_syms.append(f"sh{sym}")
        else:
            tx_syms.append(f"sz{sym}")

    url = f"https://qt.gtimg.cn/q={','.join(tx_syms)}"
    try:
        async with httpx.AsyncClient(
            timeout=6.0,
            headers={
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
                "Referer": "https://gu.qq.com/",
            },
        ) as client:
            resp = await client.get(url)
            text = resp.text
    except Exception as e:
        logger.warning(f"Tencent batch fetch failed: {e}")
        return results

    # Parse multiple lines: v_sh000001="...";v_sz000001="...";
    for line in text.strip().split(";\n"):
        line = line.strip()
        if not line or '="' not in line:
            continue
        try:
            tx_sym = line.split("=")[0].replace("v_", "")
            payload = line.split('="', 1)[1].rstrip('";\n')
            fields = payload.split("~")
            if len(fields) < 35:
                continue
            sym = fields[2]
            volume_lots = float(fields[6]) if fields[6] else 0.0
            quote = {
                "symbol": sym, "name": fields[1] or sym,
                "price": float(fields[3]) if fields[3] else 0.0,
                "pre_close": float(fields[4]) if fields[4] else 0.0,
                "open": float(fields[5]) if fields[5] else 0.0,
                "volume": int(volume_lots * 100),
                "change_pct": float(fields[32]) if fields[32] else 0.0,
                "change_amount": float(fields[31]) if fields[31] else 0.0,
                "high": float(fields[33]) if fields[33] else 0.0,
                "low": float(fields[34]) if fields[34] else 0.0,
            }
            results[sym] = quote
            try:
                await cache_realtime_quote(sym, quote, ttl=120)
            except Exception:
                pass
        except (ValueError, IndexError) as e:
            logger.debug(f"Tencent batch parse error: {e}")
            continue

    return results


@router.get("/fundamental", response_model=FundamentalResponse)
async def get_fundamental(
    symbol: str = Query(...),
    db: AsyncSession = Depends(get_db),
):
    """Get stock fundamental data."""
    # Try DB first
    result = await db.execute(
        select(StockFundamental)
        .where(StockFundamental.symbol == symbol)
        .order_by(StockFundamental.report_date.desc().nullslast())
        .limit(1)
    )
    cached = result.scalar_one_or_none()
    if cached:
        return FundamentalResponse(
            symbol=cached.symbol, name=cached.name,
            pe_ratio=cached.pe_ratio, pb_ratio=cached.pb_ratio,
            market_cap=cached.market_cap, industry=cached.industry, roe=cached.roe,
        )

    # Fetch from source
    manager = get_failover()
    try:
        fund = await manager.fetch_with_failover(DataType.FUNDAMENTAL, symbol)
        return FundamentalResponse(
            symbol=fund.symbol, name=fund.name,
            pe_ratio=fund.pe_ratio, pb_ratio=fund.pb_ratio,
            market_cap=fund.market_cap, industry=fund.industry, roe=fund.roe,
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Fundamental fetch failed: {e}")


@router.get("/health")
async def data_source_health():
    """Check health of all registered data sources."""
    manager = get_failover()
    return await manager.health_check_all()


# ============================================================================
# Market Index Data
# ============================================================================

# AkShare index code mapping
INDEX_CODES = {
    "sh000001": "上证指数",
    "sz399001": "深证成指",
    "sz399006": "创业板指",
}


class IndexResponse(BaseModel):
    symbol: str
    name: str
    price: float
    change_pct: float
    change_amount: float
    volume: Optional[int] = None
    high: Optional[float] = None
    low: Optional[float] = None


@router.get("/index", response_model=List[IndexResponse])
async def get_market_indices():
    """Get major market index data (上证/深证/创业板) via Tencent Finance.

    Tencent's qt.gtimg.cn is more reliable than East Money (no rate limit issues).
    """
    import asyncio
    import httpx

    INDEX_SYMBOLS = [
        ("sh000001", "上证指数"),
        ("sz399001", "深证成指"),
        ("sz399006", "创业板指"),
    ]

    async def _fetch_one(symbol: str) -> Optional[dict]:
        url = f"https://qt.gtimg.cn/q={symbol}"
        try:
            async with httpx.AsyncClient(
                timeout=6.0,
                headers={
                    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
                    "Referer": "https://gu.qq.com/",
                },
            ) as client:
                resp = await client.get(url)
                text = resp.text
            payload = text.split('="', 1)[1].rstrip('";\n')
            fields = payload.split("~")
            if len(fields) < 35:
                return None
            return {
                "name": fields[1], "symbol": fields[2],
                "price": float(fields[3]) if fields[3] else 0.0,
                "pre_close": float(fields[4]) if fields[4] else 0.0,
                "open": float(fields[5]) if fields[5] else 0.0,
                "volume": int(float(fields[6]) * 100) if fields[6] else 0,
                "change_amount": float(fields[31]) if fields[31] else 0.0,
                "change_pct": float(fields[32]) if fields[32] else 0.0,
                "high": float(fields[33]) if fields[33] else 0.0,
                "low": float(fields[34]) if fields[34] else 0.0,
            }
        except Exception as e:
            logger.debug(f"Tencent index fetch failed for {symbol}: {e}")
            return None

    rows = await asyncio.gather(*[_fetch_one(s) for s, _ in INDEX_SYMBOLS])

    results = []
    for (code, name), d in zip(INDEX_SYMBOLS, rows):
        if not d:
            results.append(IndexResponse(
                symbol=code, name=name,
                price=0.0, change_pct=0.0, change_amount=0.0,
            ))
            continue
        results.append(IndexResponse(
            symbol=code, name=name,
            price=d["price"], change_pct=d["change_pct"],
            change_amount=d["change_amount"],
            volume=d["volume"], high=d["high"], low=d["low"],
        ))

    # Fallback: try AkShare if all Tencent fetches returned empty
    if all(r.price == 0.0 for r in results):
        try:
            import akshare as ak
            loop = asyncio.get_event_loop()
            df = await asyncio.wait_for(
                loop.run_in_executor(None, lambda: ak.stock_zh_index_spot_sina()),
                timeout=10.0,
            )
            if df is not None and not df.empty:
                results.clear()
                for code, name in INDEX_CODES.items():
                    mask = df["代码"].astype(str).str.contains(code[2:]) if "代码" in df.columns else None
                    row = df[mask].iloc[0] if mask is not None and mask.any() else None
                    if row is not None:
                        results.append(IndexResponse(
                            symbol=code, name=name,
                            price=float(row.get("最新价", 0) or 0),
                            change_pct=float(row.get("涨跌幅", 0) or 0),
                            change_amount=float(row.get("涨跌额", 0) or 0),
                        ))
        except Exception as e:
            logger.warning(f"AkShare index fallback failed: {e}")

    return results


# -- Stock search cache (in-memory, refreshed every 1 hour) --
_stock_list_cache: Optional[List[dict]] = None
_stock_list_cache_time: Optional[float] = None
CACHE_TTL_SECONDS = 3600  # 1 hour


async def _get_stock_list() -> list:
    """Get full stock list, with in-memory caching."""
    global _stock_list_cache, _stock_list_cache_time
    import asyncio, time

    now = time.time()
    if _stock_list_cache is not None and _stock_list_cache_time is not None:
        if now - _stock_list_cache_time < CACHE_TTL_SECONDS:
            return _stock_list_cache

    import akshare as ak

    loop = asyncio.get_event_loop()
    df = None
    errors = []

    # Strategy 1: Use stock_info_a_code_name (static list)
    try:
        df = await loop.run_in_executor(None, lambda: ak.stock_info_a_code_name())
    except Exception as e:
        errors.append(f"code_name: {e}")

    # Strategy 2: Fallback to stock_zh_a_spot_em
    if df is None or df.empty:
        try:
            df = await asyncio.wait_for(
                loop.run_in_executor(None, lambda: ak.stock_zh_a_spot_em()),
                timeout=10.0,
            )
        except (asyncio.TimeoutError, Exception) as e:
            errors.append(f"spot_em: {e}")

    if df is None or df.empty:
        logger.warning(f"All stock list strategies failed: {'; '.join(errors)}")
        return []

    code_col = next((c for c in ["代码", "code", "symbol"] if c in df.columns), None)
    name_col = next((c for c in ["名称", "name", "证券简称"] if c in df.columns), None)

    if not code_col or not name_col:
        logger.warning(f"Unknown columns: {list(df.columns)[:10]}")
        return []

    results = []
    for _, row in df.iterrows():
        code = str(row[code_col])
        name = str(row[name_col])
        market = "SH" if code.startswith(("6", "9")) else "SZ"
        results.append({"symbol": code, "name": name, "market": market})

    _stock_list_cache = results
    _stock_list_cache_time = now
    logger.info(f"Stock list cached: {len(results)} stocks")
    return results


@router.get("/search")
async def search_stock(q: str = Query(..., min_length=1, description="Code or name")):
    """Search stocks by code or name. Uses cached stock list."""
    stock_list = await _get_stock_list()
    if not stock_list:
        return []

    q_lower = q.lower()
    matched = [
        s for s in stock_list
        if q_lower in s["symbol"].lower() or q_lower in s["name"].lower()
    ][:10]

    return matched
