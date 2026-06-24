# ============================================================================
# AI Stock Analysis Platform - Data Source Adapters
# ============================================================================
from __future__ import annotations

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Optional, List, Dict, Any, Tuple

import akshare as ak
import pandas as pd

logger = logging.getLogger(__name__)


def _normalize_date(raw: str) -> str:
    """Normalize date string to YYYY-MM-DD format."""
    if not raw:
        return ""
    s = str(raw).strip()
    # Already YYYY-MM-DD
    if len(s) == 10 and s[4] == "-":
        return s
    # YYYYMMDD -> YYYY-MM-DD
    if len(s) == 8 and s.isdigit():
        return f"{s[:4]}-{s[4:6]}-{s[6:8]}"
    # Try parsing various formats
    for fmt in ("%Y-%m-%d", "%Y%m%d", "%Y-%m-%d %H:%M:%S", "%Y/%m/%d"):
        try:
            from datetime import datetime as dt
            return dt.strptime(s, fmt).strftime("%Y-%m-%d")
        except (ValueError, TypeError):
            continue
    return s


# -- Data Types ---------------------------------------------------------------
class DataType(str, Enum):
    KLINE = "kline"
    REALTIME_QUOTE = "realtime_quote"
    FUNDAMENTAL = "fundamental"
    NEWS = "news"
    INDEX = "index"
    SECTOR = "sector"


class SourcePriority(int, Enum):
    P1 = 1
    P2 = 2
    P3 = 3
    P4 = 4
    P5 = 5
    P6 = 6


# -- Data Models --------------------------------------------------------------
@dataclass
class KlineData:
    symbol: str
    date: str
    open: float
    high: float
    low: float
    close: float
    volume: int
    amount: float = 0.0
    turnover_rate: float = 0.0


@dataclass
class RealtimeQuote:
    symbol: str
    name: str
    price: float
    change_pct: float
    volume: int
    high: float
    low: float
    open: float
    pre_close: float


@dataclass
class FundamentalData:
    symbol: str
    name: str
    pe_ratio: float = 0.0
    pb_ratio: float = 0.0
    market_cap: float = 0.0
    circulating_cap: float = 0.0
    industry: str = ""
    roe: float = 0.0


@dataclass
class NewsItem:
    source: str
    title: str
    content: str = ""
    url: str = ""
    published_at: Optional[datetime] = None


@dataclass
class SourceInfo:
    name: str
    priority: SourcePriority
    group: str  # cn_stock / global_stock
    reliability: str  # high / medium / low
    supported_types: List[DataType] = field(default_factory=list)


# -- Base Adapter -------------------------------------------------------------
class BaseAdapter(ABC):
    """Abstract base for all data source adapters."""

    def __init__(self, info: SourceInfo):
        self.info = info
        self._healthy = True
        self._last_error: Optional[str] = None

    @property
    def is_healthy(self) -> bool:
        return self._healthy

    @abstractmethod
    async def fetch_kline(
        self, symbol: str, frequency: str = "D",
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> List[KlineData]:
        """Fetch K-line data."""

    @abstractmethod
    async def fetch_realtime_quote(self, symbol: str) -> Optional[RealtimeQuote]:
        """Fetch real-time quote."""

    @abstractmethod
    async def fetch_fundamental(self, symbol: str) -> Optional[FundamentalData]:
        """Fetch fundamental data."""

    async def health_check(self) -> bool:
        """Check if source is healthy."""
        try:
            await self.fetch_realtime_quote("000001")
            self._healthy = True
            return True
        except Exception as e:
            self._healthy = False
            self._last_error = str(e)
            return False


# -- Concrete Adapters --------------------------------------------------------

class AkShareAdapter(BaseAdapter):
    """AkShare data source adapter - free, community-maintained.

    Uses stock_zh_a_hist (East Money) as primary, with fallback to
    stock_zh_a_daily (Sina/Tencent) when the primary source rate-limits.
    """

    @staticmethod
    def _to_ak_symbol(symbol: str) -> str:
        """Convert 6-digit symbol to akshare daily format (sz000001 / sh600519)."""
        s = symbol.replace(".SH", "").replace(".SZ", "").replace(".sh", "").replace(".sz", "")
        if s.startswith("6") or s.startswith("9"):
            return f"sh{s}"
        return f"sz{s}"

    async def fetch_kline(self, symbol: str, frequency: str = "D",
                          start_date: Optional[str] = None,
                          end_date: Optional[str] = None) -> List[KlineData]:
        max_retries = 2
        for attempt in range(max_retries):
            try:
                if not end_date:
                    end_date = datetime.now().strftime("%Y%m%d")
                if not start_date:
                    start_date = (datetime.now() - timedelta(days=90)).strftime("%Y%m%d")

                loop = asyncio.get_event_loop()

                # Primary: East Money (stock_zh_a_hist)
                try:
                    df = await loop.run_in_executor(
                        None,
                        lambda: ak.stock_zh_a_hist(
                            symbol=symbol, period="daily",
                            start_date=start_date, end_date=end_date, adjust="qfq"
                        )
                    )
                    if df is not None and not df.empty:
                        return self._parse_hist_df(symbol, df)
                except Exception as primary_err:
                    err_str = str(primary_err)
                    is_rate_limit = (
                        "RemoteDisconnected" in err_str
                        or "Connection aborted" in err_str
                        or "429" in err_str
                        or "Too Many Requests" in err_str
                    )
                    if is_rate_limit:
                        logger.warning(
                            f"AkShare primary (East Money) failed for {symbol}, "
                            f"falling back to Sina/Tencent: {primary_err}"
                        )
                    else:
                        raise  # Non-retryable, propagate up

                # Fallback: Sina/Tencent (stock_zh_a_daily)
                ak_symbol = self._to_ak_symbol(symbol)
                df = await loop.run_in_executor(
                    None,
                    lambda: ak.stock_zh_a_daily(
                        symbol=ak_symbol, start_date=start_date, end_date=end_date, adjust="qfq"
                    )
                )

                if df is None or df.empty:
                    return []

                return self._parse_daily_df(symbol, df)

            except Exception as e:
                err_str = str(e)
                if attempt < max_retries - 1 and (
                    "RemoteDisconnected" in err_str
                    or "Connection aborted" in err_str
                    or "ConnectionResetError" in err_str
                ):
                    wait = 2.0 * (attempt + 1)
                    logger.warning(
                        f"AkShare fetch_kline attempt {attempt+1}/{max_retries} failed for {symbol}, "
                        f"retrying in {wait}s: {e}"
                    )
                    await asyncio.sleep(wait)
                    continue
                logger.warning(f"AkShare fetch_kline failed for {symbol}: {e}")
                raise
        return []

    @staticmethod
    def _parse_hist_df(symbol: str, df) -> List[KlineData]:
        """Parse stock_zh_a_hist DataFrame (East Money columns)."""
        results = []
        for _, row in df.iterrows():
            results.append(KlineData(
                symbol=symbol,
                date=_normalize_date(row.get("日期", "")),
                open=float(row.get("开盘", 0)),
                high=float(row.get("最高", 0)),
                low=float(row.get("最低", 0)),
                close=float(row.get("收盘", 0)),
                volume=int(row.get("成交量", 0)),
                amount=float(row.get("成交额", 0)),
                turnover_rate=float(row.get("换手率", 0)) if row.get("换手率") else 0,
            ))
        return results

    @staticmethod
    def _parse_daily_df(symbol: str, df) -> List[KlineData]:
        """Parse stock_zh_a_daily DataFrame (Sina/Tencent columns)."""
        results = []
        for _, row in df.iterrows():
            results.append(KlineData(
                symbol=symbol,
                date=_normalize_date(str(row.get("date", ""))),
                open=float(row.get("open", 0)),
                high=float(row.get("high", 0)),
                low=float(row.get("low", 0)),
                close=float(row.get("close", 0)),
                volume=int(row.get("volume", 0)),
                amount=float(row.get("amount", 0)),
                turnover_rate=float(row.get("turnover", 0)) if row.get("turnover") else 0,
            ))
        return results

    async def fetch_realtime_quote(self, symbol: str) -> Optional[RealtimeQuote]:
        try:
            loop = asyncio.get_event_loop()
            df = await loop.run_in_executor(
                None, lambda: ak.stock_zh_a_spot_em()
            )
            if df is None or df.empty:
                return None

            row = df[df["代码"] == symbol]
            if row.empty:
                # Try with leading zeros
                padded = symbol.zfill(6)
                row = df[df["代码"] == padded]
            if row.empty:
                return None

            r = row.iloc[0]
            return RealtimeQuote(
                symbol=symbol,
                name=str(r.get("名称", "")),
                price=float(r.get("最新价", 0)),
                change_pct=float(r.get("涨跌幅", 0)),
                volume=int(r.get("成交量", 0)),
                high=float(r.get("最高", 0)),
                low=float(r.get("最低", 0)),
                open=float(r.get("今开", 0)),
                pre_close=float(r.get("昨收", 0)),
            )
        except Exception as e:
            logger.warning(f"AkShare fetch_realtime_quote failed for {symbol}: {e}")
            raise

    async def fetch_fundamental(self, symbol: str) -> Optional[FundamentalData]:
        try:
            loop = asyncio.get_event_loop()
            df = await loop.run_in_executor(
                None, lambda: ak.stock_individual_info_em(symbol=symbol)
            )
            if df is None or df.empty:
                return None

            data = dict(zip(df["item"], df["value"]))
            return FundamentalData(
                symbol=symbol,
                name=data.get("股票简称", ""),
                pe_ratio=float(data.get("市盈率-动态", 0) or 0),
                pb_ratio=float(data.get("市净率", 0) or 0),
                market_cap=float(data.get("总市值", 0) or 0) / 1e8,
                circulating_cap=float(data.get("流通市值", 0) or 0) / 1e8,
                industry=data.get("行业", ""),
                roe=float(data.get("净资产收益率", 0) or 0),
            )
        except Exception as e:
            logger.warning(f"AkShare fetch_fundamental failed for {symbol}: {e}")
            raise


class TushareAdapter(BaseAdapter):
    """Tushare data source adapter - requires API token."""

    def __init__(self, info: SourceInfo, token: Optional[str] = None):
        super().__init__(info)
        self.token = token
        self._pro = None

    async def _get_pro(self):
        if self._pro is None:
            import tushare as ts
            token = self.token or ""
            loop = asyncio.get_event_loop()
            self._pro = await loop.run_in_executor(None, lambda: ts.pro_api(token))
        return self._pro

    async def fetch_kline(self, symbol: str, frequency: str = "D",
                          start_date: Optional[str] = None,
                          end_date: Optional[str] = None) -> List[KlineData]:
        try:
            pro = await self._get_pro()
            if not end_date:
                end_date = datetime.now().strftime("%Y%m%d")
            if not start_date:
                start_date = (datetime.now() - timedelta(days=90)).strftime("%Y%m%d")

            loop = asyncio.get_event_loop()
            df = await loop.run_in_executor(
                None,
                lambda: pro.daily(
                    ts_code=f"{symbol}.SZ" if symbol.startswith("0") or symbol.startswith("3")
                    else f"{symbol}.SH",
                    start_date=start_date, end_date=end_date
                )
            )
            if df is None or df.empty:
                return []

            results = []
            for _, row in df.iterrows():
                results.append(KlineData(
                    symbol=symbol,
                    date=_normalize_date(row.get("trade_date", "")),
                    open=float(row.get("open", 0)),
                    high=float(row.get("high", 0)),
                    low=float(row.get("low", 0)),
                    close=float(row.get("close", 0)),
                    volume=int(row.get("vol", 0)),
                    amount=float(row.get("amount", 0)),
                ))
            return results
        except Exception as e:
            logger.warning(f"Tushare fetch_kline failed for {symbol}: {e}")
            raise

    async def fetch_realtime_quote(self, symbol: str) -> Optional[RealtimeQuote]:
        raise NotImplementedError("Tushare does not support real-time quotes")

    async def fetch_fundamental(self, symbol: str) -> Optional[FundamentalData]:
        try:
            pro = await self._get_pro()
            ts_code = (f"{symbol}.SZ" if symbol.startswith("0") or symbol.startswith("3")
                       else f"{symbol}.SH")

            loop = asyncio.get_event_loop()
            df = await loop.run_in_executor(
                None, lambda: pro.daily_basic(ts_code=ts_code)
            )
            if df is None or df.empty:
                return None

            row = df.iloc[0]
            return FundamentalData(
                symbol=symbol,
                name="",
                pe_ratio=float(row.get("pe", 0) or 0),
                pb_ratio=float(row.get("pb", 0) or 0),
                market_cap=float(row.get("total_mv", 0) or 0) / 1e4,
            )
        except Exception as e:
            logger.warning(f"Tushare fetch_fundamental failed for {symbol}: {e}")
            raise


class YahooFinanceAdapter(BaseAdapter):
    """Yahoo Finance adapter - good for international stocks."""

    async def fetch_kline(self, symbol: str, frequency: str = "D",
                          start_date: Optional[str] = None,
                          end_date: Optional[str] = None) -> List[KlineData]:
        try:
            import yfinance as yf

            yf_symbol = symbol
            if symbol.isdigit() and len(symbol) == 6:
                if symbol.startswith("6"):
                    yf_symbol = f"{symbol}.SS"
                else:
                    yf_symbol = f"{symbol}.SZ"

            loop = asyncio.get_event_loop()
            ticker = await loop.run_in_executor(None, lambda: yf.Ticker(yf_symbol))

            if not end_date:
                end_date = datetime.now().strftime("%Y-%m-%d")
            if not start_date:
                start_date = (datetime.now() - timedelta(days=90)).strftime("%Y-%m-%d")

            df = await loop.run_in_executor(
                None,
                lambda: ticker.history(start=start_date, end=end_date, interval="1d")
            )
            if df is None or df.empty:
                return []

            results = []
            for idx, row in df.iterrows():
                results.append(KlineData(
                    symbol=symbol,
                    date=idx.strftime("%Y-%m-%d"),
                    open=float(row["Open"]),
                    high=float(row["High"]),
                    low=float(row["Low"]),
                    close=float(row["Close"]),
                    volume=int(row["Volume"]),
                ))
            return results
        except Exception as e:
            logger.warning(f"Yahoo Finance fetch_kline failed for {symbol}: {e}")
            raise

    async def fetch_realtime_quote(self, symbol: str) -> Optional[RealtimeQuote]:
        try:
            import yfinance as yf
            yf_symbol = symbol
            if symbol.isdigit() and len(symbol) == 6:
                if symbol.startswith("6"):
                    yf_symbol = f"{symbol}.SS"
                else:
                    yf_symbol = f"{symbol}.SZ"

            loop = asyncio.get_event_loop()
            ticker = await loop.run_in_executor(None, lambda: yf.Ticker(yf_symbol))
            info = await loop.run_in_executor(None, lambda: ticker.info)

            return RealtimeQuote(
                symbol=symbol,
                name=info.get("longName", info.get("shortName", "")),
                price=float(info.get("currentPrice", info.get("regularMarketPrice", 0))),
                change_pct=float(info.get("regularMarketChangePercent", 0)),
                volume=int(info.get("volume", 0)),
                high=float(info.get("dayHigh", 0)),
                low=float(info.get("dayLow", 0)),
                open=float(info.get("regularMarketOpen", 0)),
                pre_close=float(info.get("previousClose", 0)),
            )
        except Exception as e:
            logger.warning(f"Yahoo Finance fetch_realtime_quote failed for {symbol}: {e}")
            raise

    async def fetch_fundamental(self, symbol: str) -> Optional[FundamentalData]:
        try:
            import yfinance as yf
            yf_symbol = symbol
            if symbol.isdigit() and len(symbol) == 6:
                if symbol.startswith("6"):
                    yf_symbol = f"{symbol}.SS"
                else:
                    yf_symbol = f"{symbol}.SZ"

            loop = asyncio.get_event_loop()
            ticker = await loop.run_in_executor(None, lambda: yf.Ticker(yf_symbol))
            info = await loop.run_in_executor(None, lambda: ticker.info)

            return FundamentalData(
                symbol=symbol,
                name=info.get("longName", ""),
                pe_ratio=float(info.get("trailingPE", 0) or 0),
                pb_ratio=float(info.get("priceToBook", 0) or 0),
                market_cap=float(info.get("marketCap", 0) or 0) / 1e8,
                industry=info.get("industry", ""),
                roe=float(info.get("returnOnEquity", 0) or 0) * 100,
            )
        except Exception as e:
            logger.warning(f"Yahoo Finance fetch_fundamental failed for {symbol}: {e}")
            raise


# -- Failover Manager ---------------------------------------------------------
class FailoverManager:
    """
    Manages data source failover with priority-based cascading.
    Automatically switches to next available source on failure.
    """

    def __init__(self):
        self.adapters: Dict[str, BaseAdapter] = {}
        self._source_order: List[Tuple[str, SourcePriority]] = []
        self._health_cache: Dict[str, Tuple[bool, float]] = {}
        # In-memory TTL cache: {(data_type, symbol, kwargs_key): (timestamp, data)}
        self._data_cache: Dict[Tuple, Tuple[float, Any]] = {}
        self._cache_ttl: float = 60.0  # seconds
        # Per-source rate limit tracking: {source_name: last_call_timestamp}
        self._last_call: Dict[str, float] = {}
        self._min_interval: float = 0.5  # min seconds between calls to same source

    def register(self, adapter: BaseAdapter):
        self.adapters[adapter.info.name] = adapter
        self._source_order.append((adapter.info.name, adapter.info.priority))
        self._source_order.sort(key=lambda x: x[1])

    async def _respect_rate_limit(self, source_name: str):
        """Ensure minimum interval between calls to the same source."""
        last = self._last_call.get(source_name, 0)
        elapsed = time.time() - last
        if elapsed < self._min_interval:
            await asyncio.sleep(self._min_interval - elapsed)
        self._last_call[source_name] = time.time()

    async def fetch_with_failover(self, data_type: DataType, symbol: str,
                                   **kwargs) -> Any:
        """Fetch data with automatic failover across sources.

        Includes an in-memory TTL cache (60s) to avoid repeated calls
        for the same symbol within a single analysis pipeline, and
        per-source rate limiting to prevent triggering 429 errors.
        """
        # Check cache first
        kwargs_key = tuple(sorted(kwargs.items()))
        cache_key = (data_type, symbol, kwargs_key)
        now = time.time()
        cached = self._data_cache.get(cache_key)
        if cached and (now - cached[0]) < self._cache_ttl:
            logger.debug(f"Cache hit for {data_type}:{symbol} (age={now-cached[0]:.1f}s)")
            return cached[1]

        last_error = None

        for name, priority in self._source_order:
            adapter = self.adapters.get(name)
            if not adapter:
                continue

            await self._respect_rate_limit(name)

            try:
                if data_type == DataType.KLINE:
                    result = await asyncio.wait_for(
                        adapter.fetch_kline(symbol, **kwargs),
                        timeout=15.0
                    )
                elif data_type == DataType.REALTIME_QUOTE:
                    result = await asyncio.wait_for(
                        adapter.fetch_realtime_quote(symbol),
                        timeout=10.0
                    )
                elif data_type == DataType.FUNDAMENTAL:
                    result = await asyncio.wait_for(
                        adapter.fetch_fundamental(symbol),
                        timeout=15.0
                    )
                else:
                    raise ValueError(f"Unsupported data type: {data_type}")

                if result is not None:
                    logger.debug(f"Data fetched from {name} (P{priority})")
                    # Write to cache
                    self._data_cache[cache_key] = (time.time(), result)
                    return result

            except asyncio.TimeoutError:
                logger.warning(f"Source {name} timed out for {data_type}:{symbol}")
                last_error = f"Timeout: {name}"
            except Exception as e:
                err_str = str(e)
                # Detect rate limiting (429 / Too Many Requests) and back off
                if "429" in err_str or "Too Many Requests" in err_str or "Rate limited" in err_str:
                    logger.warning(
                        f"Source {name} rate-limited for {data_type}:{symbol}, "
                        f"backing off 3s before next source"
                    )
                    await asyncio.sleep(3.0)
                logger.warning(f"Source {name} failed for {data_type}:{symbol}: {e}")
                last_error = err_str
                continue

        raise RuntimeError(
            f"All sources failed for {data_type}:{symbol}. "
            f"Last error: {last_error}"
        )

    async def health_check_all(self) -> Dict[str, bool]:
        """Check health of all registered sources."""
        results = {}
        for name, adapter in self.adapters.items():
            results[name] = await adapter.health_check()
        return results


# -- Factory ------------------------------------------------------------------
def create_default_failover_manager(
    tushare_token: Optional[str] = None
) -> FailoverManager:
    """Create failover manager with all default adapters registered."""

    manager = FailoverManager()

    # P1: AkShare (most reliable for Chinese A-shares)
    manager.register(AkShareAdapter(SourceInfo(
        name="AkShare",
        priority=SourcePriority.P1,
        group="cn_stock",
        reliability="medium",
        supported_types=[DataType.KLINE, DataType.REALTIME_QUOTE, DataType.FUNDAMENTAL, DataType.NEWS],
    )))

    # P2: Tushare
    manager.register(TushareAdapter(SourceInfo(
        name="Tushare",
        priority=SourcePriority.P2,
        group="cn_stock",
        reliability="medium",
        supported_types=[DataType.KLINE, DataType.FUNDAMENTAL],
    ), token=tushare_token))

    # P3: Yahoo Finance (fallback)
    manager.register(YahooFinanceAdapter(SourceInfo(
        name="Yahoo Finance",
        priority=SourcePriority.P3,
        group="global_stock",
        reliability="medium",
        supported_types=[DataType.KLINE, DataType.REALTIME_QUOTE, DataType.FUNDAMENTAL],
    )))

    return manager
