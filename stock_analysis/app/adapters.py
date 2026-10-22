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

    async def health_check(self) -> bool:
        """Cheap liveness probe.

        The inherited implementation pulls the *whole* A-share spot table
        (59 paged requests), which takes ~60s and would block the UI's
        data-source indicator. A single index daily bar is enough to prove
        the source is reachable.
        """
        try:
            loop = asyncio.get_event_loop()
            # 指数日线走新浪源（~0.3s），比东财行情表稳定得多
            df = await loop.run_in_executor(
                None, lambda: ak.stock_zh_index_daily(symbol="sh000300")
            )
            ok = df is not None and not df.empty
            self._healthy = bool(ok)
            return self._healthy
        except Exception as e:
            self._healthy = False
            self._last_error = str(e)
            return False

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


# -- Yahoo Finance (pure HTTP, no yfinance) -----------------------------------
# 说明：yfinance 获取 cookie/crumb 依赖 mini_racer(V8) 原生调用，一旦在 V8 里
# 死循环会持有 GIL，从而冻结整个 asyncio 事件循环（表现为全站接口无响应）。
# 这里改为直接调用 Yahoo 的公开 HTTP 接口，不引入任何 JS 引擎。
YAHOO_CHART_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
YAHOO_CRUMB_URL = "https://query2.finance.yahoo.com/v1/test/getcrumb"
YAHOO_SUMMARY_URL = "https://query1.finance.yahoo.com/v10/finance/quoteSummary/{symbol}"
YAHOO_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
    "Accept": "application/json",
}


def _yahoo_symbol(symbol: str) -> str:
    """'600519' -> '600519.SS' / '000001' -> '000001.SZ'；其他原样返回。"""
    sym = str(symbol or "").strip()
    if sym.isdigit() and len(sym) == 6:
        return f"{sym}.SS" if sym.startswith("6") else f"{sym}.SZ"
    return sym


def _yahoo_ts(value: Optional[str]) -> Optional[int]:
    if not value:
        return None
    try:
        return int(datetime.fromisoformat(str(value)[:10]).timestamp())
    except (ValueError, TypeError):
        return None


def _yahoo_rows(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    """把 chart 接口的列式数组转成行式 K 线。"""
    try:
        result = (payload.get("chart") or {}).get("result") or []
        if not result:
            return []
        r = result[0]
        ts = r.get("timestamp") or []
        quote = ((r.get("indicators") or {}).get("quote") or [{}])[0]
        o, h, l, c, v = (quote.get(k) or [] for k in ("open", "high", "low", "close", "volume"))
        rows = []
        for i, t in enumerate(ts):
            def _g(seq, idx):
                try:
                    val = seq[idx]
                    return float(val) if val is not None else 0.0
                except (IndexError, TypeError, ValueError):
                    return 0.0
            rows.append({
                "date": datetime.utcfromtimestamp(t).strftime("%Y-%m-%d"),
                "open": _g(o, i), "high": _g(h, i), "low": _g(l, i),
                "close": _g(c, i), "volume": _g(v, i),
            })
        return rows
    except Exception as exc:
        logger.debug(f"yahoo rows parse failed: {exc}")
        return []


class YahooFinanceAdapter(BaseAdapter):
    """Yahoo Finance adapter - good for international stocks (pure HTTP, no JS engine)."""

    async def fetch_kline(self, symbol: str, frequency: str = "D",
                          start_date: Optional[str] = None,
                          end_date: Optional[str] = None) -> List[KlineData]:
        try:
            import httpx

            yf_symbol = _yahoo_symbol(symbol)

            if not end_date:
                end_date = datetime.now().strftime("%Y-%m-%d")
            if not start_date:
                start_date = (datetime.now() - timedelta(days=90)).strftime("%Y-%m-%d")
            p1, p2 = _yahoo_ts(start_date), _yahoo_ts(end_date)
            interval = {"D": "1d", "W": "1wk", "M": "1mo"}.get(frequency, "1d")
            params = {"interval": interval}
            if p1 and p2:
                params.update({"period1": p1, "period2": p2 + 86400})

            async with httpx.AsyncClient(timeout=10.0, headers=YAHOO_HEADERS) as client:
                resp = await client.get(YAHOO_CHART_URL.format(symbol=yf_symbol), params=params)
                resp.raise_for_status()
                payload = resp.json()
            rows = _yahoo_rows(payload)
            if not rows:
                return []
            return [
                KlineData(symbol=symbol, date=r["date"], open=r["open"], high=r["high"],
                          low=r["low"], close=r["close"], volume=r["volume"],
                          amount=r["close"] * r["volume"])
                for r in rows
            ]
        except Exception as e:
            logger.warning(f"Yahoo Finance fetch_kline failed for {symbol}: {e}")
            raise

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
            import httpx

            yf_symbol = _yahoo_symbol(symbol)
            async with httpx.AsyncClient(timeout=10.0, headers=YAHOO_HEADERS) as client:
                resp = await client.get(
                    YAHOO_CHART_URL.format(symbol=yf_symbol),
                    params={"interval": "1d", "range": "5d"},
                )
                resp.raise_for_status()
                payload = resp.json()
            results = (payload.get("chart") or {}).get("result") or []
            if not results:
                return None
            meta = results[0].get("meta") or {}
            price = float(meta.get("regularMarketPrice") or 0)
            pre_close = float(meta.get("chartPreviousClose") or meta.get("previousClose") or 0)
            if not price:
                rows = _yahoo_rows(payload)
                price = rows[-1]["close"] if rows else 0.0
                pre_close = rows[-2]["close"] if len(rows) > 1 else pre_close
            change_pct = ((price / pre_close) - 1) * 100 if pre_close else 0.0
            return RealtimeQuote(
                symbol=symbol,
                name=meta.get("shortName") or meta.get("longName") or symbol,
                price=price,
                change_pct=change_pct,
                volume=int(meta.get("regularMarketVolume") or 0),
                high=float(meta.get("regularMarketDayHigh") or 0),
                low=float(meta.get("regularMarketDayLow") or 0),
                open=float(meta.get("regularMarketOpen") or 0),
                pre_close=pre_close,
            )
        except Exception as e:
            logger.warning(f"Yahoo Finance fetch_realtime_quote failed for {symbol}: {e}")
            raise

    async def health_check(self) -> bool:
        """Liveness probe without any JS engine (yfinance would risk a GIL freeze)."""
        try:
            import httpx

            async with httpx.AsyncClient(timeout=8.0, headers=YAHOO_HEADERS) as client:
                resp = await client.get(
                    YAHOO_CHART_URL.format(symbol="AAPL"),
                    params={"interval": "1d", "range": "1d"},
                )
                ok = resp.status_code == 200 and bool(
                    (resp.json().get("chart") or {}).get("result")
                )
            self._healthy = bool(ok)
            return self._healthy
        except Exception as e:
            self._healthy = False
            self._last_error = str(e)
            return False

    async def fetch_fundamental(self, symbol: str) -> Optional[FundamentalData]:
        try:
            import httpx

            yf_symbol = _yahoo_symbol(symbol)
            async with httpx.AsyncClient(timeout=10.0, headers=YAHOO_HEADERS) as client:
                # quoteSummary 需要 cookie + crumb，两者都可用纯 HTTP 获取（无 JS）
                try:
                    await client.get("https://fc.yahoo.com", timeout=6.0)
                    crumb_resp = await client.get(YAHOO_CRUMB_URL, timeout=6.0)
                    crumb = crumb_resp.text.strip() if crumb_resp.status_code == 200 else ""
                except Exception:
                    crumb = ""
                params = {"modules": "summaryDetail,defaultKeyStatistics,price"}
                if crumb and len(crumb) < 40:
                    params["crumb"] = crumb
                resp = await client.get(
                    YAHOO_SUMMARY_URL.format(symbol=yf_symbol), params=params
                )
                resp.raise_for_status()
                payload = resp.json()
            result = ((payload.get("quoteSummary") or {}).get("result") or [{}])[0]
            if not result:
                return None
            detail = result.get("summaryDetail") or {}
            stats = result.get("defaultKeyStatistics") or {}
            price = result.get("price") or {}

            def _num(container, key, default=0.0):
                val = container.get(key)
                if isinstance(val, dict):
                    val = val.get("raw")
                try:
                    return float(val) if val is not None else default
                except (TypeError, ValueError):
                    return default

            return FundamentalData(
                symbol=symbol,
                name=price.get("longName") or price.get("shortName") or symbol,
                pe_ratio=_num(detail, "trailingPE"),
                pb_ratio=_num(stats, "priceToBook"),
                market_cap=_num(price, "marketCap") / 1e8,
                industry=(stats.get("sector") or ""),
                roe=_num(stats, "returnOnEquity") * 100,
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

    async def health_check_all(self, timeout: float = 8.0) -> Dict[str, bool]:
        """Check health of all registered sources (in parallel, bounded)."""
        async def _one(name: str, adapter):
            try:
                return name, bool(await asyncio.wait_for(adapter.health_check(), timeout=timeout))
            except Exception:
                return name, False

        results: Dict[str, bool] = {}
        pairs = await asyncio.gather(*(_one(n, a) for n, a in self.adapters.items()))
        for name, ok in pairs:
            results[name] = ok
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
