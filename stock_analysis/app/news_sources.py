# ============================================================================
# AI Stock Analysis Platform - Multi-source Stock News Aggregation
# ============================================================================
"""多平台股票/财经新闻聚合。

来源（可失败自动跳过，互不影响）：

- 同花顺   ``ak.stock_info_global_ths()``      （20 条/次）
- 东方财富 ``ak.stock_info_global_em()``       （200 条/次）
- 财联社   ``ak.stock_info_global_cls()``      （20 条/次）
- 新浪财经 ``ak.stock_info_global_sina()``     （20 条/次）
- 腾讯财经 腾讯财经 HTTP 接口（best-effort，接口契约变动时会自动跳过）
- 雅虎财经 Yahoo Finance search（best-effort，境外网络/限流时自动跳过）

设计要点：

1. akshare 是同步阻塞库，统一用 ``asyncio.to_thread`` 执行，避免卡死事件循环；
2. 抓取后做**跨源去重**（规范化标题 + URL）；
3. 采用**按源轮询取样**，保证 50 条里能看到多个平台，而不是被东方财富 200 条淹没；
4. 入库前再与数据库近 N 小时已有新闻去重，避免每 15 分钟重复插入。
"""
from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import httpx

logger = logging.getLogger(__name__)

# 单次聚合上限（需求：最多 50 条）
MAX_TOTAL = 50
# 中国标准时间（数据源时间为北京时间，库里沿用 naive 本地时间以与既有数据/前端一致）
CN_TZ = timezone(timedelta(hours=8))
# 入库去重窗口（小时）
DEDUP_WINDOW_HOURS = 72

_HTTP_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36"
    ),
    "Referer": "https://gu.qq.com/",
}

# 需要清洗掉再比对的标点/空白，用于标题去重
_TITLE_NOISE = re.compile(r"[\s\u3000【】\[\]（）()「」“”\"'!?！？,，。.、:：;；\-—_/|]+")


@dataclass
class RawNews:
    """标准化后的单条新闻（尚未入库）。"""

    source: str
    title: str
    content: str = ""
    url: str = ""
    published_at: Optional[datetime] = None
    # 打标信息（个股新闻直接带上 related_stocks；通用新闻由关键词兜底生成）
    related_stocks: Optional[List[dict]] = None
    topic_tags: Optional[List[str]] = None
    sentiment: Optional[str] = None
    tag_source: Optional[str] = None

    def norm_title(self) -> str:
        return _TITLE_NOISE.sub("", self.title or "").lower()


# -- 时间解析 ----------------------------------------------------------------
def _now_utc() -> datetime:
    """当前时间（naive UTC，与模型 DateTime 列 / now_utc() 保持一致）。"""
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _to_utc(dt: datetime) -> datetime:
    """统一成 naive UTC：无时区按北京时间(UTC+8)解释，再换算到 UTC 并去掉 tzinfo。"""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=CN_TZ)
    return dt.astimezone(timezone.utc).replace(tzinfo=None)


def _parse_dt(value: Any, date_part: Any = None) -> datetime:
    """把数据源返回的（北京时间）时间解析成 naive UTC datetime。"""
    if value is None:
        return _now_utc()

    # 若传入的是 date 列 + time 列（财联社）
    raw = str(value).strip()
    if date_part is not None:
        date_str = str(date_part).strip()
        if date_str and date_str not in raw:
            raw = f"{date_str} {raw}".strip()

    if isinstance(value, datetime):
        return _to_utc(value)

    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d", "%Y/%m/%d %H:%M:%S"):
        try:
            return _to_utc(datetime.strptime(raw, fmt))
        except (ValueError, TypeError):
            continue

    # 兜底：截取匹配部分
    m = re.search(r"(\d{4}-\d{2}-\d{2})[ T]?(\d{2}:\d{2}(?::\d{2})?)?", raw)
    if m:
        candidate = f"{m.group(1)} {m.group(2) or '00:00:00'}"
        try:
            return _to_utc(datetime.strptime(candidate, "%Y-%m-%d %H:%M:%S"))
        except ValueError:
            pass
    return _now_utc()


# -- akshare 数据帧解析 ------------------------------------------------------
def _cell(row: Dict[str, Any], *keys: str) -> str:
    for key in keys:
        if key in row and row[key] is not None:
            val = str(row[key]).strip()
            if val and val.lower() != "nan":
                return val
    return ""


def _parse_akshare_df(
    df: Any,
    source: str,
    limit: int,
    *,
    title_keys=("标题", "title"),
    content_keys=("摘要", "内容", "content"),
    time_keys=("发布时间", "时间", "datetime"),
    date_keys: tuple = (),
    url_keys=("链接", "url"),
) -> List[RawNews]:
    if df is None or getattr(df, "empty", True):
        return []

    items: List[RawNews] = []
    for _, row in df.head(max(limit, 1) * 2).iterrows():
        record = row.to_dict()
        title = _cell(record, *title_keys)
        content = _cell(record, *content_keys)

        # 新浪等只有正文：从正文首句裁剪出标题
        if not title and content:
            title = re.split(r"[。！？!?\n]", content, maxsplit=1)[0][:80]
        if not title:
            continue

        date_part = _cell(record, *date_keys) if date_keys else None
        published_at = _parse_dt(_cell(record, *time_keys) or None, date_part)

        items.append(
            RawNews(
                source=source,
                title=title[:512],
                content=content[:4000],
                url=_cell(record, *url_keys)[:1024],
                published_at=published_at,
            )
        )
        if len(items) >= limit:
            break
    return items


# -- 各数据源抓取 ------------------------------------------------------------
async def _fetch_akshare(
    fn_name: str, source: str, limit: int, **parse_kwargs
) -> List[RawNews]:
    try:
        import akshare as ak
    except Exception as exc:  # pragma: no cover
        logger.warning(f"[news] akshare 不可用: {exc}")
        return []

    fn = getattr(ak, fn_name, None)
    if fn is None:
        logger.warning(f"[news] akshare 缺少接口 {fn_name}")
        return []

    try:
        df = await asyncio.to_thread(fn)
    except Exception as exc:
        logger.warning(f"[news] {source} 抓取失败: {type(exc).__name__}: {exc}")
        return []

    try:
        items = _parse_akshare_df(df, source, limit, **parse_kwargs)
        logger.info(f"[news] {source} 抓取 {len(items)} 条")
        return items
    except Exception as exc:
        logger.warning(f"[news] {source} 解析失败: {type(exc).__name__}: {exc}")
        return []


async def fetch_eastmoney(limit: int) -> List[RawNews]:
    """东方财富 - 全球财经快讯（200 条/次）。"""
    return await _fetch_akshare(
        "stock_info_global_em", "东方财富", limit,
        title_keys=("标题",), content_keys=("摘要", "内容"), time_keys=("发布时间",), url_keys=("链接",),
    )


async def fetch_ths(limit: int) -> List[RawNews]:
    """同花顺 - 全球财经快讯（20 条/次）。"""
    return await _fetch_akshare(
        "stock_info_global_ths", "同花顺", limit,
        title_keys=("标题",), content_keys=("内容", "摘要"), time_keys=("发布时间",), url_keys=("链接",),
    )


async def fetch_cls(limit: int) -> List[RawNews]:
    """财联社 - 电报快讯（20 条/次，日期与时间分列，需拼接）。"""
    return await _fetch_akshare(
        "stock_info_global_cls", "财联社", limit,
        title_keys=("标题",), content_keys=("内容",), time_keys=("发布时间",),
        date_keys=("发布日期",), url_keys=("链接",),
    )


async def fetch_sina(limit: int) -> List[RawNews]:
    """新浪财经 - 全球快讯（20 条/次，仅正文+时间，标题由正文裁剪）。"""
    return await _fetch_akshare(
        "stock_info_global_sina", "新浪财经", limit,
        title_keys=("标题",), content_keys=("内容",), time_keys=("时间",), url_keys=("链接",),
    )


async def fetch_tencent(limit: int, symbols: Optional[List[str]] = None) -> List[RawNews]:
    """腾讯财经 - 个股新闻（best-effort）。

    腾讯的私有搜索接口对参数/签名较敏感，且历史上多次变更。这里做多参数尝试，
    全部失败时返回空列表，不影响其它源。
    """
    symbols = symbols or ["sh600519", "sz000001", "sh601318"]
    per_symbol = max(1, limit // len(symbols) + 1)
    out: List[RawNews] = []

    async with httpx.AsyncClient(timeout=10.0, headers=_HTTP_HEADERS, follow_redirects=True) as client:
        for code in symbols:
            for path in (
                f"https://proxy.finance.qq.com/ifzqgtimg/appstock/news/info/search"
                f"?code={code}&page=0&limit={per_symbol}&type=1",
            ):
                try:
                    resp = await client.get(path)
                    payload = resp.json()
                except Exception as exc:
                    logger.debug(f"[news] 腾讯 {code} 请求异常: {exc}")
                    continue

                if not isinstance(payload, dict) or payload.get("code") != 0:
                    logger.debug(f"[news] 腾讯 {code} 返回异常: {payload}")
                    continue

                rows = payload.get("data") or []
                if isinstance(rows, dict):
                    rows = rows.get("list") or rows.get("data") or []
                for row in rows:
                    if not isinstance(row, dict):
                        continue
                    title = _cell(row, "title", "Title", "shortTitle")
                    if not title:
                        continue
                    ts = row.get("time") or row.get("publish_time") or row.get("ctime")
                    pub = _parse_dt(ts)
                    out.append(
                        RawNews(
                            source="腾讯财经",
                            title=title[:512],
                            content=_cell(row, "content", "digest", "summary")[:4000],
                            url=_cell(row, "url", "link")[:1024],
                            published_at=pub,
                        )
                    )
            if len(out) >= limit:
                break

    if out:
        logger.info(f"[news] 腾讯财经 抓取 {len(out[:limit])} 条")
    else:
        logger.info("[news] 腾讯财经 本次无可用数据（接口参数变更或限流），已跳过")
    return out[:limit]


async def fetch_yahoo(limit: int, query: str = "stock market") -> List[RawNews]:
    """雅虎财经 - 资讯搜索（best-effort）。境外网络受限或 429 时返回空列表。"""
    url = "https://query1.finance.yahoo.com/v1/finance/search"
    params = {
        "q": query,
        "newsCount": min(limit, 50),
        "quotesCount": 0,
        "enableFuzzyQuery": "false",
    }
    try:
        async with httpx.AsyncClient(timeout=10.0, headers=_HTTP_HEADERS, follow_redirects=True) as client:
            resp = await client.get(url, params=params)
            if resp.status_code != 200:
                logger.info(f"[news] 雅虎财经 返回 {resp.status_code}，已跳过")
                return []
            payload = resp.json()
    except Exception as exc:
        logger.info(f"[news] 雅虎财经 请求失败，已跳过: {type(exc).__name__}: {exc}")
        return []

    out: List[RawNews] = []
    for row in (payload.get("news") or [])[:limit]:
        if not isinstance(row, dict):
            continue
        title = str(row.get("title") or "").strip()
        if not title:
            continue
        ts = row.get("providerPublishTime")
        pub = _now_utc()
        if isinstance(ts, (int, float)):
            # providerPublishTime 为 UTC 秒级时间戳 -> naive UTC
            pub = datetime.fromtimestamp(ts, tz=timezone.utc).replace(tzinfo=None)
        out.append(
            RawNews(
                source="雅虎财经",
                title=title[:512],
                content=str(row.get("summary") or "")[:4000],
                url=str(row.get("link") or "")[:1024],
                published_at=pub,
            )
        )
    if out:
        logger.info(f"[news] 雅虎财经 抓取 {len(out)} 条")
    return out


async def fetch_stock_news(
    symbols: List[str], per_symbol: int = 10, total_limit: int = MAX_TOTAL
) -> List[RawNews]:
    """按个股拉取新闻（东方财富 ``stock_news_em``），用于"关注股票相关新闻"。

    返回条目已带 ``related_stocks``，可直接被 ``/api/v1/watchlist/news`` 命中。
    """
    if not symbols:
        return []
    try:
        import akshare as ak
    except Exception as exc:  # pragma: no cover
        logger.warning(f"[news] akshare 不可用: {exc}")
        return []
    fn = getattr(ak, "stock_news_em", None)
    if fn is None:
        return []

    async def _one(sym: str) -> List[RawNews]:
        try:
            df = await asyncio.to_thread(fn, symbol=sym)
        except Exception as exc:
            logger.debug(f"[news] 个股新闻 {sym} 抓取失败: {exc}")
            return []
        if df is None or getattr(df, "empty", True):
            return []
        # 只保留 72 小时内，避免旧闻挤占 50 条名额（接口展示同样按 72h 过滤）
        fresh_cutoff = _now_utc() - timedelta(hours=DEDUP_WINDOW_HOURS)
        out: List[RawNews] = []
        for _, row in df.head(per_symbol).iterrows():
            rec = row.to_dict()
            title = _cell(rec, "新闻标题", "标题", "title")
            if not title:
                continue
            pub = _parse_dt(_cell(rec, "发布时间") or None)
            if pub and pub < fresh_cutoff:
                continue
            out.append(
                RawNews(
                    source="东方财富",
                    title=title[:512],
                    content=_cell(rec, "新闻内容", "内容")[:4000],
                    url=_cell(rec, "新闻链接", "链接")[:1024],
                    published_at=pub,
                    related_stocks=[{"symbol": str(sym), "name": "", "relevance": 1.0}],
                    tag_source="stock",
                )
            )
        return out

    results = await asyncio.gather(*[_one(s) for s in symbols[:20]])
    out: List[RawNews] = []
    for r in results:
        out.extend(r)
    if out:
        logger.info(f"[news] 关注股票个股新闻 抓取 {len(out)} 条（{min(len(symbols), 20)} 只）")
    return out[:total_limit]


# 抓取顺序即轮询取样时的优先级
SOURCES = ("东方财富", "同花顺", "财联社", "新浪财经", "腾讯财经", "雅虎财经")


async def _safe(coro, source: str) -> List[RawNews]:
    try:
        return await coro
    except Exception as exc:  # 兜底，任何源异常都不影响整体
        logger.warning(f"[news] {source} 抓取异常: {type(exc).__name__}: {exc}")
        return []


async def fetch_all_news(limit: int = MAX_TOTAL) -> List[RawNews]:
    """并发抓取所有源，跨源去重后按源轮询取至多 ``limit`` 条。"""
    # 每个源多取一些，给去重后留余量
    per_source = max(limit, 20)
    results = await asyncio.gather(
        _safe(fetch_eastmoney(per_source), "东方财富"),
        _safe(fetch_ths(per_source), "同花顺"),
        _safe(fetch_cls(per_source), "财联社"),
        _safe(fetch_sina(per_source), "新浪财经"),
        _safe(fetch_tencent(per_source), "腾讯财经"),
        _safe(fetch_yahoo(per_source), "雅虎财经"),
        return_exceptions=False,
    )

    by_source: Dict[str, List[RawNews]] = {}
    for name, items in zip(SOURCES, results):
        # 源内先按时间倒序，保证取样时优先拿新的
        by_source[name] = sorted(
            items, key=lambda x: x.published_at or datetime.min, reverse=True
        )

    seen_titles: set = set()
    seen_urls: set = set()
    merged: List[RawNews] = []

    # 轮询：每轮从每个源各取一条，直到凑满 limit 或全部耗尽
    cursor = {name: 0 for name in SOURCES}
    exhausted = False
    while len(merged) < limit and not exhausted:
        exhausted = True
        for name in SOURCES:
            if len(merged) >= limit:
                break
            items = by_source.get(name) or []
            idx = cursor[name]
            # 跳过重复项，直到取到一条新的或该源耗尽
            while idx < len(items):
                cand = items[idx]
                idx += 1
                nt = cand.norm_title()
                if not nt:
                    continue
                if nt in seen_titles or (cand.url and cand.url in seen_urls):
                    continue
                seen_titles.add(nt)
                if cand.url:
                    seen_urls.add(cand.url)
                merged.append(cand)
                exhausted = False
                break
            cursor[name] = idx
            if cursor[name] < len(items):
                exhausted = False

    logger.info(
        "[news] 聚合完成: 各源=%s, 去重后取 %d 条",
        {k: len(v) for k, v in by_source.items()}, len(merged),
    )
    return merged[:limit]


# -- 入库（供 Celery 任务与手动刷新接口复用）--------------------------------
async def _watched_symbols(db: Any) -> List[str]:
    """读取全站关注列表股票代码（去重），用于拉取个股新闻。"""
    from sqlalchemy import distinct, select

    from app.models import WatchlistItem

    try:
        res = await db.execute(select(distinct(WatchlistItem.symbol)))
        return [str(s) for s in res.scalars().all() if s]
    except Exception as exc:
        logger.debug(f"[news] 读取关注列表失败: {exc}")
        return []


async def _keyword_tag_items(items: List[RawNews]) -> None:
    """对仍无 related_stocks 的通用新闻做关键词兜底打标（不走 LLM，零成本）。"""
    if not any(it.related_stocks is None for it in items):
        return
    try:
        from app.news_tagging import get_stock_index, keyword_tag

        index = await get_stock_index()
    except Exception as exc:
        logger.debug(f"[news] 关键词打标跳过（无法获取股票索引）: {exc}")
        return
    for it in items:
        if it.related_stocks is not None:
            continue
        try:
            tags = keyword_tag(it.title, index)
        except Exception:
            continue
        it.related_stocks = tags.get("related_stocks") or []
        it.topic_tags = tags.get("topic_tags") or []
        it.sentiment = tags.get("sentiment")
        it.tag_source = "keyword"


def _merge_stock_links(existing: Any, incoming: Any) -> List[dict]:
    links: Dict[str, dict] = {}
    for stock in list(existing or []) + list(incoming or []):
        if isinstance(stock, str):
            stock = {"symbol": stock}
        if not isinstance(stock, dict):
            continue
        symbol = str(stock.get("symbol") or stock.get("code") or "").upper()
        if symbol:
            previous = links.get(symbol, {})
            links[symbol] = {**previous, **{k: v for k, v in stock.items() if v is not None and v != ""}}
    return list(links.values())


async def ingest_news(db: Any, limit: int = MAX_TOTAL) -> Dict[str, Any]:
    """抓取 -> 跨源去重 -> 关键词打标 -> 与库内近 N 小时去重 -> 写入。

    来源：通用多源快讯（东财/同花顺/财联社/新浪 + 腾讯/雅虎）
    ＋ 关注股票个股新闻（东方财富，带 related_stocks）。
    返回统计信息，便于任务日志与接口响应排查。
    """
    from uuid import uuid4

    from sqlalchemy import select

    from app.models import NewsArticle, now_utc

    # 1) 通用多源新闻
    candidate_limit = max(200, limit)
    general = await fetch_all_news(limit=candidate_limit)
    # 2) 关注股票个股新闻（自带 related_stocks）
    symbols = await _watched_symbols(db)
    stock_items = await fetch_stock_news(symbols, per_symbol=10, total_limit=candidate_limit)

    # 合并：个股新闻优先（保证 related_stocks 不丢失），再通用
    merged: List[RawNews] = []
    candidates_by_title: Dict[str, RawNews] = {}
    candidates_by_url: Dict[str, RawNews] = {}
    for it in list(stock_items) + list(general):
        nt = it.norm_title()
        if not nt and not it.url:
            continue
        duplicate = candidates_by_title.get(nt) or candidates_by_url.get(it.url)
        if duplicate is not None:
            duplicate.related_stocks = _merge_stock_links(duplicate.related_stocks, it.related_stocks) or None
            if not duplicate.url:
                duplicate.url = it.url
        else:
            duplicate = it
            merged.append(it)
        if nt:
            candidates_by_title[nt] = duplicate
        if it.url:
            candidates_by_url[it.url] = duplicate

    if not merged:
        return {"fetched": 0, "inserted": 0, "skipped": 0, "related": 0, "by_source": {}}

    # 3) 通用新闻关键词兜底打标
    await _keyword_tag_items(merged)

    # 4) 拉取近期已有新闻（含 related_stocks）做去重/补标
    cutoff = now_utc() - timedelta(hours=DEDUP_WINDOW_HOURS)
    existing = (
        await db.execute(select(NewsArticle).where(NewsArticle.published_at >= cutoff))
    ).scalars().all()
    by_title: Dict[str, NewsArticle] = {}
    by_url: Dict[str, NewsArticle] = {}
    for a in existing:
        if a.title:
            by_title[_TITLE_NOISE.sub("", a.title).lower()] = a
        if a.url:
            by_url[a.url] = a

    inserted = 0
    skipped = 0
    related_cnt = 0
    by_source: Dict[str, int] = {}
    for item in merged:
        nt = item.norm_title()
        dup = by_title.get(nt) if nt else None
        if dup is None and item.url:
            dup = by_url.get(item.url)

        if dup is not None:
            links = _merge_stock_links(dup.related_stocks, item.related_stocks)
            if links != (dup.related_stocks or []):
                dup.related_stocks = links
                related_cnt += 1
            skipped += 1
            continue

        if inserted >= limit:
            skipped += 1
            continue

        article = NewsArticle(
            id=str(uuid4()),
            source=item.source,
            title=item.title,
            content=item.content or None,
            summary=(item.content or "")[:200] or None,
            url=item.url or None,
            published_at=item.published_at or _now_utc(),
            related_stocks=item.related_stocks or None,
            topic_tags=item.topic_tags or None,
            sentiment=item.sentiment,
            tag_source=item.tag_source,
        )
        db.add(article)
        if nt:
            by_title[nt] = article
        if item.url:
            by_url[item.url] = article
        if item.related_stocks:
            related_cnt += 1
        inserted += 1
        by_source[item.source] = by_source.get(item.source, 0) + 1

    await db.commit()
    logger.info(
        "[news] 入库完成: fetched=%d inserted=%d skipped=%d related=%d by_source=%s",
        len(merged), inserted, skipped, related_cnt, by_source,
    )
    return {
        "fetched": len(merged),
        "inserted": inserted,
        "skipped": skipped,
        "related": related_cnt,
        "by_source": by_source,
    }
