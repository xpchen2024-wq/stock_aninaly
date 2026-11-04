"""全球市场速览（Dashboard 顶部一行行情条）：美股 + 日韩科技股。

数据源：Yahoo Finance 公开 HTTP 接口（纯 HTTP，无 JS 引擎，见 app/adapters.py 说明）。
一次刷新并发拉取全部标的，结果在进程内缓存 55 秒 —— 前端每 60 秒轮询一次即可，
既满足"每分钟更新"，又不会把上游打爆。
"""
from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import httpx

logger = logging.getLogger(__name__)

YAHOO_CHART_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
YAHOO_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
    "Accept": "application/json",
}
CACHE_TTL = 55          # 秒；略小于前端 60s 轮询周期
FETCH_TIMEOUT = 6.0     # 单标的超时
MAX_CONCURRENCY = 8

GROUP_LABELS = {"us": "美股", "kr": "韩国科技"}

# 顺序即前端展示顺序
SYMBOLS: List[Dict[str, str]] = [
    # —— 美股（指数 + 权重科技）——
    {"symbol": "^IXIC", "name": "纳斯达克", "group": "us", "kind": "index"},
    {"symbol": "^GSPC", "name": "标普500", "group": "us", "kind": "index"},
    {"symbol": "^DJI", "name": "道琼斯", "group": "us", "kind": "index"},
    {"symbol": "NVDA", "name": "英伟达", "group": "us", "kind": "stock"},
    {"symbol": "AAPL", "name": "苹果", "group": "us", "kind": "stock"},
    {"symbol": "MSFT", "name": "微软", "group": "us", "kind": "stock"},
    {"symbol": "TSLA", "name": "特斯拉", "group": "us", "kind": "stock"},
    # —— 韩国科技 ——
    {"symbol": "^KS11", "name": "韩国综合", "group": "kr", "kind": "index"},
    {"symbol": "005930.KS", "name": "三星电子", "group": "kr", "kind": "stock"},
    {"symbol": "000660.KS", "name": "SK海力士", "group": "kr", "kind": "stock"},
]

# ---- 进程内缓存 -------------------------------------------------------------
_cache: Dict[str, Any] = {"at": 0.0, "payload": None}
_lock = asyncio.Lock()


def _num(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _parse_meta(meta: Dict[str, Any]) -> Dict[str, Any]:
    """从 chart.meta 提取涨跌信息（Yahoo 直接给 regularMarketChangePercent）。"""
    price = _num(meta.get("regularMarketPrice")) or _num(meta.get("fulldayPrice")) or 0.0
    change_pct = _num(meta.get("regularMarketChangePercent"))
    if change_pct is None:
        change_pct = _num(meta.get("fulldayChangePercent"))
    if change_pct is None:
        prev = _num(meta.get("previousClose")) or _num(meta.get("chartPreviousClose"))
        change_pct = ((price / prev) - 1) * 100 if prev else None
    prev_close = _num(meta.get("previousClose")) or _num(meta.get("chartPreviousClose"))
    change = _num(meta.get("regularMarketChange"))
    if change is None and prev_close and price:
        change = price - prev_close
    return {
        "price": price,
        "change_pct": round(change_pct, 2) if change_pct is not None else None,
        "change": round(change, 2) if change is not None else None,
        "currency": meta.get("currency"),
        "market_time": meta.get("regularMarketTime"),
        "exchange": meta.get("fullExchangeName") or meta.get("exchangeName"),
    }


async def _fetch_one(client: httpx.AsyncClient, item: Dict[str, str]) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "symbol": item["symbol"],
        "name": item["name"],
        "group": item["group"],
        "group_label": GROUP_LABELS.get(item["group"], item["group"]),
        "kind": item.get("kind", "stock"),
        "price": None,
        "change_pct": None,
        "change": None,
        "currency": None,
        "market_time": None,
        "ok": False,
    }
    try:
        resp = await client.get(
            YAHOO_CHART_URL.format(symbol=item["symbol"]),
            # chartPreviousClose is relative to the requested range's start.
            params={"interval": "1d", "range": "1d"},
        )
        resp.raise_for_status()
        results = (resp.json().get("chart") or {}).get("result") or []
        if not results:
            return row
        row.update(_parse_meta(results[0].get("meta") or {}))
        row["ok"] = row["price"] is not None
    except Exception as exc:  # 单标的失败不影响整条行情带
        logger.debug(f"global quote failed for {item['symbol']}: {exc}")
    return row


async def _fetch_all() -> List[Dict[str, Any]]:
    sem = asyncio.Semaphore(MAX_CONCURRENCY)

    async with httpx.AsyncClient(
        timeout=FETCH_TIMEOUT, headers=YAHOO_HEADERS, follow_redirects=True
    ) as client:
        async def _guarded(item):
            async with sem:
                return await _fetch_one(client, item)

        return list(await asyncio.gather(*(_guarded(i) for i in SYMBOLS)))


async def get_global_markets(force: bool = False) -> Dict[str, Any]:
    """返回全球市场速览；命中缓存则直接返回（带 stale 标记）。"""
    now = time.time()
    if not force and _cache["payload"] and now - _cache["at"] < CACHE_TTL:
        payload = dict(_cache["payload"])
        payload["cached"] = True
        payload["age_sec"] = round(now - _cache["at"], 1)
        return payload

    async with _lock:
        # 双检：等锁期间别的请求可能已刷新
        now = time.time()
        if not force and _cache["payload"] and now - _cache["at"] < CACHE_TTL:
            payload = dict(_cache["payload"])
            payload["cached"] = True
            payload["age_sec"] = round(now - _cache["at"], 1)
            return payload

        items = await _fetch_all()
        ok_count = sum(1 for i in items if i["ok"])
        if ok_count == 0 and _cache["payload"]:
            # 上游全挂：回吐上一次成功的数据，标记 stale
            payload = dict(_cache["payload"])
            payload.update({"cached": True, "stale": True,
                            "age_sec": round(time.time() - _cache["at"], 1)})
            logger.warning("global markets: upstream unavailable, serving stale cache")
            return payload

        payload = {
            "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "tz_note": "美/日/韩当地交易时段，非交易时段为最近收盘价",
            "ttl": CACHE_TTL,
            "cached": False,
            "stale": False,
            "age_sec": 0,
            "ok_count": ok_count,
            "total": len(items),
            "groups": [
                {"key": key, "label": label,
                 "items": [i for i in items if i["group"] == key]}
                for key, label in GROUP_LABELS.items()
            ],
            "items": items,
        }
        _cache.update({"at": time.time(), "payload": payload})
        return payload


def clear_cache() -> None:
    """测试/手动刷新用。"""
    _cache.update({"at": 0.0, "payload": None})
