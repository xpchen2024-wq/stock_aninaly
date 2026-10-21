# ============================================================================
# AI Stock Analysis Platform - Stock Master Data (代码/名称检索)
# ============================================================================
"""全市场股票代码-名称主数据的获取与检索。

原先股票列表放在 ``app/api/data.py`` 的进程内内存缓存里，有两个问题：

1. uvicorn 以 ``--workers 4`` 启动时，每个 worker 各拉一次 AkShare 全量列表
   （5000+ 只，首个请求要等数秒到十几秒）；
2. 进程重启即失效，缓存无法跨进程共享。

这里把主数据统一收敛到本模块，改用 Redis 缓存（多 worker 共享、跨重启保留），
并提供按代码/名称的模糊匹配（支持中文名、6 位代码、带 .SH/.SZ 后缀、代码+名称混输）。

``app/api/data.py`` 的 /data/search 与 ``app/news_tagging.py`` 的股票名索引
都改为复用本模块，避免重复拉取。
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

STOCK_LIST_CACHE_KEY = "stock_master:list"
STOCK_LIST_TTL = 12 * 3600  # 12 小时

# 进程内兜底缓存（Redis 不可用时使用）
_mem_cache: Optional[List[Dict[str, str]]] = None
_mem_cache_time: float = 0.0
_MEM_TTL = 3600.0


def _guess_market(code: str) -> str:
    """按代码首位判断交易所。"""
    if code.startswith(("6", "9")):
        return "SH"
    if code.startswith(("4", "8")):
        return "BJ"
    return "SZ"


def _normalize_symbol(raw: str) -> str:
    """从用户输入中提取 6 位纯代码。

    支持：``300750`` / ``300750.SZ`` / ``300750 sz`` / ``SZ300750``。
    """
    if not raw:
        return ""
    text = raw.strip().upper()
    m = re.search(r"(\d{6})", text)
    return m.group(1) if m else ""


async def _fetch_stock_list_from_akshare() -> List[Dict[str, str]]:
    """从 AkShare 拉取全 A 股代码-名称表。"""
    import akshare as ak

    def _load() -> List[Dict[str, str]]:
        df = ak.stock_info_a_code_name()
        code_col = next((c for c in ["代码", "code", "symbol"] if c in df.columns), None)
        name_col = next((c for c in ["名称", "name", "证券简称"] if c in df.columns), None)
        if not code_col or not name_col:
            raise ValueError(f"未知列: {list(df.columns)[:10]}")
        out: List[Dict[str, str]] = []
        for _, row in df.iterrows():
            code = str(row[code_col]).strip()
            name = str(row[name_col]).strip()
            if len(code) == 6 and code.isdigit() and name:
                out.append({"symbol": code, "name": name, "market": _guess_market(code)})
        return out

    return await asyncio.to_thread(_load)


async def get_stock_list() -> List[Dict[str, str]]:
    """获取全市场股票列表，优先走 Redis 缓存。"""
    global _mem_cache, _mem_cache_time

    # 1) Redis 缓存
    try:
        from app.cache import CacheManager
        cached = await CacheManager.get(STOCK_LIST_CACHE_KEY)
        if cached:
            data = json.loads(cached)
            if isinstance(data, list) and data:
                return data
    except Exception as exc:
        logger.warning(f"读取股票主数据缓存失败: {exc}")

    # 2) 进程内兜底缓存
    import time
    now = time.time()
    if _mem_cache and now - _mem_cache_time < _MEM_TTL:
        return _mem_cache

    # 3) 实际拉取
    try:
        stocks = await _fetch_stock_list_from_akshare()
    except Exception as exc:
        logger.warning(f"拉取股票主数据失败: {exc}")
        return _mem_cache or []

    _mem_cache = stocks
    _mem_cache_time = now
    logger.info(f"Stock master loaded: {len(stocks)} stocks")

    try:
        from app.cache import CacheManager
        await CacheManager.set(
            STOCK_LIST_CACHE_KEY,
            json.dumps(stocks, ensure_ascii=False),
            STOCK_LIST_TTL,
        )
    except Exception as exc:
        logger.warning(f"写入股票主数据缓存失败: {exc}")

    return stocks


async def get_stock_name_index() -> Dict[str, str]:
    """返回 {股票名称: 代码} 映射。"""
    stocks = await get_stock_list()
    return {s["name"]: s["symbol"] for s in stocks if s.get("name")}


def _score_match(stock: Dict[str, str], keyword: str) -> int:
    """匹配打分：分数越高越靠前，0 表示不匹配。"""
    code = stock.get("symbol", "")
    name = stock.get("name", "")
    if not code or not name:
        return 0
    if code == keyword:
        return 100          # 代码完全匹配
    if name == keyword:
        return 95           # 名称完全匹配
    if code.startswith(keyword):
        return 85           # 代码前缀匹配
    if name.startswith(keyword):
        return 80           # 名称前缀匹配
    if keyword in name:
        return 60           # 名称包含
    if keyword in code:
        return 50           # 代码包含
    return 0


def search_in_list(stocks: List[Dict[str, str]], query: str, limit: int = 20) -> List[Dict[str, str]]:
    """在股票列表中按代码/名称模糊匹配，按相关度排序。

    支持空格分隔的多关键词（如 ``"300750 宁德"``：任一命中即算匹配）。
    """
    query = (query or "").strip()
    if not query:
        return []

    # 去掉交易所后缀，只用主体做匹配
    keywords = [k for k in re.split(r"[\s,，]+", query) if k]
    keywords = [k.upper() for k in keywords]
    if not keywords:
        return []

    # 纯代码输入：优先做代码精确/前缀匹配
    code_keyword = _normalize_symbol(query)
    if code_keyword:
        exact = [s for s in stocks if s["symbol"] == code_keyword]
        if exact:
            return exact[:limit]

    scored: List[Tuple[int, Dict[str, str]]] = []
    for s in stocks:
        best = 0
        for kw in keywords:
            score = _score_match(s, kw)
            if score > best:
                best = score
        if best > 0:
            scored.append((best, s))

    # 分数降序；同分时按代码升序，保证结果稳定
    scored.sort(key=lambda x: (-x[0], x[1]["symbol"]))
    return [s for _, s in scored[:limit]]
