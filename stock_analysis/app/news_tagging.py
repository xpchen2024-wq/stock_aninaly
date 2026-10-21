# ============================================================================
# AI Stock Analysis Platform - News Tagging (HT-002 前置步骤)
# ============================================================================
"""为新闻抽取热点标签与关联股票。

背景：``news_articles.topic_tags`` / ``related_stocks`` 此前没有任何写入方，
而 ``compute_hot_topics`` 完全依赖 ``topic_tags`` 分组聚合，导致热点榜恒为空
（接口 /api/v1/news/hot-topics 永远返回 []）。

本模块提供两条路径：

1. **LLM 批量打标（主路径）**：一次调用处理多条新闻标题，要求模型输出严格 JSON，
   兼顾质量与成本（当前默认模型为免费额度）。
2. **关键词兜底**：LLM 不可用或返回非法 JSON 时，用题材词典 + 股票名称匹配
   做最低限度标注，保证热点链路不中断。
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any, Dict, List, Optional, Sequence

from app.utils.json_extract import parse_llm_json

logger = logging.getLogger(__name__)


# -- 题材词典（兜底抽取用）--------------------------------------------------
TOPIC_KEYWORDS: Dict[str, List[str]] = {
    "AI芯片": ["芯片", "半导体", "晶圆", "光刻", "算力", "GPU", "先进封装", "存储", "封测", "EDA"],
    "人工智能": ["人工智能", "AI", "大模型", "AIGC", "智能体", "多模态", "GPT", "推理", "机器学习"],
    "新能源": ["锂电", "光伏", "储能", "风电", "氢能", "碳酸锂", "硅料", "逆变器", "钠电"],
    "机器人": ["机器人", "人形机器人", "灵巧手", "谐波减速器", "丝杠", "减速器"],
    "军工": ["军工", "国防", "航空发动机", "导弹", "军贸", "航天", "军费"],
    "医药生物": ["创新药", "医保", "疫苗", "CXO", "减肥药", "GLP-1", "医药", "集采", "生物科技"],
    "消费": ["白酒", "食品饮料", "免税", "旅游", "家电", "以旧换新", "零售", "餐饮"],
    "地产": ["房地产", "楼市", "保交楼", "房贷", "土地", "房企", "限购"],
    "金融": ["银行", "券商", "保险", "降准", "降息", "LPR", "金融", "信贷", "央行"],
    "汽车": ["汽车", "新能源车", "智能驾驶", "自动驾驶", "车企", "智驾", "车市"],
    "有色金属": ["黄金", "铜", "稀土", "锂矿", "铝", "钴", "有色", "白银"],
    "煤炭": ["煤炭", "动力煤", "焦煤", "煤价"],
    "石油石化": ["原油", "油价", "成品油", "天然气", "炼化", "OPEC"],
    "通信": ["5G", "6G", "光模块", "运营商", "卫星", "光通信", "交换机"],
    "游戏传媒": ["游戏", "版号", "影视", "短剧", "漫剧", "传媒", "票房"],
    "农业": ["猪价", "生猪", "种业", "粮食", "养殖", "转基因"],
}

STOCK_INDEX_CACHE_KEY = "news_tagging:stock_index"
STOCK_INDEX_TTL = 12 * 3600
LLM_BATCH_SIZE = 10
LLM_CONCURRENCY = 5

# 每日 LLM 调用配额（openrouter 免费额度是**账号级共享 50 次/天**，
# 所有 :free 模型加在一起算，因此后台打标必须限量，否则一天就把额度打光）
LLM_BUDGET_KEY = "news_tagging:llm_budget"
LLM_DAILY_BUDGET = 20          # 新闻精标：每天 20 次调用
LLM_CONCLUSION_BUDGET = 2     # 热点 AI 结论：每天 2 轮（每轮 top5）
BUDGET_TTL = 24 * 3600

# 热点 AI 结论已生成轮次计数（每天上限 LLM_CONCLUSION_BUDGET 轮）
CONCLUSION_ROUNDS_KEY = "hot_topics:ai_conclusion_rounds"


# -- 每日配额 --------------------------------------------------------------
async def get_llm_budget() -> int:
    """返回今日剩余的 LLM 调用次数。"""
    from app.cache import CacheManager
    try:
        used = int(await CacheManager.get(LLM_BUDGET_KEY) or 0)
    except Exception:
        used = 0
    return max(LLM_DAILY_BUDGET - used, 0)


async def consume_llm_budget(cost: int = 1) -> int:
    """消耗配额，返回消耗后的剩余次数；不足时不扣减并返回 0。"""
    from app.cache import CacheManager
    try:
        used = int(await CacheManager.get(LLM_BUDGET_KEY) or 0)
    except Exception:
        used = 0
    remaining = max(LLM_DAILY_BUDGET - used, 0)
    if remaining < cost:
        return 0
    try:
        await CacheManager.set(LLM_BUDGET_KEY, str(used + cost), BUDGET_TTL)
    except Exception as exc:
        logger.warning(f"写入 LLM 配额计数失败: {exc}")
    return remaining - cost


def pick_free_config(gateway: Any) -> Optional[str]:
    """挑一个免费模型（model_name 以 :free 结尾）的配置 ID，用于优先走免费额度。"""
    try:
        configs = getattr(gateway, "_configs", {}) or {}
    except Exception:
        return None
    for cfg in configs.values():
        if getattr(cfg, "is_enabled", False) and str(cfg.model_name).endswith(":free"):
            return str(cfg.id)
    return None


# -- 股票名称索引 ------------------------------------------------------------
async def get_stock_index() -> Dict[str, str]:
    """返回 {股票名称: 股票代码} 映射。

    复用 ``app/stocks.py`` 的统一主数据（Redis 缓存，多 worker 共享），
    避免与 /data/search 重复拉取全市场列表。
    """
    from app.stocks import get_stock_name_index
    return await get_stock_name_index()


# -- 关键词兜底 --------------------------------------------------------------
def keyword_tag(title: str, stock_index: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    """不依赖 LLM 的最低限度标注：题材词典 + 股票名称/代码匹配。"""
    title = title or ""

    tags = [
        topic for topic, keywords in TOPIC_KEYWORDS.items()
        if any(kw in title for kw in keywords)
    ][:5]

    related: List[Dict[str, Any]] = []
    seen_codes = set()
    for name, code in (stock_index or {}).items():
        # 两字股票名易误命中（如"中国""东方"），只匹配 3 字及以上的名称
        if len(name) >= 3 and name in title and code not in seen_codes:
            seen_codes.add(code)
            related.append({"symbol": code, "name": name, "relevance": 0.8})
        if len(related) >= 8:
            break

    for code in re.findall(r"\b(\d{6})\b", title):
        if code not in seen_codes:
            seen_codes.add(code)
            related.append({"symbol": code, "name": "", "relevance": 0.6})

    sentiment = None
    if any(kw in title for kw in ("大涨", "涨停", "拉升", "走高", "增长", "超预期", "利好")):
        sentiment = "positive"
    elif any(kw in title for kw in ("大跌", "跌停", "跳水", "走低", "下滑", "亏损", "利空")):
        sentiment = "negative"

    return {
        "topic_tags": tags,
        "related_stocks": related,
        "sentiment": sentiment,
    }


# -- LLM 批量打标 ------------------------------------------------------------
TAGGING_PROMPT = """你是一位资深财经舆情分析师。请为下面 {count} 条财经新闻打标签。

## 新闻列表
{news_block}

## 输出要求
严格输出一个 JSON 数组，不要输出任何解释文字、不要用 markdown 代码块。每个元素对应一条新闻，用 "i" 字段对应上面的序号：
[{{"i": 1, "tags": ["题材标签1"], "stocks": [{{"symbol": "6位代码", "name": "股票名"}}], "s": "positive"}}]

规则：
1. tags：从标题提炼 1~3 个中文题材标签（如 半导体 / 新能源 / 军工 / 白酒），粒度一致，不要太泛也不要太细。若标题只是指数或大盘播报且无明确题材，tags 给 ["大盘"]。
2. stocks：只填标题中**明确点名**的个股，symbol 填 6 位代码、name 填股票简称；没有明确个股就给空数组，禁止臆测。
3. s：标题倾向判为 positive / negative / neutral，看不出则 neutral。
4. 字段名严格用 i / tags / stocks / s，不要用其他名字；不要输出除 i 之外的任何多余字段。
"""


def _parse_json_array(text: str) -> List[Dict[str, Any]]:
    """从 LLM 回复中解析出 JSON 数组（parse_llm_json 只处理对象）。"""
    if not text:
        return []
    raw = text.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```[a-zA-Z]*\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw).strip()
    try:
        data = json.loads(raw)
        return data if isinstance(data, list) else []
    except json.JSONDecodeError:
        pass
    match = re.search(r"\[.*\]", raw, re.S)
    if match:
        try:
            data = json.loads(match.group(0))
            return data if isinstance(data, list) else []
        except json.JSONDecodeError:
            return []
    return []


def _normalize_tag_item(item: Dict[str, Any]) -> Dict[str, Any]:
    """兼容模型可能使用的多种字段名，归一化为统一结构。"""
    tags = item.get("tags")
    if tags is None:
        tags = item.get("topic_tags")
    if isinstance(tags, str):
        tags = [tags]
    tags = [str(t).strip() for t in (tags or []) if str(t).strip()][:5]

    stocks = item.get("stocks")
    if stocks is None:
        stocks = item.get("related_stocks")
    cleaned_stocks = []
    for s in stocks if isinstance(stocks, list) else []:
        if not isinstance(s, dict):
            continue
        symbol = str(s.get("symbol", "")).strip()
        name = str(s.get("name", "")).strip()
        if symbol or name:
            cleaned_stocks.append({"symbol": symbol, "name": name, "relevance": 0.9})

    sentiment = str(item.get("s") or item.get("sentiment") or "").strip().lower()
    if sentiment not in ("positive", "negative", "neutral"):
        sentiment = None

    return {
        "index": item.get("i", item.get("index", item.get("id"))),
        "topic_tags": tags,
        "related_stocks": cleaned_stocks,
        "sentiment": sentiment,
    }


async def tag_batch_with_llm(
    gateway: Any,
    articles: Sequence[Any],
    prefer_config_id: Optional[str] = None,
) -> List[Optional[Dict[str, Any]]]:
    """一次 LLM 调用为一批新闻打标。

    prefer_config_id：优先使用该模型（如免费模型），失败时仍走网关的故障转移链。
    返回与入参等长的列表，元素为标签信息或 None（该条打标失败）。
    """
    if not articles:
        return []

    lines = [f"{i + 1}. {(getattr(art, 'title', '') or '').strip()}"
             for i, art in enumerate(articles)]
    prompt = TAGGING_PROMPT.format(count=len(articles), news_block="\n".join(lines))

    try:
        response = await gateway.chat(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=3000,
            temperature=0.2,
            prefer_config_id=prefer_config_id,
        )
    except Exception as exc:
        logger.warning(f"LLM 打标调用异常: {exc}")
        return [None] * len(articles)

    if not getattr(response, "success", False):
        logger.warning(f"LLM 打标失败: {getattr(response, 'error', 'unknown error')}")
        return [None] * len(articles)

    items = _parse_json_array(getattr(response, "content", "") or "")
    if not items:
        logger.warning("LLM 打标返回非法 JSON，转用关键词兜底")
        return [None] * len(articles)

    normalized = [_normalize_tag_item(it) for it in items if isinstance(it, dict)]
    by_index: Dict[Any, Dict[str, Any]] = {}
    for item in normalized:
        key = item["index"]
        try:
            key = int(key)
        except (TypeError, ValueError):
            continue
        by_index[key] = item

    result: List[Optional[Dict[str, Any]]] = []
    for i in range(len(articles)):
        hit = by_index.get(i + 1)
        if hit and hit["topic_tags"]:
            result.append({
                "topic_tags": hit["topic_tags"],
                "related_stocks": hit["related_stocks"],
                "sentiment": hit["sentiment"],
            })
        else:
            result.append(None)
    return result
