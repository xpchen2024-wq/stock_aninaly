# ============================================================================
# AI Stock Analysis Platform - KOL Opinion Feed (news-grounded)
# ============================================================================
"""基于真实新闻的 KOL 观点生成（供定时任务与手动刷新复用）。

设计原则（诚实、可溯源）：
- **不伪造平台原文**。每条观点都绑定一条真实、近期（48h 内）的财经新闻：
  摘要取真实事件核心，``content_url`` 指向真实原文链接，个股/题材取新闻的
  ``related_stocks`` / ``topic_tags``，方向由真实文本与情绪判定。
- KOL 为看板内的合成人格（种子数据），因此不存在对真实人物的错误归属；
  观点内容本身均有真实来源可经链接核验。
- 每个 KOL 在 48h 内最多保留 ``PER_KOL_CAP`` 条，避免每 30 分钟重复刷屏。
"""
from __future__ import annotations

import logging
import math
import re
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional
from uuid import uuid4

logger = logging.getLogger(__name__)

PER_KOL_CAP = 5      # 每个 KOL 近 48h 最多保留的观点数
MAX_KOLS = 100       # 最多处理的 KOL 数（抖音50 + 微博50）
NEWS_SCAN = 200      # 扫描的近期新闻上限

_BULL = ("涨", "涨停", "大涨", "走高", "新高", "利好", "超预期", "增长", "突破", "飙升",
         "净流入", "看多", "加仓", "走强", "反弹", "上调", "中标", "订单", "回暖", "创新高")
_BEAR = ("跌", "跌停", "大跌", "走低", "新低", "利空", "下滑", "亏损", "跳水", "净流出",
         "看空", "减仓", "走弱", "下调", "风险", "处罚", "诉讼", "暴跌", "承压", "减持")
_BRACKET = re.compile(r"^[\s【\[（(]+|[\s】\]）)]+$")


def _naive_utc(dt: Optional[datetime]) -> Optional[datetime]:
    """转成 naive UTC，与模型 DateTime 列 / now_utc() 口径一致。"""
    if dt is None:
        return None
    if dt.tzinfo is None:
        return dt
    return dt.astimezone(timezone.utc).replace(tzinfo=None)


def _clean(text: str) -> str:
    text = (text or "").strip()
    # 去掉标题常见的【xxx】前缀（正文本身常以它开头，摘要里重复）
    text = re.sub(r"^【[^】]*】", "", text).strip()
    # 去掉新闻电头，如"财联社10月27日电，"、"证券时报网讯，"，更像观点表述
    text = re.sub(
        r"^(?:财联社|证券时报|第一财经|新华财经|界面新闻|智通财经|每经|新华社)\s*"
        r"(?:\d{1,2}月\d{1,2}日)?(?:电|讯)?[，,：:]\s*",
        "",
        text,
    ).strip()
    return _BRACKET.sub("", text)


def _direction(article: Any) -> str:
    """方向判定：先看已打标情绪，再用文本关键词，默认中性。"""
    sent = getattr(article, "sentiment", None)
    if sent == "positive":
        return "bullish"
    if sent == "negative":
        return "bearish"
    title = f"{getattr(article, 'title', '') or ''} {getattr(article, 'summary', '') or ''}"
    bull = sum(1 for k in _BULL if k in title)
    bear = sum(1 for k in _BEAR if k in title)
    if bull > bear:
        return "bullish"
    if bear > bull:
        return "bearish"
    return "neutral"


def _build_summary(article: Any, direction: str) -> str:
    """把真实新闻凝练成一句"观点式"摘要（≤150 字）。"""
    core = _clean(getattr(article, "summary", "") or getattr(article, "title", "") or "")
    if not core:
        core = _clean(getattr(article, "title", "") or "")
    core = core[:110]
    lead = {"bullish": "看多", "bearish": "看空", "neutral": "关注"}.get(direction, "关注")
    tags = getattr(article, "topic_tags", None) or []
    if isinstance(tags, list) and tags:
        summary = f"{lead}｜{core}（{tags[0]}）"
    else:
        summary = f"{lead}｜{core}"
    return summary[:150]


def _heat(kol: Any, article: Any, max_followers: int, now_naive: datetime) -> float:
    """热度 = 粉丝量(对数归一) + 时效(48h 线性衰减)。落在 50~100。"""
    fw = math.log10(max(getattr(kol, "followers_count", 0) or 0, 1) + 10) / math.log10(
        max(max_followers, 1) + 10
    )
    pub = _naive_utc(getattr(article, "published_at", None)) or now_naive
    hours = max((now_naive - pub).total_seconds() / 3600.0, 0.0)
    rec = max(0.0, 1.0 - hours / 48.0)
    return round(min(50.0 + 40.0 * fw + 10.0 * rec, 99.9), 2)


def _mentioned_stocks(article: Any, direction: str) -> List[Dict[str, str]]:
    """新闻 related_stocks -> KOL 观点的 mentioned_stocks（code/name/direction，UI 用 name||code）。"""
    out: List[Dict[str, str]] = []
    for s in (getattr(article, "related_stocks", None) or [])[:4]:
        if isinstance(s, dict):
            code = str(s.get("symbol") or s.get("code") or "")
            name = str(s.get("name") or "")
            if code or name:
                out.append({"code": code, "name": name, "direction": direction})
        elif isinstance(s, str):
            out.append({"code": s, "name": "", "direction": direction})
    return out


async def generate_kol_opinions(
    db: Any, per_kol_cap: int = PER_KOL_CAP, platforms: tuple = ("douyin", "weibo")
) -> Dict[str, Any]:
    """基于近 48h 真实新闻为活跃 KOL 生成观点并入库。

    抖音与微博均为**合成人格**（微博种子已改为"财经观察01"这类，不再使用真实人物姓名）。
    仍保留"V认证"护栏作双保险：任何真实认证账号一律跳过，绝不伪造真人言论。
    返回统计：kols / news / inserted / skipped。
    """
    from sqlalchemy import func, select

    from app.models import KOL, KOLOpinion, NewsArticle, now_utc

    now = now_utc()
    cutoff = now - timedelta(hours=48)

    kols = (
        await db.execute(
            select(KOL).where(KOL.is_active == True, KOL.platform.in_(platforms)).limit(MAX_KOLS)
        )
    ).scalars().all()
    # 安全护栏：真实名人（V认证）不生成观点，避免把 AI 内容伪装成其言论
    kols = [k for k in kols if "V认证" not in (k.certification or "")]
    if not kols:
        return {"kols": 0, "news": 0, "inserted": 0, "skipped": 0}

    news = (
        await db.execute(
            select(NewsArticle)
            .where(NewsArticle.published_at >= cutoff, NewsArticle.is_archived == False)
            .order_by(NewsArticle.published_at.desc())
            .limit(NEWS_SCAN)
        )
    ).scalars().all()
    if not news:
        logger.info("[kol] 近 48h 无新闻，无法生成观点")
        return {"kols": len(kols), "news": 0, "inserted": 0, "skipped": 0}

    # 优先用带真实个股关联的新闻，让观点更有"料"（有可点开的标的）
    news_stock = [n for n in news if n.related_stocks]
    ordered = news_stock + [n for n in news if n not in news_stock]
    n_ordered = len(ordered)

    # 已有观点：每个 KOL 的计数 + (kol, url) / (kol, summary) 去重集
    cnt_rows = await db.execute(
        select(KOLOpinion.kol_id, func.count())
        .where(KOLOpinion.published_at >= cutoff)
        .group_by(KOLOpinion.kol_id)
    )
    counts = {str(k): int(c) for k, c in cnt_rows.all()}

    ex_rows = await db.execute(
        select(KOLOpinion.kol_id, KOLOpinion.content_url, KOLOpinion.summary)
        .where(KOLOpinion.published_at >= cutoff)
    )
    seen_url = set()
    seen_sum = set()
    for kid, url, summ in ex_rows.all():
        if url:
            seen_url.add((str(kid), url))
        if summ:
            seen_sum.add((str(kid), summ))

    max_followers = max((k.followers_count or 0 for k in kols), default=1) or 1
    n_news = len(news)
    inserted = 0
    skipped = 0

    for idx, kol in enumerate(kols):
        kid = str(kol.id)
        if counts.get(kid, 0) >= per_kol_cap:
            skipped += 1
            continue

        # 轮询挑选一条"该 KOL 还没用过"的真实新闻（优先带个股的），保证多样、避免重复
        article = None
        for off in range(n_ordered):
            cand = ordered[(idx + off) % n_ordered]
            cu = cand.url or ""
            if cu and (kid, cu) in seen_url:
                continue
            candidate_summary = _build_summary(cand, _direction(cand))
            if not candidate_summary or (kid, candidate_summary) in seen_sum:
                continue
            article = cand
            break
        if article is None:
            skipped += 1
            continue

        direction = _direction(article)
        summary = _build_summary(article, direction)
        if not summary or (kid, summary) in seen_sum:
            skipped += 1
            continue

        stocks = _mentioned_stocks(article, direction)
        heat = _heat(kol, article, max_followers, now)

        pub = _naive_utc(getattr(article, "published_at", None)) or now
        if pub < cutoff:  # 兜底，确保落在 48h 窗口内
            pub = now
        hours = max((now - pub).total_seconds() / 3600.0, 0.0)
        rec = max(0.0, 1.0 - hours / 48.0)
        base = max(kol.followers_count or 0, 1000)
        likes = int(base * 0.0015 * (0.5 + 0.5 * rec))
        comments = max(1, likes // 15)
        shares = max(1, likes // 25)

        db.add(
            KOLOpinion(
                id=str(uuid4()),
                kol_id=kol.id,
                platform=kol.platform,
                content_url=(article.url or None),
                summary=summary,
                direction=direction,
                raw_text=(article.summary or article.content or article.title),
                mentioned_stocks=stocks or None,
                topic_tags=article.topic_tags,
                heat_score=heat,
                likes_count=likes,
                comments_count=comments,
                shares_count=shares,
                published_at=pub,
            )
        )
        if article.url:
            seen_url.add((kid, article.url))
        seen_sum.add((kid, summary))
        counts[kid] = counts.get(kid, 0) + 1
        inserted += 1

    await db.commit()
    stats = {"kols": len(kols), "news": n_news, "inserted": inserted, "skipped": skipped}
    logger.info(f"[kol] 观点生成完成: {stats}")
    return stats
