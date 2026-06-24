# ============================================================================
# AI Stock Analysis Platform - News & Hot Topics API (HT-001 ~ HT-004)
# ============================================================================
from __future__ import annotations

import logging
from typing import Optional, List
from datetime import datetime, timedelta

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func

from app.database import get_db
from app.config import get_settings
from app.models import NewsArticle, HotTopic

logger = logging.getLogger(__name__)
router = APIRouter()
settings = get_settings()


# -- Schemas --
class NewsResponse(BaseModel):
    id: str
    source: str
    title: str
    summary: Optional[str]
    url: Optional[str]
    published_at: str
    related_stocks: Optional[list]
    topic_tags: Optional[list]
    sentiment: Optional[str]
    heat_score: Optional[float]


class HotTopicResponse(BaseModel):
    id: str
    topic_name: str
    description: Optional[str]
    heat_index: float
    news_count: int
    related_stocks: Optional[list]
    ai_conclusion: Optional[str]
    generated_at: str


# -- Routes --
@router.get("", response_model=List[NewsResponse])
async def list_news(
    source: Optional[str] = Query(None),
    hours: int = Query(72, ge=1, le=168, description="Filter within N hours"),
    limit: int = Query(20, le=100),
    db: AsyncSession = Depends(get_db),
):
    """List recent news articles (HT-001, HT-004: 72h window)."""
    logger.info(f"Listing news: source={source or 'all'}, hours={hours}, limit={limit}")
    cutoff = datetime.utcnow() - timedelta(hours=hours)
    query = (
        select(NewsArticle)
        .where(
            NewsArticle.published_at >= cutoff,
            NewsArticle.is_archived == False,
        )
        .order_by(NewsArticle.published_at.desc())
        .limit(limit)
    )
    if source:
        query = query.where(NewsArticle.source == source)

    result = await db.execute(query)
    articles = result.scalars().all()
    logger.info(f"News articles found: {len(articles)}")
    return [
        NewsResponse(
            id=str(a.id), source=a.source, title=a.title,
            summary=a.summary, url=a.url,
            published_at=a.published_at.isoformat() if a.published_at else "",
            related_stocks=a.related_stocks, topic_tags=a.topic_tags,
            sentiment=a.sentiment, heat_score=a.heat_score,
        )
        for a in articles
    ]


@router.get("/hot-topics", response_model=List[HotTopicResponse])
async def list_hot_topics(
    limit: int = Query(10, le=50),
    db: AsyncSession = Depends(get_db),
):
    """List hot topics ranked by heat index (HT-002, HT-003)."""
    logger.info(f"Listing hot topics: limit={limit}")
    now = datetime.utcnow()
    result = await db.execute(
        select(HotTopic)
        .where(HotTopic.expires_at > now)
        .order_by(HotTopic.heat_index.desc())
        .limit(limit)
    )
    topics = result.scalars().all()
    logger.info(f"Hot topics found: {len(topics)}")
    return [
        HotTopicResponse(
            id=str(t.id), topic_name=t.topic_name,
            description=t.description, heat_index=t.heat_index,
            news_count=t.news_count, related_stocks=t.related_stocks,
            ai_conclusion=t.ai_conclusion,
            generated_at=t.generated_at.isoformat() if t.generated_at else "",
        )
        for t in topics
    ]


@router.get("/{news_id}")
async def get_news_detail(news_id: str, db: AsyncSession = Depends(get_db)):
    """Get full news article detail."""
    logger.info(f"Getting news detail: news_id={news_id}")
    result = await db.execute(select(NewsArticle).where(NewsArticle.id == news_id))
    article = result.scalar_one_or_none()
    if not article:
        logger.warning(f"News article not found: news_id={news_id}")
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="News not found")
    logger.debug(f"News detail retrieved: title={article.title[:50]}...")
    return {
        "id": str(article.id),
        "source": article.source,
        "title": article.title,
        "content": article.content,
        "summary": article.summary,
        "url": article.url,
        "published_at": article.published_at.isoformat() if article.published_at else "",
        "related_stocks": article.related_stocks,
        "topic_tags": article.topic_tags,
        "sentiment": article.sentiment,
        "heat_score": article.heat_score,
    }


@router.get("/stats/summary")
async def news_stats(db: AsyncSession = Depends(get_db)):
    """News statistics summary."""
    logger.info("Computing news stats summary")
    cutoff = datetime.utcnow() - timedelta(hours=72)
    total = await db.scalar(
        select(func.count(NewsArticle.id)).where(
            NewsArticle.published_at >= cutoff,
            NewsArticle.is_archived == False,
        )
    )
    by_source = await db.execute(
        select(NewsArticle.source, func.count(NewsArticle.id))
        .where(NewsArticle.published_at >= cutoff)
        .group_by(NewsArticle.source)
    )
    logger.info(f"News stats: total_72h={total or 0}")
    return {
        "total_72h": total or 0,
        "by_source": {row[0]: row[1] for row in by_source},
    }
