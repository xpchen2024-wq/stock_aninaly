# ============================================================================
# AI Stock Analysis Platform - Celery Tasks (Scheduled Crawlers & Workers)
# ============================================================================
from __future__ import annotations

import logging
from datetime import datetime, timedelta, date
from uuid import uuid4

from app.celery_app import celery_app
from app.config import get_settings
from app.database import async_session_factory
from app.models import (
    NewsArticle, HotTopic, KOL, KOLOpinion, KOLConsensus,
    AnalysisConclusion, KlineCache, StockFundamental,
)

logger = logging.getLogger(__name__)
settings = get_settings()


@celery_app.task(name="crawl_news")
def crawl_news():
    """Crawl financial news every 15 minutes (HT-001, HT-003)."""
    import asyncio
    asyncio.run(_crawl_news())


async def _crawl_news():
    import akshare as ak
    sources = ["财联社", "华尔街见闻", "新浪财经", "东方财富"]
    logger.info(f"Crawl news task started: sources={sources}")

    async with async_session_factory() as db:
        for source_name in sources:
            try:
                # AkShare provides stock news
                if "财联社" in source_name:
                    df = ak.stock_info_global_em()
                else:
                    df = ak.stock_info_global_em()

                if df is None or df.empty:
                    continue

                for _, row in df.head(50).iterrows():
                    title = str(row.get("标题", row.get("title", "")))
                    content = str(row.get("内容", row.get("content", "")))
                    pub_time = row.get("发布时间", row.get("datetime", datetime.now()))

                    if isinstance(pub_time, str):
                        try:
                            pub_time = datetime.fromisoformat(pub_time)
                        except (ValueError, TypeError):
                            pub_time = datetime.now()

                    article = NewsArticle(
                        id=str(uuid4()),
                        source=source_name,
                        title=title[:512],
                        content=content,
                        url=str(row.get("链接", row.get("url", ""))),
                        published_at=pub_time,
                    )
                    db.add(article)
                await db.commit()
                logger.info(f"Crawled news from {source_name}")
            except Exception as e:
                logger.warning(f"Failed to crawl {source_name}: {e}")
                await db.rollback()


@celery_app.task(name="compute_hot_topics")
def compute_hot_topics():
    """Compute hot topics ranking every 15 minutes (HT-002, HT-003)."""
    import asyncio
    asyncio.run(_compute_hot_topics())


async def _compute_hot_topics():
    from sqlalchemy import select, func
    logger.info("Compute hot topics task started")
    async with async_session_factory() as db:
        # Group news by topic_tags and compute heat
        cutoff = datetime.utcnow() - timedelta(hours=72)
        result = await db.execute(
            select(NewsArticle).where(
                NewsArticle.published_at >= cutoff,
                NewsArticle.is_archived == False,
            )
        )
        articles = result.scalars().all()

        topic_map = {}
        for a in articles:
            for tag in (a.topic_tags or []):
                if tag not in topic_map:
                    topic_map[tag] = {
                        "name": tag, "count": 0, "stocks": [], "articles": [],
                    }
                topic_map[tag]["count"] += 1
                topic_map[tag]["articles"].append(a.title)
                for s in (a.related_stocks or []):
                    if s not in topic_map[tag]["stocks"]:
                        topic_map[tag]["stocks"].append(s)

        # Compute heat index and save top topics
        for tag, info in sorted(topic_map.items(), key=lambda x: x[1]["count"], reverse=True)[:10]:
            heat = HotTopicEngine.compute_heat_index(
                info["count"], info["count"] / 10.0, 0.5
            ) if hasattr(HotTopicEngine, 'compute_heat_index') else info["count"] * 10

            topic = HotTopic(
                id=str(uuid4()),
                topic_name=info["name"],
                description=f"{info['name']} - {info['count']}篇相关报道",
                heat_index=heat,
                news_count=info["count"],
                related_stocks=info["stocks"][:10],
                expires_at=datetime.utcnow() + timedelta(hours=24),
            )
            db.add(topic)

        await db.commit()
        logger.info(f"Computed {len(topic_map)} hot topics")


@celery_app.task(name="crawl_kol_opinions")
def crawl_kol_opinions():
    """Crawl KOL opinions every 30 minutes (KV-001)."""
    import asyncio
    asyncio.run(_crawl_kol_opinions())


async def _crawl_kol_opinions():
    """Crawl KOL opinions from Douyin and Weibo.

    Note: In production, this would use platform-specific APIs or scrapers.
    This is a placeholder that demonstrates the data flow.
    """
    from app.model_gateway import get_model_gateway
    from app.engines import KOLEngine
    logger.info("Crawl KOL opinions task started")

    async with async_session_factory() as db:
        from sqlalchemy import select
        result = await db.execute(select(KOL).where(KOL.is_active == True))
        kols = result.scalars().all()

        gateway = get_model_gateway()
        engine = KOLEngine(gateway)

        # Get max followers for heat normalization
        max_followers = max((k.followers_count for k in kols), default=1)

        for kol in kols:
            try:
                # Placeholder: In production, fetch from real platform API
                # raw_texts = await platform_api.fetch_recent_posts(kol)
                # For now, skip actual crawling
                logger.debug(f"Would crawl opinions for {kol.nickname} ({kol.platform})")
            except Exception as e:
                logger.warning(f"Failed to crawl KOL {kol.nickname}: {e}")

        await db.commit()


@celery_app.task(name="generate_kol_consensus")
def generate_kol_consensus():
    """Generate daily KOL consensus (KV-005, AC-008)."""
    import asyncio
    asyncio.run(_generate_kol_consensus())


async def _generate_kol_consensus():
    from sqlalchemy import select
    from app.model_gateway import get_model_gateway
    from app.engines import KOLEngine
    logger.info("Generate KOL consensus task started")

    async with async_session_factory() as db:
        cutoff = datetime.utcnow() - timedelta(hours=48)
        result = await db.execute(
            select(KOLOpinion).where(KOLOpinion.published_at >= cutoff)
        )
        opinions = result.scalars().all()

        if not opinions:
            logger.info("No KOL opinions to aggregate")
            return

        # Build opinion data
        kols_result = await db.execute(select(KOL))
        kols = {str(k.id): k for k in kols_result.scalars().all()}
        max_followers = max((k.followers_count for k in kols.values()), default=1)

        opinion_data = []
        for op in opinions:
            kol = kols.get(str(op.kol_id))
            opinion_data.append({
                "mentioned_stocks": op.mentioned_stocks or [],
                "direction": op.direction,
            })

        gateway = get_model_gateway()
        engine = KOLEngine(gateway)
        consensus = await engine.generate_consensus(opinion_data, date.today().isoformat())

        record = KOLConsensus(
            id=str(uuid4()),
            summary_date=date.today(),
            topic="Daily KOL Consensus",
            bullish_stocks=consensus.get("hot_stocks", []),
            mention_count=len(opinions),
            ai_summary=consensus.get("ai_summary", ""),
        )
        db.add(record)
        await db.commit()
        logger.info(f"Generated KOL consensus with {len(opinions)} opinions")


@celery_app.task(name="cleanup_expired_data")
def cleanup_expired_data():
    """Clean up expired data (HT-004, KV retention)."""
    import asyncio
    asyncio.run(_cleanup_expired_data())


async def _cleanup_expired_data():
    from sqlalchemy import delete, select
    logger.info("Cleanup expired data task started")
    async with async_session_factory() as db:
        # Archive news older than 72h
        cutoff_news = datetime.utcnow() - timedelta(hours=72)
        from sqlalchemy import update
        await db.execute(
            update(NewsArticle)
            .where(NewsArticle.published_at < cutoff_news, NewsArticle.is_archived == False)
            .values(is_archived=True)
        )

        # Delete KOL opinions older than 30 days
        cutoff_kol = datetime.utcnow() - timedelta(days=30)
        await db.execute(
            delete(KOLOpinion).where(KOLOpinion.published_at < cutoff_kol)
        )

        # Delete K-line cache older than 90 days
        cutoff_kline = date.today() - timedelta(days=90)
        await db.execute(
            delete(KlineCache).where(KlineCache.trade_date < cutoff_kline)
        )

        # Delete expired hot topics
        await db.execute(
            delete(HotTopic).where(HotTopic.expires_at < datetime.utcnow())
        )

        await db.commit()
        logger.info("Expired data cleanup completed")


@celery_app.task(name="update_watchlist_cache")
def update_watchlist_cache():
    """Update K-line cache for watchlist stocks (WL-004)."""
    import asyncio
    asyncio.run(_update_watchlist_cache())


async def _update_watchlist_cache():
    from sqlalchemy import select
    from app.models import WatchlistItem
    from app.adapters import create_default_failover_manager, DataType
    logger.info("Update watchlist cache task started")

    async with async_session_factory() as db:
        result = await db.execute(select(WatchlistItem))
        items = result.scalars().all()

        manager = create_default_failover_manager(tushare_token=settings.TUSHARE_TOKEN)

        for item in items:
            try:
                klines = await manager.fetch_with_failover(
                    DataType.KLINE, item.symbol, frequency="D"
                )
                for k in klines[-90:]:
                    cache = KlineCache(
                        id=str(uuid4()),
                        symbol=item.symbol, frequency="D",
                        trade_date=datetime.strptime(k.date, "%Y-%m-%d").date()
                        if "-" in k.date else datetime.strptime(k.date, "%Y%m%d").date(),
                        open=k.open, high=k.high, low=k.low, close=k.close,
                        volume=k.volume, amount=k.amount, turnover_rate=k.turnover_rate,
                    )
                    db.add(cache)
                await db.commit()
                logger.info(f"Updated cache for {item.symbol}")
            except Exception as e:
                logger.warning(f"Failed to update cache for {item.symbol}: {e}")
                await db.rollback()
