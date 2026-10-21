# ============================================================================
# AI Stock Analysis Platform - Celery Tasks (Scheduled Crawlers & Workers)
# ============================================================================
from __future__ import annotations

import logging
from datetime import datetime, timedelta, date, timezone
from uuid import uuid4

from app.celery_app import celery_app
from app.config import get_settings
from app.database import task_session
from app.models import now_utc
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

    async with task_session() as db:
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


@celery_app.task(name="tag_news_topics")
def tag_news_topics():
    """为尚未打标签的新闻抽取题材标签与关联股票（HT-002 前置步骤）。

    此前 news_articles.topic_tags 没有任何写入方，而 compute_hot_topics 依赖
    topic_tags 分组聚合，导致热点榜恒为空。LLM 批量打标为主，关键词兜底为备。
    """
    import asyncio
    asyncio.run(_tag_news_topics())


async def _tag_news_topics():
    """两级打标：

    1) 关键词全量打标（题材词典 + 股票名匹配，0 成本、秒级）——保证热点榜永远有数据；
    2) LLM 精标（每日限量、优先走免费模型）——给最新一批新闻更准的标签与关联个股。

    免费模型额度是账号级共享 50 次/天，不限量的话一天就会被后台任务打光。
    """
    import asyncio
    from sqlalchemy import select
    from app.model_gateway import get_model_gateway
    from app.news_tagging import (
        LLM_BATCH_SIZE, LLM_CONCURRENCY, consume_llm_budget, get_llm_budget,
        get_stock_index, keyword_tag, pick_free_config, tag_batch_with_llm,
    )

    async with task_session() as db:
        cutoff = now_utc() - timedelta(hours=24)

        # ── 第一级：关键词全量打标 ──
        result = await db.execute(
            select(NewsArticle)
            .where(
                NewsArticle.published_at >= cutoff,
                NewsArticle.is_archived == False,
                NewsArticle.tag_source.is_(None),
            )
            .order_by(NewsArticle.published_at.desc())
            .limit(300)
        )
        articles = result.scalars().all()
        if not articles:
            logger.info("No untagged news found")
            return

        stock_index = await get_stock_index()
        keyword_tagged = 0
        for art in articles:
            info = keyword_tag(art.title, stock_index)
            if info.get("topic_tags"):
                art.topic_tags = info["topic_tags"]
                art.related_stocks = info.get("related_stocks") or []
                art.sentiment = info.get("sentiment")
                art.tag_source = "keyword"
                keyword_tagged += 1
        await db.commit()
        logger.info(f"Keyword tagging done: {keyword_tagged}/{len(articles)} tagged")

        # ── 第二级：LLM 精标（每日限量 + 免费模型优先）──
        budget = await get_llm_budget()
        if budget <= 0:
            logger.info("Daily LLM budget exhausted — keyword tagging only this round")
            return

        refine_window = now_utc() - timedelta(hours=3)
        result = await db.execute(
            select(NewsArticle)
            .where(
                NewsArticle.published_at >= refine_window,
                NewsArticle.tag_source == "keyword",
            )
            .order_by(NewsArticle.published_at.desc())
            .limit(LLM_BATCH_SIZE * 3)
        )
        candidates = result.scalars().all()
        if not candidates:
            return

        batches = [
            candidates[i:i + LLM_BATCH_SIZE]
            for i in range(0, len(candidates), LLM_BATCH_SIZE)
        ]
        affordable = min(len(batches), budget)
        if affordable <= 0:
            return
        batches = batches[:affordable]
        await consume_llm_budget(affordable)

        gateway = get_model_gateway()
        free_config_id = pick_free_config(gateway)
        logger.info(
            f"LLM refining {len(candidates)} news with {affordable} batch(es), "
            f"free_model={'yes' if free_config_id else 'no'}, budget_left={budget - affordable}"
        )

        semaphore = asyncio.Semaphore(LLM_CONCURRENCY)

        async def _run_batch(chunk):
            async with semaphore:
                return await tag_batch_with_llm(gateway, chunk, prefer_config_id=free_config_id)

        batch_results = await asyncio.gather(
            *[_run_batch(chunk) for chunk in batches],
            return_exceptions=True,
        )

        refined = 0
        for chunk, batch_result in zip(batches, batch_results):
            if isinstance(batch_result, BaseException):
                logger.warning(f"打标批次异常: {batch_result}")
                continue
            for art, info in zip(chunk, batch_result):
                if info and info.get("topic_tags"):
                    art.topic_tags = info["topic_tags"]
                    if info.get("related_stocks"):
                        art.related_stocks = info["related_stocks"]
                    if info.get("sentiment"):
                        art.sentiment = info["sentiment"]
                    art.tag_source = "llm"
                    refined += 1

        await db.commit()
        logger.info(f"LLM refine done: {refined}/{len(candidates)} refined")


@celery_app.task(name="compute_hot_topics")
def compute_hot_topics():
    """Compute hot topics ranking every 15 minutes (HT-002, HT-003)."""
    import asyncio
    asyncio.run(_compute_hot_topics())


async def _compute_hot_topics():
    from sqlalchemy import select, delete
    from app.model_gateway import get_model_gateway
    from app.engines import HotTopicEngine
    logger.info("Compute hot topics task started")
    async with task_session() as db:
        # 清理已过期热点，避免历史数据无限堆积
        await db.execute(delete(HotTopic).where(HotTopic.expires_at <= now_utc()))
        await db.flush()

        # Group news by topic_tags and compute heat
        cutoff = now_utc() - timedelta(hours=72)
        result = await db.execute(
            select(NewsArticle).where(
                NewsArticle.published_at >= cutoff,
                NewsArticle.is_archived == False,
            )
        )
        articles = result.scalars().all()
        tagged = [a for a in articles if a.topic_tags]
        logger.info(f"Compute hot topics: {len(articles)} articles, {len(tagged)} tagged")

        topic_map: dict = {}
        for a in tagged:
            for tag in (a.topic_tags or []):
                info = topic_map.setdefault(tag, {"count": 0, "stocks": {}, "articles": []})
                info["count"] += 1
                if len(info["articles"]) < 10:
                    info["articles"].append(a.title)
                for s in (a.related_stocks or []):
                    if not isinstance(s, dict):
                        continue
                    key = s.get("symbol") or s.get("name")
                    if key:
                        info["stocks"][key] = s

        # 已存在的热点按 topic_name 复用（upsert），避免每轮重复插入同一话题
        existing = {
            t.topic_name: t
            for t in (await db.execute(select(HotTopic))).scalars().all()
        }

        top = sorted(topic_map.items(), key=lambda x: x[1]["count"], reverse=True)[:10]
        topic_objs = []
        for tag, info in top:
            heat = HotTopicEngine.compute_heat_index(
                info["count"], min(info["count"] / 10.0, 1.0), 0.5,
            )
            related = list(info["stocks"].values())[:10]
            topic = existing.get(tag)
            if topic is None:
                topic = HotTopic(id=str(uuid4()), topic_name=tag)
                db.add(topic)
            topic.description = f"{tag} - {info['count']}篇相关报道"
            topic.heat_index = heat
            topic.news_count = info["count"]
            topic.related_stocks = related
            topic.generated_at = now_utc()
            topic.expires_at = now_utc() + timedelta(hours=24)
            topic_objs.append((topic, info))

        await db.flush()

        # 为热度前 5 的热点生成 AI 结论（AC-001）
        # 限量：免费模型额度账号级共享 50 次/天，AI 结论每天最多生成 2 轮，
        # 其余轮次沿用上一次的结论（热点排序仍每 15 分钟刷新）。
        from app.cache import CacheManager
        from app.news_tagging import CONCLUSION_ROUNDS_KEY, LLM_CONCLUSION_BUDGET

        try:
            rounds = int(await CacheManager.get(CONCLUSION_ROUNDS_KEY) or 0)
        except Exception:
            rounds = 0

        if rounds < LLM_CONCLUSION_BUDGET:
            gateway = get_model_gateway()
            engine = HotTopicEngine(gateway)
            for topic, info in topic_objs[:5]:
                try:
                    topic.ai_conclusion = await engine.generate_ai_conclusion(
                        topic.topic_name, info["articles"], topic.related_stocks or [],
                    )
                except Exception as exc:
                    logger.warning(f"AI 结论生成失败 topic={topic.topic_name}: {exc}")
            try:
                await CacheManager.set(
                    CONCLUSION_ROUNDS_KEY, str(rounds + 1), 24 * 3600,
                )
            except Exception as exc:
                logger.warning(f"写入热点结论轮次计数失败: {exc}")
        else:
            logger.info(
                f"Hot topic AI conclusion daily budget reached ({rounds}/"
                f"{LLM_CONCLUSION_BUDGET}), reusing previous conclusions"
            )

        await db.commit()
        logger.info(f"Computed {len(topic_objs)} hot topics")


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

    async with task_session() as db:
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

    async with task_session() as db:
        cutoff = now_utc() - timedelta(hours=48)
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
    async with task_session() as db:
        # Archive news older than 72h
        cutoff_news = now_utc() - timedelta(hours=72)
        from sqlalchemy import update
        await db.execute(
            update(NewsArticle)
            .where(NewsArticle.published_at < cutoff_news, NewsArticle.is_archived == False)
            .values(is_archived=True)
        )

        # Delete KOL opinions older than 30 days
        cutoff_kol = now_utc() - timedelta(days=30)
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
            delete(HotTopic).where(HotTopic.expires_at < now_utc())
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

    async with task_session() as db:
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


@celery_app.task(name="evaluate_decisions")
def evaluate_decisions(days: int = 5, limit: int = 100):
    """Retrospectively score agent decisions against realized prices."""
    import asyncio
    asyncio.run(_evaluate_decisions(days=days, limit=limit))


async def _evaluate_decisions(days: int = 5, limit: int = 100):
    from app.api.evaluation import evaluate_pending_decisions
    logger.info(f"Decision evaluation task started: days={days}, limit={limit}")
    async with task_session() as db:
        result = await evaluate_pending_decisions(days=days, limit=limit, db=db)
        await db.commit()
        logger.info(f"Decision evaluation task done: created={result['created']}, "
                    f"skipped={len(result['skipped'])}")
