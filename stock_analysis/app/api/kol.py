# ============================================================================
# AI Stock Analysis Platform - KOL Opinions API (KV-001 ~ KV-005)
# ============================================================================
from __future__ import annotations

import logging
import math
from typing import Optional, List
from datetime import datetime, timedelta, date
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_

from app.database import get_db
from app.config import get_settings
from app.models import KOL, KOLOpinion, KOLConsensus
from app.model_gateway import get_model_gateway
from app.engines import KOLEngine

logger = logging.getLogger(__name__)
router = APIRouter()
settings = get_settings()


# -- Schemas --
class KOLResponse(BaseModel):
    id: str
    platform: str
    nickname: str
    avatar_url: Optional[str]
    certification: Optional[str]
    followers_count: int
    rank_position: Optional[int]


class KOLOpinionResponse(BaseModel):
    id: str
    kol_id: str
    platform: str
    content_url: Optional[str]
    summary: Optional[str]
    direction: Optional[str]
    mentioned_stocks: Optional[list]
    topic_tags: Optional[list]
    heat_score: Optional[float]
    likes_count: int
    comments_count: int
    shares_count: int
    published_at: str
    # Joined KOL info
    kol_nickname: Optional[str]
    kol_certification: Optional[str]
    kol_followers: Optional[int]


class KOLConsensusResponse(BaseModel):
    id: str
    summary_date: str
    topic: Optional[str]
    bullish_stocks: Optional[list]
    bearish_stocks: Optional[list]
    mention_count: int
    consensus_score: Optional[float]
    ai_summary: Optional[str]
    generated_at: str


class KOLOpinionExtractRequest(BaseModel):
    raw_text: str


class ExtractedOpinion(BaseModel):
    summary: str
    direction: str
    mentioned_stocks: list
    topic_tags: list
    confidence: float


class KOLManageRequest(BaseModel):
    platform: str
    nickname: str
    certification: Optional[str] = None
    followers_count: int = 0
    rank_position: Optional[int] = None


# -- Routes --
@router.get("/kols", response_model=List[KOLResponse])
async def list_kols(
    platform: Optional[str] = Query(None, description="douyin / weibo"),
    limit: int = Query(100, le=200),
    db: AsyncSession = Depends(get_db),
):
    """List KOLs (KV-001: Top 100 = 50 per platform)."""
    logger.info(f"Listing KOLs: platform={platform or 'all'}, limit={limit}")
    query = select(KOL).where(KOL.is_active == True)
    if platform:
        query = query.where(KOL.platform == platform)
    query = query.order_by(KOL.rank_position.asc().nullslast(), KOL.followers_count.desc()).limit(limit)

    result = await db.execute(query)
    kols = result.scalars().all()
    logger.info(f"KOLs found: {len(kols)}")
    return [
        KOLResponse(
            id=str(k.id), platform=k.platform, nickname=k.nickname,
            avatar_url=k.avatar_url, certification=k.certification,
            followers_count=k.followers_count, rank_position=k.rank_position,
        )
        for k in kols
    ]


@router.post("/kols", response_model=KOLResponse, status_code=201)
async def add_kol(req: KOLManageRequest, db: AsyncSession = Depends(get_db)):
    """Add a new KOL (KV-002: manual management)."""
    logger.info(f"Adding KOL: platform={req.platform}, nickname={req.nickname}")
    kol = KOL(
        id=str(uuid4()), platform=req.platform,
        nickname=req.nickname, certification=req.certification,
        followers_count=req.followers_count, rank_position=req.rank_position,
    )
    db.add(kol)
    await db.flush()
    logger.info(f"KOL added: id={kol.id}, nickname={kol.nickname}")
    return KOLResponse(
        id=str(kol.id), platform=kol.platform, nickname=kol.nickname,
        avatar_url=kol.avatar_url, certification=kol.certification,
        followers_count=kol.followers_count, rank_position=kol.rank_position,
    )


@router.delete("/kols/{kol_id}", status_code=204)
async def remove_kol(kol_id: str, db: AsyncSession = Depends(get_db)):
    """Remove a KOL."""
    logger.info(f"Removing KOL: id={kol_id}")
    result = await db.execute(select(KOL).where(KOL.id == kol_id))
    kol = result.scalar_one_or_none()
    if not kol:
        logger.warning(f"KOL not found: id={kol_id}")
        raise HTTPException(status_code=404, detail="KOL not found")
    logger.info(f"Removing KOL: nickname={kol.nickname}, platform={kol.platform}")
    await db.delete(kol)


@router.get("/opinions", response_model=List[KOLOpinionResponse])
async def list_opinions(
    platform: Optional[str] = Query(None, description="douyin / weibo"),
    direction: Optional[str] = Query(None, description="bullish / bearish / neutral"),
    hours: int = Query(48, ge=1, le=168, description="Within N hours (KV-004: 48h default)"),
    sort_by: str = Query("heat", description="heat / time"),
    limit: int = Query(100, le=200, description="Max opinions to return (KV-001~005: default 100)"),
    db: AsyncSession = Depends(get_db),
):
    """List KOL opinions within time window (KV-004)."""
    logger.info(
        f"Listing KOL opinions: platform={platform or 'all'}, direction={direction or 'all'}, "
        f"hours={hours}, sort={sort_by}, limit={limit}"
    )
    cutoff = datetime.utcnow() - timedelta(hours=hours)

    query = (
        select(KOLOpinion, KOL)
        .join(KOL, KOLOpinion.kol_id == KOL.id)
        .where(KOLOpinion.published_at >= cutoff)
    )
    if platform:
        query = query.where(KOLOpinion.platform == platform)
    if direction:
        query = query.where(KOLOpinion.direction == direction)

    if sort_by == "time":
        query = query.order_by(KOLOpinion.published_at.desc())
    else:
        query = query.order_by(KOLOpinion.heat_score.desc().nullslast())

    query = query.limit(limit)

    result = await db.execute(query)
    rows = result.all()
    logger.info(f"KOL opinions found: {len(rows)}")
    return [
        KOLOpinionResponse(
            id=str(op.id), kol_id=str(op.kol_id), platform=op.platform,
            content_url=op.content_url, summary=op.summary,
            direction=op.direction, mentioned_stocks=op.mentioned_stocks,
            topic_tags=op.topic_tags, heat_score=op.heat_score,
            likes_count=op.likes_count, comments_count=op.comments_count,
            shares_count=op.shares_count,
            published_at=op.published_at.isoformat() if op.published_at else "",
            kol_nickname=kol.nickname, kol_certification=kol.certification,
            kol_followers=kol.followers_count,
        )
        for op, kol in rows
    ]


@router.post("/opinions/extract", response_model=ExtractedOpinion)
async def extract_opinion(req: KOLOpinionExtractRequest):
    """AI extract opinion from raw text (KV-003)."""
    logger.info(f"Extracting opinion from raw text ({len(req.raw_text)} chars)")
    gateway = get_model_gateway()
    engine = KOLEngine(gateway)
    result = await engine.extract_opinion(req.raw_text)
    logger.info(
        f"Opinion extracted: direction={result.get('direction')}, "
        f"stocks={len(result.get('mentioned_stocks', []))}"
    )
    return ExtractedOpinion(
        summary=result.get("summary", ""),
        direction=result.get("direction", "neutral"),
        mentioned_stocks=result.get("mentioned_stocks", []),
        topic_tags=result.get("topic_tags", []),
        confidence=result.get("confidence", 0.5),
    )


@router.get("/consensus", response_model=List[KOLConsensusResponse])
async def list_consensus(
    days: int = Query(7, ge=1, le=30),
    db: AsyncSession = Depends(get_db),
):
    """List KOL consensus summaries (KV-005, AC-008)."""
    logger.info(f"Listing KOL consensus: days={days}")
    cutoff = date.today() - timedelta(days=days)
    result = await db.execute(
        select(KOLConsensus)
        .where(KOLConsensus.summary_date >= cutoff)
        .order_by(KOLConsensus.summary_date.desc())
    )
    items = result.scalars().all()
    logger.info(f"KOL consensus found: {len(items)}")
    return [
        KOLConsensusResponse(
            id=str(c.id),
            summary_date=c.summary_date.isoformat() if c.summary_date else "",
            topic=c.topic, bullish_stocks=c.bullish_stocks,
            bearish_stocks=c.bearish_stocks, mention_count=c.mention_count,
            consensus_score=c.consensus_score, ai_summary=c.ai_summary,
            generated_at=c.generated_at.isoformat() if c.generated_at else "",
        )
        for c in items
    ]


@router.post("/consensus/generate")
async def generate_consensus(
    db: AsyncSession = Depends(get_db),
):
    """Generate today's KOL consensus (KV-005, AC-008)."""
    logger.info("Generating KOL consensus")
    # Fetch recent 48h opinions
    cutoff = datetime.utcnow() - timedelta(hours=48)
    result = await db.execute(
        select(KOLOpinion).where(KOLOpinion.published_at >= cutoff)
    )
    opinions = result.scalars().all()

    if not opinions:
        logger.warning("Consensus generation: no recent opinions found")
        raise HTTPException(status_code=404, detail="No recent opinions to aggregate")

    # Get max followers for heat normalization
    kols_result = await db.execute(select(KOL))
    kols = {str(k.id): k for k in kols_result.scalars().all()}
    max_followers = max((k.followers_count for k in kols.values()), default=1)

    # Build opinion list for engine
    opinion_data = []
    for op in opinions:
        kol = kols.get(str(op.kol_id))
        heat = KOLEngine.compute_heat_score(
            op.likes_count, op.comments_count, op.shares_count,
            kol.followers_count if kol else 0, max_followers,
        ) if kol else 0
        opinion_data.append({
            "mentioned_stocks": op.mentioned_stocks or [],
            "direction": op.direction,
            "heat_score": heat,
        })

    # Generate consensus
    gateway = get_model_gateway()
    engine = KOLEngine(gateway)
    today_str = date.today().isoformat()
    consensus = await engine.generate_consensus(opinion_data, today_str)

    # Persist
    record = KOLConsensus(
        id=str(uuid4()),
        summary_date=date.today(),
        topic="Daily KOL Consensus",
        bullish_stocks=consensus.get("hot_stocks", []),
        bearish_stocks=[],
        mention_count=len(opinions),
        consensus_score=0.0,
        ai_summary=consensus.get("ai_summary", ""),
    )
    db.add(record)
    await db.flush()

    logger.info(
        f"KOL consensus generated: id={record.id}, opinions={len(opinions)}, "
        f"hot_stocks={len(consensus.get('hot_stocks', []))}"
    )
    return {
        "id": str(record.id),
        "summary_date": today_str,
        "opinion_count": len(opinions),
        "hot_stocks": consensus.get("hot_stocks", []),
        "ai_summary": consensus.get("ai_summary", ""),
    }


@router.get("/stats/overview")
async def kol_stats(
    db: AsyncSession = Depends(get_db),
):
    """KOL opinion stats overview for dashboard."""
    logger.info("Computing KOL stats overview")
    cutoff_48h = datetime.utcnow() - timedelta(hours=48)

    # Count KOLs
    douyin_count = await db.scalar(
        select(func.count(KOL.id)).where(KOL.platform == "douyin", KOL.is_active == True)
    )
    weibo_count = await db.scalar(
        select(func.count(KOL.id)).where(KOL.platform == "weibo", KOL.is_active == True)
    )

    # Count 48h opinions
    opinion_count = await db.scalar(
        select(func.count(KOLOpinion.id)).where(KOLOpinion.published_at >= cutoff_48h)
    )

    # Hot stocks (≥3 mentions)
    all_opinions = await db.execute(
        select(KOLOpinion).where(KOLOpinion.published_at >= cutoff_48h)
    )
    stock_mentions = {}
    for op in all_opinions.scalars().all():
        for stock in (op.mentioned_stocks or []):
            code = stock.get("code", "")
            if code:
                stock_mentions[code] = stock_mentions.get(code, 0) + 1
    hot_stocks = [s for s, c in stock_mentions.items() if c >= 3]

    # Latest consensus
    latest_consensus = await db.execute(
        select(KOLConsensus).order_by(KOLConsensus.generated_at.desc()).limit(1)
    )
    latest = latest_consensus.scalar_one_or_none()

    logger.info(
        f"KOL stats: kols={douyin_count or 0}+{weibo_count or 0}, "
        f"opinions_48h={opinion_count or 0}, hot_stocks={len(hot_stocks)}"
    )
    return {
        "douyin_kols": douyin_count or 0,
        "weibo_kols": weibo_count or 0,
        "total_kols": (douyin_count or 0) + (weibo_count or 0),
        "opinions_48h": opinion_count or 0,
        "hot_stocks_count": len(hot_stocks),
        "hot_stocks": hot_stocks[:10],
        "latest_consensus_at": latest.generated_at.isoformat() if latest and latest.generated_at else None,
    }


@router.get("/ranking")
async def kol_ranking(
    platform: Optional[str] = Query(None),
    limit: int = Query(10, le=50),
    db: AsyncSession = Depends(get_db),
):
    """KOL influence ranking (KV-002)."""
    logger.info(f"Computing KOL ranking: platform={platform or 'all'}, limit={limit}")
    query = select(KOL).where(KOL.is_active == True)
    if platform:
        query = query.where(KOL.platform == platform)
    query = query.order_by(KOL.followers_count.desc()).limit(limit)

    result = await db.execute(query)
    kols = result.scalars().all()

    # Get 48h opinion count and avg heat per KOL
    cutoff = datetime.utcnow() - timedelta(hours=48)
    ranking = []
    for kol in kols:
        op_result = await db.execute(
            select(
                func.count(KOLOpinion.id),
                func.avg(KOLOpinion.heat_score),
            ).where(
                KOLOpinion.kol_id == kol.id,
                KOLOpinion.published_at >= cutoff,
            )
        )
        op_count, avg_heat = op_result.one()
        ranking.append({
            "id": str(kol.id),
            "platform": kol.platform,
            "nickname": kol.nickname,
            "certification": kol.certification,
            "followers": kol.followers_count,
            "rank": kol.rank_position,
            "opinions_48h": op_count or 0,
            "avg_heat": round(float(avg_heat), 1) if avg_heat else 0,
        })

    # Sort by followers then avg_heat
    ranking.sort(key=lambda x: (x["followers"], x["avg_heat"]), reverse=True)
    for i, r in enumerate(ranking):
        r["position"] = i + 1
    logger.info(f"KOL ranking computed: {len(ranking)} entries")
    return ranking


# ============================================================================
# Cross Comparison: KOL Opinions vs System Analysis (KV-005)
# ============================================================================

@router.get("/cross-comparison")
async def cross_comparison(
    db: AsyncSession = Depends(get_db),
):
    """
    Compare KOL recommended stocks with system analysis results
    (trend analysis + research reports).

    For each hot stock (≥3 KOL mentions), fetch the latest trend analysis
    and research report to show agreement/divergence.
    """
    logger.info("Computing KOL cross-comparison")
    from app.models import TrendAnalysis, ResearchReport

    # 1. Aggregate KOL mentioned stocks in last 48h
    cutoff = datetime.utcnow() - timedelta(hours=48)
    result = await db.execute(
        select(KOLOpinion).where(KOLOpinion.published_at >= cutoff)
    )
    opinions = result.scalars().all()

    stock_mentions = {}  # code -> {name, bullish, bearish, total, directions}
    for op in opinions:
        for stock in (op.mentioned_stocks or []):
            code = stock.get("code", "")
            if not code:
                continue
            if code not in stock_mentions:
                stock_mentions[code] = {
                    "code": code,
                    "name": stock.get("name", code),
                    "bullish_count": 0,
                    "bearish_count": 0,
                    "neutral_count": 0,
                    "total_count": 0,
                }
            sm = stock_mentions[code]
            sm["total_count"] += 1
            direction = stock.get("direction", "")
            if direction == "看多":
                sm["bullish_count"] += 1
            elif direction == "看空":
                sm["bearish_count"] += 1
            else:
                sm["neutral_count"] += 1

    # Hot stocks: ≥3 mentions
    hot_stocks = [s for s in stock_mentions.values() if s["total_count"] >= 1]

    # 2. For each hot stock, fetch latest trend analysis and research report
    comparisons = []
    for sm in hot_stocks[:20]:  # limit to 20 stocks
        code = sm["code"]

        # Latest trend analysis
        trend_result = await db.execute(
            select(TrendAnalysis)
            .where(TrendAnalysis.symbol == code)
            .order_by(TrendAnalysis.generated_at.desc())
            .limit(1)
        )
        trend = trend_result.scalar_one_or_none()

        # Latest research report
        report_result = await db.execute(
            select(ResearchReport)
            .where(ResearchReport.stock_symbol == code)
            .order_by(ResearchReport.published_at.desc())
            .limit(1)
        )
        report = report_result.scalar_one_or_none()

        # Determine KOL direction
        if sm["bullish_count"] > sm["bearish_count"]:
            kol_direction = "bullish"
        elif sm["bearish_count"] > sm["bullish_count"]:
            kol_direction = "bearish"
        else:
            kol_direction = "neutral"

        # Determine consistency
        trend_dir = trend.trend_direction if trend else None
        report_rating = report.rating if report else None

        consistency = "unknown"
        if trend_dir and kol_direction:
            if kol_direction == "bullish" and trend_dir == "bullish":
                consistency = "consistent"
            elif kol_direction == "bearish" and trend_dir == "bearish":
                consistency = "consistent"
            elif kol_direction != trend_dir and trend_dir != "sideways":
                consistency = "divergent"
            else:
                consistency = "partial"

        comparisons.append({
            "symbol": code,
            "name": sm["name"],
            "kol_direction": kol_direction,
            "kol_mentions": sm["total_count"],
            "kol_bullish": sm["bullish_count"],
            "kol_bearish": sm["bearish_count"],
            "trend_direction": trend_dir,
            "trend_confidence": trend.confidence if trend else None,
            "trend_date": trend.generated_at.isoformat() if trend else None,
            "report_rating": report_rating,
            "report_broker": report.broker if report else None,
            "report_target_price": report.target_price if report else None,
            "consistency": consistency,
        })

    # Sort by mention count desc
    comparisons.sort(key=lambda x: x["kol_mentions"], reverse=True)

    logger.info(
        f"Cross-comparison computed: {len(comparisons)} stocks, "
        f"consistent={sum(1 for c in comparisons if c['consistency'] == 'consistent')}, "
        f"divergent={sum(1 for c in comparisons if c['consistency'] == 'divergent')}"
    )
    return {
        "total_hot_stocks": len(comparisons),
        "comparisons": comparisons,
        "summary": {
            "consistent": sum(1 for c in comparisons if c["consistency"] == "consistent"),
            "divergent": sum(1 for c in comparisons if c["consistency"] == "divergent"),
            "partial": sum(1 for c in comparisons if c["consistency"] == "partial"),
            "unknown": sum(1 for c in comparisons if c["consistency"] == "unknown"),
        },
    }
