# ============================================================================
# AI Stock Analysis Platform - Analysis API (TA/SA/IR + AC conclusions)
# ============================================================================
from __future__ import annotations

import json
import logging
from typing import Optional, List
from datetime import datetime
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.database import get_db
from app.config import get_settings
from app.model_gateway import get_model_gateway
from app.adapters import create_default_failover_manager, FailoverManager, DataType
from app.engines import TrendAnalysisEngine, SerenityEngine, HotTopicEngine
from app.models import TrendAnalysis, SerenityAnalysis, ResearchReport, AnalysisConclusion

logger = logging.getLogger(__name__)
router = APIRouter()
settings = get_settings()

_gateway = None
_failover: Optional[FailoverManager] = None


def get_gateway():
    global _gateway
    if _gateway is None:
        _gateway = get_model_gateway()
    return _gateway


def get_failover() -> FailoverManager:
    global _failover
    if _failover is None:
        _failover = create_default_failover_manager(tushare_token=settings.TUSHARE_TOKEN)
    return _failover


# -- Schemas --
class TrendAnalysisRequest(BaseModel):
    symbol: str
    name: Optional[str] = ""
    frequency: str = "D"


class TrendAnalysisResponse(BaseModel):
    symbol: str
    name: str
    frequency: str
    trend_direction: Optional[str]
    confidence: Optional[str]
    ai_conclusion: str
    indicators: Optional[dict]


class SerenityRequest(BaseModel):
    symbol: str
    name: Optional[str] = ""
    sector: Optional[str] = ""


class SerenityResponse(BaseModel):
    symbol: str
    name: str
    step1_bom: str
    step2_bottleneck: str
    step3_adversarial: str
    step4_float: str
    step5_matrix: str
    conditions_met: Optional[dict]
    ai_conclusion: str


class ReportListResponse(BaseModel):
    id: str
    broker: str
    title: str
    stock_symbol: Optional[str]
    stock_name: Optional[str]
    rating: Optional[str]
    target_price: Optional[float]
    published_at: str


class ConclusionResponse(BaseModel):
    id: str
    scene: str
    symbol: Optional[str]
    conclusion_md: str
    generated_at: str


# -- Routes --
@router.post("/trend", response_model=TrendAnalysisResponse)
async def analyze_trend(req: TrendAnalysisRequest, db: AsyncSession = Depends(get_db)):
    """AI Trend Analysis (TA-001 ~ TA-005, AC-002)."""
    logger.info(f"Trend analysis started: symbol={req.symbol}, frequency={req.frequency}")
    engine = TrendAnalysisEngine(get_gateway(), get_failover())
    result = await engine.analyze(req.symbol, req.name, req.frequency)

    logger.info(
        f"Trend analysis completed: symbol={req.symbol}, "
        f"direction={result.get('trend_direction')}, confidence={result.get('confidence')}"
    )

    # Persist result
    record = TrendAnalysis(
        id=str(uuid4()), symbol=req.symbol, name=req.name or "",
        frequency=req.frequency,
        trend_direction=result.get("trend_direction"),
        confidence=result.get("confidence"),
        ai_conclusion=result.get("ai_conclusion", ""),
        raw_indicators=result.get("raw_indicators"),
    )
    db.add(record)
    await db.flush()

    return TrendAnalysisResponse(
        symbol=result["symbol"], name=result.get("name", ""),
        frequency=result.get("frequency", "D"),
        trend_direction=result.get("trend_direction"),
        confidence=result.get("confidence"),
        ai_conclusion=result.get("ai_conclusion", "分析失败"),
        indicators=result.get("raw_indicators"),
    )


@router.post("/serenity", response_model=SerenityResponse)
async def run_serenity(req: SerenityRequest, db: AsyncSession = Depends(get_db)):
    """Serenity 5-step deep analysis (SA-001 ~ SA-003, AC-004)."""
    logger.info(f"Serenity analysis started: symbol={req.symbol}, sector={req.sector or 'N/A'}")
    engine = SerenityEngine(get_gateway())
    result = await engine.analyze(req.symbol, req.name, req.sector)

    logger.info(f"Serenity analysis completed: symbol={req.symbol}")

    record = SerenityAnalysis(
        id=str(uuid4()), symbol=req.symbol, name=req.name or "",
        step1_bom=result["step1_bom"],
        step2_bottleneck=result["step2_bottleneck"],
        step3_adversarial=result["step3_adversarial"],
        step4_float=result["step4_float"],
        step5_matrix=result["step5_matrix"],
        conditions_met=result["conditions_met"],
        ai_conclusion=result["ai_conclusion"],
    )
    db.add(record)
    await db.flush()

    return SerenityResponse(
        symbol=result["symbol"], name=result.get("name", ""),
        step1_bom=result["step1_bom"], step2_bottleneck=result["step2_bottleneck"],
        step3_adversarial=result["step3_adversarial"], step4_float=result["step4_float"],
        step5_matrix=result["step5_matrix"],
        conditions_met=result["conditions_met"],
        ai_conclusion=result["ai_conclusion"],
    )


@router.post("/serenity/stream")
async def run_serenity_stream(req: SerenityRequest, db: AsyncSession = Depends(get_db)):
    """Serenity 5-step analysis with SSE streaming progress."""
    logger.info(f"Serenity stream started: symbol={req.symbol}, sector={req.sector or 'N/A'}")
    async def event_generator():
        try:
            engine = SerenityEngine(get_gateway())
            async for chunk in engine.analyze_stream(req.symbol, req.name, req.sector):
                yield chunk
        except Exception as e:
            import traceback
            logger.error(f"Serenity stream error: {traceback.format_exc()}")
            error_data = {"message": str(e) or "分析过程发生未知错误"}
            yield f"event: error\ndata: {json.dumps(error_data, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/reports", response_model=List[ReportListResponse])
async def list_reports(
    symbol: Optional[str] = Query(None),
    broker: Optional[str] = Query(None),
    limit: int = Query(20, le=100),
    db: AsyncSession = Depends(get_db),
):
    """List research reports (IR-001 ~ IR-005)."""
    logger.info(f"Listing reports: symbol={symbol or 'all'}, broker={broker or 'all'}, limit={limit}")
    query = select(ResearchReport).order_by(ResearchReport.published_at.desc()).limit(limit)
    if symbol:
        query = query.where(ResearchReport.stock_symbol == symbol)
    if broker:
        query = query.where(ResearchReport.broker == broker)

    result = await db.execute(query)
    reports = result.scalars().all()
    logger.info(f"Research reports found: {len(reports)}")
    return [
        ReportListResponse(
            id=str(r.id), broker=r.broker, title=r.title,
            stock_symbol=r.stock_symbol, stock_name=r.stock_name,
            rating=r.rating, target_price=r.target_price,
            published_at=r.published_at.isoformat() if r.published_at else "",
        )
        for r in reports
    ]


@router.get("/conclusions", response_model=List[ConclusionResponse])
async def list_conclusions(
    scene: Optional[str] = Query(None, description="hot_topic/trend/report/serenity/decision/kol"),
    symbol: Optional[str] = Query(None),
    limit: int = Query(20, le=100),
    db: AsyncSession = Depends(get_db),
):
    """List AI-generated conclusions (AC-001 ~ AC-008)."""
    logger.info(f"Listing conclusions: scene={scene or 'all'}, symbol={symbol or 'all'}, limit={limit}")
    query = select(AnalysisConclusion).order_by(
        AnalysisConclusion.generated_at.desc()
    ).limit(limit)
    if scene:
        query = query.where(AnalysisConclusion.scene == scene)
    if symbol:
        query = query.where(AnalysisConclusion.symbol == symbol)

    result = await db.execute(query)
    conclusions = result.scalars().all()
    logger.info(f"Analysis conclusions found: {len(conclusions)}")
    return [
        ConclusionResponse(
            id=str(c.id), scene=c.scene, symbol=c.symbol,
            conclusion_md=c.conclusion_md,
            generated_at=c.generated_at.isoformat() if c.generated_at else "",
        )
        for c in conclusions
    ]


@router.get("/history/trend")
async def trend_history(
    symbol: str = Query(...),
    limit: int = Query(10, le=50),
    db: AsyncSession = Depends(get_db),
):
    """Get historical trend analyses for a symbol."""
    logger.info(f"Getting trend history: symbol={symbol}, limit={limit}")
    result = await db.execute(
        select(TrendAnalysis)
        .where(TrendAnalysis.symbol == symbol)
        .order_by(TrendAnalysis.generated_at.desc())
        .limit(limit)
    )
    items = result.scalars().all()
    logger.info(f"Trend history found: symbol={symbol}, count={len(items)}")
    return [
        {
            "id": str(t.id), "frequency": t.frequency,
            "trend_direction": t.trend_direction, "confidence": t.confidence,
            "ai_conclusion": t.ai_conclusion,
            "generated_at": t.generated_at.isoformat() if t.generated_at else "",
        }
        for t in items
    ]


@router.get("/history/serenity")
async def serenity_history(
    symbol: str = Query(...),
    limit: int = Query(5, le=20),
    db: AsyncSession = Depends(get_db),
):
    """Get historical Serenity analyses for a symbol."""
    logger.info(f"Getting serenity history: symbol={symbol}, limit={limit}")
    result = await db.execute(
        select(SerenityAnalysis)
        .where(SerenityAnalysis.symbol == symbol)
        .order_by(SerenityAnalysis.generated_at.desc())
        .limit(limit)
    )
    items = result.scalars().all()
    logger.info(f"Serenity history found: symbol={symbol}, count={len(items)}")
    return [
        {
            "id": str(s.id), "ai_conclusion": s.ai_conclusion,
            "conditions_met": s.conditions_met,
            "generated_at": s.generated_at.isoformat() if s.generated_at else "",
        }
        for s in items
    ]
