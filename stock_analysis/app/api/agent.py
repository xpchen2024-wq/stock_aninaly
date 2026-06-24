# ============================================================================
# AI Stock Analysis Platform - Agent Decision API (AGENT-xxx)
# ============================================================================
from __future__ import annotations

import logging
from typing import Optional, List
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.database import get_db
from app.config import get_settings
from app.model_gateway import get_model_gateway
from app.adapters import create_default_failover_manager, FailoverManager
from app.agents import AgentWorkflow
from app.models import AgentDecision, AnalysisConclusion

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
class AgentRunRequest(BaseModel):
    symbol: str
    name: Optional[str] = ""
    fundamentals: Optional[dict] = None
    news_items: Optional[List[dict]] = None


class AgentRunResponse(BaseModel):
    session_id: str
    symbol: str
    name: str
    status: str
    fundamental_report: Optional[dict]
    sentiment_report: Optional[dict]
    news_report: Optional[dict]
    technical_report: Optional[dict]
    debate_conclusion: Optional[dict]
    trade_proposal: Optional[dict]
    risk_assessment: Optional[dict]
    approval_decision: Optional[dict]
    errors: List[str]
    completed_at: str


class DecisionListResponse(BaseModel):
    id: str
    session_id: str
    symbol: str
    name: str
    trade_side: Optional[str]
    approval_status: Optional[str]
    execution_status: Optional[str]
    created_at: str


# -- Routes --
@router.post("/run", response_model=AgentRunResponse)
async def run_agent_workflow(req: AgentRunRequest, db: AsyncSession = Depends(get_db)):
    """Run complete 5-phase agent workflow for a stock."""
    logger.info(f"Agent workflow started: symbol={req.symbol}, name={req.name or 'N/A'}")
    workflow = AgentWorkflow(get_gateway(), get_failover())
    result = await workflow.run(
        symbol=req.symbol, name=req.name or "",
        fundamentals=req.fundamentals, news_items=req.news_items,
    )

    session_id = result["session_id"]
    status = result.get("status", "unknown")
    errors = result.get("errors", [])
    logger.info(
        f"Agent workflow completed: session_id={session_id}, symbol={req.symbol}, "
        f"status={status}, errors={len(errors)}"
    )

    # Persist decision record
    record = AgentDecision(
        id=str(uuid4()),
        session_id=result["session_id"],
        symbol=req.symbol, name=req.name or "",
        analyst_reports={
            "fundamental": result.get("fundamental_report"),
            "sentiment": result.get("sentiment_report"),
            "news": result.get("news_report"),
            "technical": result.get("technical_report"),
        },
        debate_conclusion=result.get("debate_conclusion"),
        trade_side=result.get("trade_proposal", {}).get("side"),
        trade_quantity=result.get("trade_proposal", {}).get("quantity"),
        stop_loss=result.get("trade_proposal", {}).get("stop_loss"),
        take_profit=result.get("trade_proposal", {}).get("take_profit"),
        risk_assessment=result.get("risk_assessment"),
        var_value=result.get("risk_assessment", {}).get("var_value"),
        cvar_value=result.get("risk_assessment", {}).get("cvar_value"),
        approval_status=result.get("approval_decision", {}).get("status"),
        approval_reason=result.get("approval_decision", {}).get("rationale"),
    )
    db.add(record)
    await db.flush()

    # Store AI conclusion (AC-005)
    approval = result.get("approval_decision", {})
    if approval.get("rationale"):
        conclusion = AnalysisConclusion(
            id=str(uuid4()), scene="decision",
            reference_id=result["session_id"],
            symbol=req.symbol,
            conclusion_md=approval["rationale"],
            model_used="agent_workflow",
        )
        db.add(conclusion)

    return AgentRunResponse(
        session_id=result["session_id"],
        symbol=result["symbol"], name=result.get("name", ""),
        status=result["status"],
        fundamental_report=result.get("fundamental_report"),
        sentiment_report=result.get("sentiment_report"),
        news_report=result.get("news_report"),
        technical_report=result.get("technical_report"),
        debate_conclusion=result.get("debate_conclusion"),
        trade_proposal=result.get("trade_proposal"),
        risk_assessment=result.get("risk_assessment"),
        approval_decision=result.get("approval_decision"),
        errors=result.get("errors", []),
        completed_at=result.get("completed_at", ""),
    )


@router.get("/decisions", response_model=List[DecisionListResponse])
async def list_decisions(
    symbol: Optional[str] = Query(None),
    limit: int = Query(20, le=100),
    db: AsyncSession = Depends(get_db),
):
    """List past agent decisions."""
    logger.info(f"Listing agent decisions: symbol={symbol or 'all'}, limit={limit}")
    query = select(AgentDecision).order_by(
        AgentDecision.created_at.desc()
    ).limit(limit)
    if symbol:
        query = query.where(AgentDecision.symbol == symbol)

    result = await db.execute(query)
    items = result.scalars().all()
    logger.info(f"Agent decisions found: {len(items)}")
    return [
        DecisionListResponse(
            id=str(d.id), session_id=d.session_id,
            symbol=d.symbol, name=d.name,
            trade_side=d.trade_side,
            approval_status=d.approval_status,
            execution_status=d.execution_status,
            created_at=d.created_at.isoformat() if d.created_at else "",
        )
        for d in items
    ]


@router.get("/decisions/{session_id}")
async def get_decision(session_id: str, db: AsyncSession = Depends(get_db)):
    """Get detailed agent decision by session ID."""
    logger.info(f"Getting agent decision: session_id={session_id}")
    result = await db.execute(
        select(AgentDecision).where(AgentDecision.session_id == session_id)
    )
    decision = result.scalar_one_or_none()
    if not decision:
        logger.warning(f"Agent decision not found: session_id={session_id}")
        raise HTTPException(status_code=404, detail="Decision not found")
    logger.info(f"Agent decision retrieved: session_id={session_id}, symbol={decision.symbol}")
    return {
        "id": str(decision.id),
        "session_id": decision.session_id,
        "symbol": decision.symbol,
        "name": decision.name,
        "analyst_reports": decision.analyst_reports,
        "debate_conclusion": decision.debate_conclusion,
        "trade_side": decision.trade_side,
        "trade_quantity": decision.trade_quantity,
        "stop_loss": decision.stop_loss,
        "take_profit": decision.take_profit,
        "risk_assessment": decision.risk_assessment,
        "var_value": decision.var_value,
        "cvar_value": decision.cvar_value,
        "approval_status": decision.approval_status,
        "approval_reason": decision.approval_reason,
        "execution_status": decision.execution_status,
        "created_at": decision.created_at.isoformat() if decision.created_at else "",
    }
