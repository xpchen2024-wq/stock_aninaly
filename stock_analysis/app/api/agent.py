# ============================================================================
# AI Stock Analysis Platform - Agent Decision API (AGENT-xxx)
# ============================================================================
from __future__ import annotations

import json
import logging
from typing import Optional, List
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import PlainTextResponse, StreamingResponse
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
    fallback_detailed_report: Optional[dict] = None
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


# -- Persistence & report helpers --
async def _persist_decision(db: AsyncSession, result: dict) -> None:
    """Store the workflow outcome as an AgentDecision row (+ decision conclusion)."""
    proposal = result.get("trade_proposal") or {}
    risk = result.get("risk_assessment") or {}
    approval = result.get("approval_decision") or {}

    record = AgentDecision(
        id=str(uuid4()),
        session_id=result["session_id"],
        symbol=result["symbol"], name=result.get("name", ""),
        analyst_reports={
            "fundamental": result.get("fundamental_report"),
            "sentiment": result.get("sentiment_report"),
            "news": result.get("news_report"),
            "technical": result.get("technical_report"),
        },
        debate_conclusion=result.get("debate_conclusion"),
        trade_side=proposal.get("side"),
        trade_quantity=proposal.get("quantity"),
        stop_loss=proposal.get("stop_loss"),
        take_profit=proposal.get("take_profit"),
        risk_assessment=risk,
        var_value=risk.get("var_value"),
        cvar_value=risk.get("cvar_value"),
        approval_status=approval.get("status"),
        approval_reason=approval.get("rationale"),
    )
    db.add(record)
    await db.flush()

    if approval.get("rationale"):
        conclusion = AnalysisConclusion(
            id=str(uuid4()), scene="decision",
            reference_id=result["session_id"],
            symbol=result["symbol"],
            conclusion_md=approval["rationale"],
            model_used="agent_workflow",
        )
        db.add(conclusion)


def _build_decision_markdown(decision: AgentDecision) -> str:
    """Render a stored AgentDecision as a structured Chinese markdown report."""
    d = decision
    lines: List[str] = []
    lines.append(f"# AI 多智能体分析报告：{d.name}（{d.symbol}）")
    lines.append("")
    lines.append(f"- 会话 ID: `{d.session_id}`")
    lines.append(f"- 生成时间: {d.created_at.isoformat() if d.created_at else ''}")
    lines.append("")

    reports = d.analyst_reports or {}
    lines.append("## 一、分析师团队摘要")
    lines.append("")
    label_map = {
        "fundamental": "基本面", "sentiment": "情绪",
        "news": "新闻影响", "technical": "技术面",
    }
    for key, label in label_map.items():
        r = reports.get(key)
        if not r:
            lines.append(f"### {label}分析师\n\n- 数据缺失或分析失败")
            lines.append("")
            continue
        conf = r.get("confidence")
        lines.append(f"### {label}分析师（置信度 {conf if conf is not None else 'N/A'}）")
        if key == "fundamental":
            scores = r.get("scores") or {}
            lines.append(f"- 综合评分: {scores.get('overall', 'N/A')}")
            flags = r.get("red_flags") or []
            if flags:
                lines.append(f"- 风险信号: {'；'.join(map(str, flags))}")
            rng = r.get("intrinsic_value_range") or {}
            if rng:
                lines.append(f"- 内在价值区间: {rng.get('low')} ~ {rng.get('high')}")
        elif key == "sentiment":
            lines.append(f"- 情绪得分: {r.get('sentiment_score', 'N/A')}"
                         f"（趋势: {r.get('sentiment_trend', 'N/A')}）")
            drivers = r.get("key_drivers") or []
            if drivers:
                lines.append(f"- 关键驱动: {'；'.join(map(str, drivers))}")
        elif key == "news":
            lines.append(f"- 事件摘要: {r.get('event_summary', '') or '无'}")
            lines.append(f"- 影响得分: {r.get('impact_score', 'N/A')}")
        elif key == "technical":
            lines.append(f"- 趋势方向: {r.get('trend_direction', 'N/A')}")
            patterns = r.get("patterns") or []
            if patterns:
                lines.append(f"- 技术形态: {'、'.join(map(str, patterns))}")
            sup = r.get("support_levels") or []
            res = r.get("resistance_levels") or []
            level_bits = []
            if sup:
                level_bits.append(f"支撑 {sup}")
            if res:
                level_bits.append(f"阻力 {res}")
            if level_bits:
                lines.append(f"- 关键点位: {'，'.join(level_bits)}")
        lines.append("")

    debate = d.debate_conclusion or {}
    lines.append("## 二、多空辩论")
    lines.append("")
    lines.append(f"- 最终立场: **{debate.get('final_stance', 'N/A')}**"
                 f"（信心水平 {debate.get('conviction_level', 'N/A')}）")
    bull = debate.get("bull_arguments") or []
    bear = debate.get("bear_arguments") or []
    if bull:
        lines.append("- 看涨论点:")
        lines.extend(f"  - {a}" for a in bull)
    if bear:
        lines.append("- 看跌/风险论点:")
        lines.extend(f"  - {a}" for a in bear)
    risks = debate.get("key_risks") or []
    if risks:
        lines.append(f"- 关键风险: {'；'.join(map(str, risks))}")
    lines.append("")

    risk = d.risk_assessment or {}
    lines.append("## 三、交易提案")
    lines.append("")
    lines.append(f"- 方向: **{d.trade_side or 'hold'}**，"
                 f"数量: {d.trade_quantity if d.trade_quantity is not None else '—'} 股")
    lines.append(f"- 止损: {d.stop_loss if d.stop_loss is not None else '—'}，"
                 f"止盈: {d.take_profit if d.take_profit is not None else '—'}")
    lines.append("")
    lines.append("## 四、风险评估")
    lines.append("")
    lines.append(f"- 风险等级: {risk.get('risk_level', 'N/A')}")
    lines.append(f"- VaR: {d.var_value if d.var_value is not None else '—'}，"
                 f"CVaR: {d.cvar_value if d.cvar_value is not None else '—'}")
    recs = risk.get("recommendations") or []
    if recs:
        lines.append("- 建议:")
        lines.extend(f"  - {r}" for r in recs)
    lines.append("")

    lines.append("## 五、审批结论")
    lines.append("")
    lines.append(f"- 审批状态: **{d.approval_status or 'N/A'}**")
    if d.approval_reason:
        lines.append("")
        lines.append(f"> {d.approval_reason}")
    lines.append("")
    lines.append("---")
    lines.append("*本报告由多智能体系统自动生成，仅供研究参考，不构成投资建议。*")
    return "\n".join(lines)


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
    await _persist_decision(db, result)

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
        fallback_detailed_report=result.get("fallback_detailed_report"),
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


@router.post("/run/stream")
async def run_agent_workflow_stream(req: AgentRunRequest, db: AsyncSession = Depends(get_db)):
    """Run the agent workflow with SSE progress events (phase-by-phase)."""
    logger.info(f"Agent workflow stream started: symbol={req.symbol}, name={req.name or 'N/A'}")

    async def event_generator():
        try:
            workflow = AgentWorkflow(get_gateway(), get_failover())
            result = None
            async for kind, payload in workflow.run_stream(
                symbol=req.symbol, name=req.name or "",
                fundamentals=req.fundamentals, news_items=req.news_items,
            ):
                if kind == "result":
                    result = payload
                    await _persist_decision(db, result)
                    yield f"event: result\ndata: {json.dumps(payload, ensure_ascii=False, default=str)}\n\n"
                else:
                    yield f"event: phase\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"
            if result is None:
                raise RuntimeError("workflow finished without a result event")
        except Exception as e:
            import traceback
            logger.error(f"Agent stream error: {traceback.format_exc()}")
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


@router.get("/decisions/{session_id}/report", response_class=PlainTextResponse)
async def export_decision_report(session_id: str, db: AsyncSession = Depends(get_db)):
    """Export a stored agent decision as a markdown report."""
    logger.info(f"Exporting decision report: session_id={session_id}")
    result = await db.execute(
        select(AgentDecision).where(AgentDecision.session_id == session_id)
    )
    decision = result.scalar_one_or_none()
    if not decision:
        raise HTTPException(status_code=404, detail="Decision not found")

    markdown = _build_decision_markdown(decision)
    return PlainTextResponse(
        content=markdown,
        media_type="text/markdown; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="decision_{session_id}.md"'},
    )
