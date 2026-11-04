# ============================================================================
# 投研圆桌 API · TradingAgents 风格多智能体
# ============================================================================
"""HTTP 接口：

- POST /api/v1/roundtable/run/stream   SSE 流式跑一场圆桌（分析师→辩论→终审）
- GET  /api/v1/roundtable/runs         历史圆桌列表
- GET  /api/v1/roundtable/runs/{sid}   单场圆桌完整实录
- GET  /api/v1/roundtable/memory       决策记忆与复盘（可按标的过滤）
- POST /api/v1/roundtable/memory/settle 手动结算到期决策
- GET  /api/v1/roundtable/meta         默认配置与模型档位状态
"""
from __future__ import annotations

import json
import logging
import re
from datetime import date, datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.adapters import FailoverManager, create_default_failover_manager
from app.config import get_settings
from app.database import get_db
from app.model_gateway import get_model_gateway
from app.models import (
    AgentModelBinding, KOLOpinion, ModelConfig, RoundtableMemory, RoundtableRun,
    now_utc,
)
from app.roundtable.graph import RoundtableGraph
from app.roundtable.memory import (
    load_memory_context, save_decision, serialize_memory, settle_due,
)
from app.roundtable.state import RoundtableConfig

logger = logging.getLogger(__name__)
router = APIRouter()
settings = get_settings()

_gateway = None
_failover: Optional[FailoverManager] = None

SCENE_DEEP = "roundtable_deep"
SCENE_QUICK = "roundtable_quick"


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


# -- Schemas ------------------------------------------------------------------
class RoundtableRunRequest(BaseModel):
    symbol: str
    name: Optional[str] = ""
    trade_date: Optional[str] = None
    max_debate_rounds: int = 2
    max_risk_rounds: int = 1
    horizon_days: int = 10
    portfolio_note: Optional[str] = ""


class HistoryItem(BaseModel):
    session_id: str
    symbol: str
    name: str
    trade_date: Optional[str]
    status: str
    signal: Optional[str]
    confidence: Optional[float]
    elapsed_ms: Optional[int]
    created_at: str


# -- Helpers ------------------------------------------------------------------
async def _resolve_symbol(raw: str) -> tuple[str, str]:
    """名称/代码 → (6位标准代码, 名称)。"""
    from app.api.data import resolve_stock_symbol
    from app.stocks import get_stock_list, search_in_list

    text = (raw or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="请输入股票代码或名称")

    # resolve_stock_symbol 对未找到会抛 404/504，直接透传给前端
    resolved = await resolve_stock_symbol(text)
    m = re.search(r"(\d{6})", str(resolved))
    code = m.group(1) if m else str(resolved)

    name = ""
    try:
        hits = search_in_list(await get_stock_list(), code, limit=1)
        if hits:
            name = hits[0].get("name", "")
    except Exception as exc:
        logger.info(f"[roundtable] name lookup skipped for {code}: {exc}")
    return code, name


async def _resolve_tiers(db: AsyncSession) -> tuple[Optional[str], Optional[str]]:
    """把「深思考 / 快思考」两个档位映射到已绑定且启用的模型配置。

    优先读专门为圆桌配置的 scene 绑定；没有就退回到项目里已有的
    decision（深）/ trend_analysis（快）场景，最后交给网关默认模型。
    """
    rows = (await db.execute(
        select(AgentModelBinding).where(AgentModelBinding.is_active.is_(True))
    )).scalars().all()
    by_scene = {r.scene: str(r.model_config_id) for r in rows if r.scene}

    enabled_ids = set((await db.execute(
        select(ModelConfig.id).where(ModelConfig.is_enabled.is_(True))
    )).scalars().all())

    def _pick(*scenes: str) -> Optional[str]:
        for s in scenes:
            cid = by_scene.get(s)
            if cid and cid in enabled_ids:
                return cid
        return None

    deep = _pick(SCENE_DEEP, "decision", "serenity")
    quick = _pick(SCENE_QUICK, "trend_analysis", "report", "decision")
    return deep, quick


async def _kol_context(db: AsyncSession, code: str, limit: int = 8,
                       as_of: Optional[datetime] = None) -> str:
    """近期提到该标的的大V观点，作为情绪分析师的社媒输入（A 股替代 StockTwits）。"""
    end = as_of or now_utc()
    cutoff = end - timedelta(days=7)
    rows = (await db.execute(
        select(KOLOpinion)
        .where(KOLOpinion.published_at >= cutoff, KOLOpinion.published_at < end,
               KOLOpinion.collected_at < end)
        .order_by(KOLOpinion.heat_score.desc().nullslast())
        .limit(120)
    )).scalars().all()

    hits: List[str] = []
    for op in rows:
        stocks = op.mentioned_stocks or []
        if not isinstance(stocks, list):
            continue
        for s in stocks:
            if not isinstance(s, dict):
                continue
            sym = str(s.get("code") or s.get("symbol") or "")
            if code and code in sym:
                direction = s.get("direction") or op.direction or "neutral"
                hits.append(
                    f"- [{op.published_at:%m-%d}] {op.platform} 大V观点（{direction}）："
                    f"{(op.summary or '')[:120]}"
                )
                break
        if len(hits) >= limit:
            break

    if not hits:
        return "### 社媒/大V观点\n近 7 日无该标的的大V观点数据。\n"
    return "### 社媒/大V观点（近7日）\n" + "\n".join(hits) + "\n"


async def _persist_run(db: AsyncSession, result: Dict[str, Any]) -> None:
    decision = result.get("decision") or {}
    record = RoundtableRun(
        session_id=result["session_id"],
        symbol=result["symbol"],
        name=result.get("name") or result["symbol"],
        trade_date=_parse_date(result.get("trade_date")),
        status=result.get("status", "completed"),
        config=result.get("config"),
        snapshot=result.get("snapshot"),
        analyst_reports=result.get("analyst_reports"),
        research_debate=result.get("research_debate"),
        trader_plan=result.get("trader_plan"),
        risk_debate=result.get("risk_debate"),
        decision=decision,
        signal=decision.get("action"),
        confidence=decision.get("confidence"),
        transcript=result.get("transcript"),
        state={
            "checkpoints": result.get("checkpoints"),
            "memory_stats": result.get("memory_stats"),
            "errors": result.get("errors"),
            "degraded": result.get("degraded"),
        },
        errors=result.get("errors"),
        elapsed_ms=result.get("elapsed_ms"),
        completed_at=now_utc(),
    )
    db.add(record)

    trade_date = record.trade_date or date.today()
    await save_decision(
        db, session_id=result["session_id"], symbol=result["symbol"],
        name=record.name, trade_date=trade_date, decision=decision,
        entry_price=(result.get("snapshot") or {}).get("price"),
        horizon_days=int((result.get("config") or {}).get("horizon_days") or 10),
    )
    await db.flush()


def _parse_date(value: Any) -> Optional[date]:
    if isinstance(value, date):
        return value
    if not value:
        return None
    try:
        return datetime.strptime(str(value)[:10], "%Y-%m-%d").date()
    except ValueError:
        return None


def _serialize_run(row: RoundtableRun) -> Dict[str, Any]:
    return {
        "session_id": row.session_id,
        "symbol": row.symbol,
        "name": row.name,
        "trade_date": row.trade_date.isoformat() if row.trade_date else None,
        "status": row.status,
        "config": row.config,
        "snapshot": row.snapshot,
        "analyst_reports": row.analyst_reports,
        "research_debate": row.research_debate,
        "trader_plan": row.trader_plan,
        "risk_debate": row.risk_debate,
        "decision": row.decision,
        "transcript": row.transcript,
        "errors": row.errors,
        "elapsed_ms": row.elapsed_ms,
        "created_at": row.created_at.isoformat() if row.created_at else None,
        "completed_at": row.completed_at.isoformat() if row.completed_at else None,
    }


# -- Routes -------------------------------------------------------------------
@router.get("/meta")
async def get_meta(db: AsyncSession = Depends(get_db)):
    deep, quick = await _resolve_tiers(db)

    async def _model_name(cid: Optional[str]) -> Optional[str]:
        if not cid:
            return None
        cfg = (await db.execute(
            select(ModelConfig).where(ModelConfig.id == cid)
        )).scalar_one_or_none()
        return cfg.name if cfg else None

    memory_count = (await db.execute(
        select(func.count()).select_from(RoundtableMemory)
    )).scalar() or 0
    settled = (await db.execute(
        select(func.count()).select_from(RoundtableMemory)
        .where(RoundtableMemory.settled.is_(True))
    )).scalar() or 0

    return {
        "defaults": {
            "max_debate_rounds": 2,
            "max_risk_rounds": 1,
            "horizon_days": 10,
        },
        "limits": {
            "max_debate_rounds": [1, 3],
            "max_risk_rounds": [1, 2],
            "horizon_days": [3, 20],
        },
        "models": {
            "deep": await _model_name(deep),
            "quick": await _model_name(quick),
            "fallback": await _model_name(deep) is None,
        },
        "memory": {"total": int(memory_count), "settled": int(settled)},
        "pipeline": [
            "分析师团队（4）", "多空辩论", "研究经理裁决",
            "交易计划", "三方风控辩论", "组合经理终审", "记忆反思",
        ],
    }


@router.post("/run/stream")
async def run_roundtable_stream(req: RoundtableRunRequest,
                                db: AsyncSession = Depends(get_db)):
    """SSE 跑一场完整圆桌，逐节点推送进度与发言。"""
    symbol, resolved_name = await _resolve_symbol(req.symbol)
    name = (req.name or "").strip() or resolved_name or symbol
    code = "".join(ch for ch in symbol if ch.isdigit())[:6]

    config = RoundtableConfig(
        max_debate_rounds=req.max_debate_rounds,
        max_risk_rounds=req.max_risk_rounds,
        horizon_days=req.horizon_days,
        portfolio_note=req.portfolio_note or "",
    ).clamp()
    deep, quick = await _resolve_tiers(db)
    config.deep_model_id = deep
    config.quick_model_id = quick

    today = datetime.now(timezone(timedelta(hours=8))).date()
    requested_date = _parse_date(req.trade_date) if req.trade_date else today
    if requested_date is None or requested_date > today:
        raise HTTPException(status_code=400, detail="分析日期须为有效的非未来日期")
    trade_date = requested_date.isoformat()
    as_of = None
    if requested_date < today:
        as_of = datetime.combine(requested_date + timedelta(days=1), datetime.min.time())
        as_of -= timedelta(hours=8)
    logger.info(
        f"[roundtable] start symbol={symbol} name={name} trade_date={trade_date} "
        f"debate_rounds={config.max_debate_rounds} risk_rounds={config.max_risk_rounds} "
        f"deep={deep} quick={quick}"
    )

    async def event_generator():
        result: Optional[Dict[str, Any]] = None
        try:
            try:
                kol = await _kol_context(db, code, as_of=as_of)
            except Exception as exc:
                logger.warning(f"[roundtable] kol context failed: {exc}")
                kol = ""
            try:
                memory_text = await load_memory_context(db, symbol, as_of=as_of)
            except Exception as exc:
                logger.warning(f"[roundtable] memory context failed: {exc}")
                memory_text = ""

            graph = RoundtableGraph(get_gateway(), get_failover(), config)

            async def _settle():
                return await settle_due(db, get_failover(), get_gateway(), config)

            async def _refresh():
                return await load_memory_context(db, symbol, as_of=as_of)

            async for kind, payload in graph.run_stream(
                symbol=symbol, name=name, trade_date=trade_date,
                kol_context=kol, memory_text=memory_text,
                settle=_settle, refresh_memory=_refresh,
            ):
                if kind == "result":
                    result = payload
                    try:
                        await _persist_run(db, result)
                        await db.commit()
                    except Exception as exc:
                        logger.error(f"[roundtable] persist failed: {exc}", exc_info=True)
                        result.setdefault("errors", []).append(f"结果落库失败：{exc}")
                    yield f"event: result\ndata: {json.dumps(result, ensure_ascii=False, default=str)}\n\n"
                else:
                    yield f"event: {kind}\ndata: {json.dumps(payload, ensure_ascii=False, default=str)}\n\n"

            if result is None:
                raise RuntimeError("圆桌结束但没有产出结果")
        except Exception as exc:
            import traceback
            logger.error(f"[roundtable] stream error: {traceback.format_exc()}")
            yield (
                "event: error\ndata: "
                + json.dumps({"message": str(exc) or "圆桌运行异常"}, ensure_ascii=False)
                + "\n\n"
            )

    return _sse_response(event_generator())


def _sse_response(generator):
    from fastapi.responses import StreamingResponse
    return StreamingResponse(
        generator,
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/runs", response_model=List[HistoryItem])
async def list_runs(symbol: Optional[str] = Query(None),
                    limit: int = Query(20, ge=1, le=100),
                    db: AsyncSession = Depends(get_db)):
    query = select(RoundtableRun).order_by(RoundtableRun.created_at.desc()).limit(limit)
    if symbol:
        query = query.where(RoundtableRun.symbol == symbol)
    rows = (await db.execute(query)).scalars().all()
    return [
        HistoryItem(
            session_id=r.session_id, symbol=r.symbol, name=r.name,
            trade_date=r.trade_date.isoformat() if r.trade_date else None,
            status=r.status, signal=r.signal, confidence=r.confidence,
            elapsed_ms=r.elapsed_ms,
            created_at=r.created_at.isoformat() if r.created_at else "",
        )
        for r in rows
    ]


@router.get("/runs/{session_id}")
async def get_run(session_id: str, db: AsyncSession = Depends(get_db)):
    row = (await db.execute(
        select(RoundtableRun).where(RoundtableRun.session_id == session_id)
    )).scalar_one_or_none()
    if not row:
        raise HTTPException(status_code=404, detail="圆桌记录不存在")
    return _serialize_run(row)


@router.get("/memory")
async def list_memory(symbol: Optional[str] = Query(None),
                      limit: int = Query(20, ge=1, le=100),
                      db: AsyncSession = Depends(get_db)):
    query = select(RoundtableMemory).order_by(
        RoundtableMemory.trade_date.desc()
    ).limit(limit)
    if symbol:
        query = query.where(RoundtableMemory.symbol == symbol)
    rows = (await db.execute(query)).scalars().all()
    settled = [r for r in rows if r.settled and r.realized_return_pct is not None]
    hit = sum(
        1 for r in settled
        if (r.signal == "buy" and (r.realized_return_pct or 0) > 0)
        or (r.signal == "sell" and (r.realized_return_pct or 0) < 0)
    )
    return {
        "items": [serialize_memory(r) for r in rows],
        "stats": {
            "total": len(rows),
            "settled": len(settled),
            "hit": hit,
            "hit_rate_pct": round(hit / len(settled) * 100, 1) if settled else None,
            "avg_return_pct": round(
                sum(r.realized_return_pct or 0 for r in settled) / len(settled), 2
            ) if settled else None,
            "avg_alpha_pct": round(
                sum(r.alpha_pct or 0 for r in settled if r.alpha_pct is not None)
                / max(len([r for r in settled if r.alpha_pct is not None]), 1), 2
            ) if any(r.alpha_pct is not None for r in settled) else None,
        },
    }


@router.post("/memory/settle")
async def settle_memory(limit: int = Query(5, ge=1, le=20),
                        db: AsyncSession = Depends(get_db)):
    config = RoundtableConfig(horizon_days=10)
    deep, quick = await _resolve_tiers(db)
    config.deep_model_id = deep
    config.quick_model_id = quick
    stats = await settle_due(db, get_failover(), get_gateway(), config, limit=limit)
    await db.commit()
    logger.info(f"[roundtable] manual settle: {stats}")
    return stats
