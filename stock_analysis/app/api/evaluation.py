# ============================================================================
# AI Stock Analysis Platform - Decision Outcome Evaluation API
# ============================================================================
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import List, Optional
from uuid import uuid4

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel
from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.database import get_db
from app.adapters import create_default_failover_manager, DataType
from app.models import AgentDecision, DecisionEvaluation

logger = logging.getLogger(__name__)
router = APIRouter()
settings = get_settings()

_failover = None


def get_failover():
    global _failover
    if _failover is None:
        _failover = create_default_failover_manager(tushare_token=settings.TUSHARE_TOKEN)
    return _failover


class EvaluationItem(BaseModel):
    session_id: str
    symbol: str
    name: str
    predicted_direction: str
    horizon_days: int
    baseline_date: str
    baseline_price: float
    target_date: str
    target_price: float
    actual_change_pct: float
    hit: bool
    approval_status: Optional[str]
    evaluated_at: str


class EvaluationSummary(BaseModel):
    total_evaluated: int
    hits: int
    misses: int
    hit_rate_pct: float
    avg_actual_change_pct: float
    by_direction: dict


def _parse_kline_date(value) -> Optional[datetime]:
    if isinstance(value, datetime):
        return _as_naive(value)
    if hasattr(value, "date"):
        return datetime.combine(value, datetime.min.time())
    for fmt in ("%Y-%m-%d", "%Y%m%d"):
        try:
            return datetime.strptime(str(value), fmt)
        except (ValueError, TypeError):
            continue
    return None


def _as_naive(value: datetime) -> datetime:
    """Normalize to naive UTC.

    The DB mixes `timestamp with time zone` (agent_decisions.created_at) and
    `timestamp without time zone` (decision_evaluations.evaluated_at), so
    asyncpg can hand back either aware or naive datetimes. Comparing them
    directly raises "can't subtract offset-naive and offset-aware datetimes",
    so everything is normalized to naive UTC before comparison.
    """
    if value.tzinfo is not None:
        return value.astimezone(timezone.utc).replace(tzinfo=None)
    return value


def _pick_kline_window(klines: list, decision_date, horizon_days: int):
    """Locate (baseline_kline, target_kline) around decision_date.

    baseline = last kline on/before decision date;
    target   = kline `horizon_days` trading bars after baseline.
    Returns (None, None) when insufficient data.
    """
    parsed = [(_parse_kline_date(k.date), k) for k in klines]
    parsed = [(d, k) for d, k in parsed if d is not None]
    if not parsed:
        return None, None
    parsed.sort(key=lambda pair: pair[0])
    decision_date = _as_naive(decision_date) if isinstance(decision_date, datetime) else decision_date
    baseline_idx = None
    for i, (k_date, _k) in enumerate(parsed):
        if k_date <= decision_date:
            baseline_idx = i
        else:
            break
    if baseline_idx is None:
        return None, None
    target_idx = baseline_idx + horizon_days
    if target_idx >= len(parsed):
        return None, None
    return parsed[baseline_idx][1], parsed[target_idx][1]


async def evaluate_pending_decisions(days: int, limit: int, db: AsyncSession) -> dict:
    """Score directional agent decisions against realized price movement."""
    from app.models import now_utc

    cutoff = now_utc()  # naive UTC — matches TIMESTAMP WITHOUT TIME ZONE columns
    result = await db.execute(
        select(AgentDecision)
        .where(AgentDecision.trade_side.in_(["buy", "sell"]))
        .where(AgentDecision.created_at < cutoff)
        .order_by(AgentDecision.created_at.desc())
        .limit(limit * 3)
    )
    candidates = result.scalars().all()

    evaluated_rows = await db.execute(
        select(DecisionEvaluation.session_id).limit(limit * 5)
    )
    already_done = {r[0] for r in evaluated_rows}

    created, skipped = [], []
    for d in candidates:
        if len(created) >= limit:
            break
        if d.session_id in already_done:
            continue
        try:
            klines = await get_failover().fetch_with_failover(DataType.KLINE, d.symbol)
            baseline, target = _pick_kline_window(klines, d.created_at, days)
        except Exception as e:
            logger.warning(f"Evaluation fetch failed for {d.symbol}: {e}")
            skipped.append({"session_id": d.session_id, "reason": f"data_fetch_failed: {e}"})
            continue

        if baseline is None or not baseline.price or not target.price:
            skipped.append({"session_id": d.session_id, "reason": "insufficient_kline_history"})
            continue

        change_pct = (target.price - baseline.price) / baseline.price * 100
        predicted_up = d.trade_side == "buy"
        actual_up = change_pct > 0

        record = DecisionEvaluation(
            id=str(uuid4()),
            session_id=d.session_id,
            symbol=d.symbol,
            name=d.name,
            predicted_direction=d.trade_side,
            horizon_days=days,
            baseline_date=_parse_kline_date(baseline.date).date(),
            baseline_price=float(baseline.price),
            target_date=_parse_kline_date(target.date).date(),
            target_price=float(target.price),
            actual_change_pct=round(change_pct, 4),
            hit=predicted_up == actual_up,
            approval_status=d.approval_status,
        )
        db.add(record)
        created.append(record)

    if created:
        await db.flush()
    logger.info(f"Decision evaluation: created={len(created)}, skipped={len(skipped)}")
    return {"created": len(created), "skipped": skipped[:20]}


@router.post("/run")
async def run_evaluation(
    days: int = Query(5, ge=1, le=60, description="Trading-day horizon for outcome check"),
    limit: int = Query(50, ge=1, le=200),
    db: AsyncSession = Depends(get_db),
):
    """Evaluate pending directional decisions against real price movement."""
    logger.info(f"Manual evaluation run: days={days}, limit={limit}")
    summary = await evaluate_pending_decisions(days, limit, db)
    return {"status": "ok", **summary}


@router.get("/list", response_model=List[EvaluationItem])
async def list_evaluations(
    symbol: Optional[str] = Query(None),
    limit: int = Query(50, le=200),
    db: AsyncSession = Depends(get_db),
):
    query = select(DecisionEvaluation).order_by(
        DecisionEvaluation.evaluated_at.desc()
    ).limit(limit)
    if symbol:
        query = query.where(DecisionEvaluation.symbol == symbol)
    rows = (await db.execute(query)).scalars().all()
    return [
        EvaluationItem(
            session_id=e.session_id, symbol=e.symbol, name=e.name,
            predicted_direction=e.predicted_direction,
            horizon_days=e.horizon_days,
            baseline_date=e.baseline_date.isoformat(),
            baseline_price=e.baseline_price,
            target_date=e.target_date.isoformat(),
            target_price=e.target_price,
            actual_change_pct=e.actual_change_pct,
            hit=e.hit,
            approval_status=e.approval_status,
            evaluated_at=e.evaluated_at.isoformat() if e.evaluated_at else "",
        )
        for e in rows
    ]


@router.get("/summary", response_model=EvaluationSummary)
async def evaluation_summary(db: AsyncSession = Depends(get_db)):
    """Aggregate hit-rate stats for the agent pipeline."""
    rows = (await db.execute(select(DecisionEvaluation))).scalars().all()
    total = len(rows)
    if total == 0:
        return EvaluationSummary(
            total_evaluated=0, hits=0, misses=0, hit_rate_pct=0.0,
            avg_actual_change_pct=0.0, by_direction={},
        )

    hits = sum(1 for r in rows if r.hit)
    by_dir: dict = {}
    for r in rows:
        bucket = by_dir.setdefault(r.predicted_direction, {"total": 0, "hits": 0})
        bucket["total"] += 1
        if r.hit:
            bucket["hits"] += 1

    return EvaluationSummary(
        total_evaluated=total,
        hits=hits,
        misses=total - hits,
        hit_rate_pct=round(hits / total * 100, 2),
        avg_actual_change_pct=round(sum(r.actual_change_pct for r in rows) / total, 4),
        by_direction={
            k: {**v, "hit_rate_pct": round(v["hits"] / v["total"] * 100, 2)}
            for k, v in by_dir.items()
        },
    )
