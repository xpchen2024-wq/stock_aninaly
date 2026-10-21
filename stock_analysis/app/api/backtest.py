# ============================================================================
# AI Stock Analysis Platform - Strategy Backtest API
# ============================================================================
from __future__ import annotations

import logging
from datetime import date, timedelta
from typing import List, Optional
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.database import get_db
from app.adapters import create_default_failover_manager, DataType
from app.backtest import BacktestParams, run_backtest, SUPPORTED_STRATEGIES
from app.models import BacktestRecord

logger = logging.getLogger(__name__)
router = APIRouter()
settings = get_settings()

_failover = None


def get_failover():
    global _failover
    if _failover is None:
        _failover = create_default_failover_manager(tushare_token=settings.TUSHARE_TOKEN)
    return _failover


class BacktestRunRequest(BaseModel):
    symbol: str
    name: Optional[str] = ""
    strategy: str = "ma_cross"
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    initial_capital: float = Field(1_000_000, gt=0)
    position_limit_pct: float = Field(20.0, ge=1, le=100)
    fee_rate: float = Field(0.0003, ge=0, le=0.05)
    slippage_ticks: int = Field(1, ge=0)
    stop_loss_pct: float = Field(2.0, ge=0)
    take_profit_pct: float = Field(15.0, ge=0)


def _parse_date(value: Optional[str]) -> Optional[date]:
    if not value:
        return None
    try:
        return date.fromisoformat(str(value)[:10])
    except ValueError:
        return None


def _name_hint(symbol: str) -> str:
    """'300750.SZ 宁德时代' -> '宁德时代' (UI sends code + name together)."""
    parts = str(symbol or "").strip().split()
    return parts[1] if len(parts) > 1 else ""


@router.post("/run")
async def run_strategy_backtest(req: BacktestRunRequest, db: AsyncSession = Depends(get_db)):
    """Run a deterministic strategy backtest over real K-lines."""
    logger.info(f"Backtest started: {req.symbol} strategy={req.strategy} "
                f"range={req.start_date}~{req.end_date}")

    if req.strategy not in SUPPORTED_STRATEGIES:
        raise HTTPException(
            status_code=400,
            detail=f"不支持的策略: {req.strategy}（可选: {', '.join(SUPPORTED_STRATEGIES)}）",
        )

    # UI sends "300750.SZ 宁德时代" style input; extract the code part
    raw_symbol = req.symbol.strip().split()[0] if req.symbol.strip() else ""
    if not raw_symbol:
        raise HTTPException(status_code=400, detail="请提供股票代码")

    # Ask sources for the full requested window; without explicit dates the
    # adapter defaults to only the most recent ~90 days, which silently
    # breaks any older backtest range.
    fetch_start = req.start_date or (date.today() - timedelta(days=1100)).isoformat()
    fetch_end = req.end_date or date.today().isoformat()

    try:
        klines = await get_failover().fetch_with_failover(
            DataType.KLINE, raw_symbol,
            start_date=fetch_start.replace("-", ""),
            end_date=fetch_end.replace("-", ""),
        )
    except Exception as e:
        logger.warning(f"Backtest kline fetch failed for {raw_symbol}: {e}")
        raise HTTPException(status_code=502, detail=f"K线数据获取失败: {e}")

    if not klines:
        raise HTTPException(
            status_code=422,
            detail=f"数据源未返回 {raw_symbol} 的日K线（请检查代码是否正确，或该标的暂无数据）",
        )

    params = BacktestParams(
        symbol=raw_symbol, name=req.name or _name_hint(req.symbol),
        strategy=req.strategy,
        start_date=req.start_date, end_date=req.end_date,
        initial_capital=req.initial_capital,
        position_limit_pct=req.position_limit_pct,
        fee_rate=req.fee_rate,
        slippage_ticks=req.slippage_ticks,
        stop_loss_pct=req.stop_loss_pct,
        take_profit_pct=req.take_profit_pct,
    )
    try:
        result = run_backtest(klines, params)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    m = result["metrics"]
    record = BacktestRecord(
        id=str(uuid4()),
        symbol=result["symbol"], name=result.get("name") or req.name or "",
        strategy=result["strategy"],
        start_date=_parse_date(result["start_date"]),
        end_date=_parse_date(result["end_date"]),
        initial_capital=m["initial_capital"],
        final_equity=m["final_equity"],
        total_return_pct=m["total_return_pct"],
        annualized_return_pct=m["annualized_return_pct"],
        max_drawdown_pct=m["max_drawdown_pct"],
        sharpe_ratio=m["sharpe_ratio"],
        win_rate_pct=m["win_rate_pct"],
        trade_count=m["trade_count"],
        params_json={
            "fee_rate": req.fee_rate,
            "position_limit_pct": req.position_limit_pct,
            "slippage_ticks": req.slippage_ticks,
            "stop_loss_pct": req.stop_loss_pct,
            "take_profit_pct": req.take_profit_pct,
            "requested_start": req.start_date,
            "requested_end": req.end_date,
        },
    )
    db.add(record)
    await db.flush()
    result["record_id"] = str(record.id)

    logger.info(f"Backtest completed: {req.symbol} return={m['total_return_pct']}% "
                f"trades={m['trade_count']}")
    result = await _attach_benchmark(result, fetch_start, fetch_end)
    return {"status": "completed", "result": result}


async def _attach_benchmark(result: dict, fetch_start: str, fetch_end: str) -> dict:
    """Attach a 沪深300 buy&hold benchmark curve aligned to the backtest dates."""
    try:
        from app.api.forecast import bench_history

        bench = await bench_history()
        if bench is None or bench.empty or "close" not in bench:
            return result
        closes = bench["close"].astype(float)
        dates = [str(d)[:10] for d in bench["date"]]
        series = dict(zip(dates, closes))
        equity_dates = [p["date"] for p in result["equity_curve"]]
        base = series.get(equity_dates[0])
        if not base:
            return result
        initial = float(result["metrics"]["initial_capital"])
        curve = []
        for d in equity_dates:
            px = series.get(d)
            curve.append(round(initial * (px / base), 2) if px else None)
        bench_return = (curve[-1] / initial - 1) * 100 if curve[-1] else 0.0
        result["benchmark"] = {
            "code": "sh000300",
            "name": "沪深300",
            "total_return_pct": round(bench_return, 3),
            "excess_return_pct": round(result["metrics"]["total_return_pct"] - bench_return, 3),
            "curve": [{"date": d, "equity": v} for d, v in zip(equity_dates, curve)],
        }
    except Exception as exc:
        logger.warning(f"benchmark attach failed: {exc}")
    return result


@router.get("/strategies")
async def list_strategies():
    """Supported strategy ids with display names."""
    names = {
        "ma_cross": "均线交叉策略 (MA5/MA20)",
        "macd_cross": "MACD 金叉策略",
        "trend_follow": "AI 趋势跟踪策略 (MA20/MA60)",
    }
    return [{"id": s, "name": names.get(s, s)} for s in SUPPORTED_STRATEGIES]


@router.get("/list")
async def list_backtests(limit: int = Query(20, le=100), db: AsyncSession = Depends(get_db)):
    """List historical backtest runs."""
    rows = (await db.execute(
        select(BacktestRecord)
        .order_by(BacktestRecord.created_at.desc())
        .limit(limit)
    )).scalars().all()
    return [
        {
            "id": str(r.id),
            "symbol": r.symbol,
            "name": r.name,
            "strategy": r.strategy,
            "start_date": r.start_date.isoformat() if r.start_date else None,
            "end_date": r.end_date.isoformat() if r.end_date else None,
            "initial_capital": r.initial_capital,
            "final_equity": r.final_equity,
            "total_return_pct": r.total_return_pct,
            "annualized_return_pct": r.annualized_return_pct,
            "max_drawdown_pct": r.max_drawdown_pct,
            "sharpe_ratio": r.sharpe_ratio,
            "win_rate_pct": r.win_rate_pct,
            "trade_count": r.trade_count,
            "created_at": r.created_at.isoformat() if r.created_at else "",
        }
        for r in rows
    ]
