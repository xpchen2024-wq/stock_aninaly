# ============================================================================
# AI Stock Analysis Platform - Investment Insight API
# ============================================================================
"""估值分位 / 财务质量 / 综合评分卡。

端点：
- GET /valuation  : PE/PB 五年分位、合理估值区间、股息率
- GET /quality    : 财务质量与成长评分（多报告期趋势）
- GET /scorecard  : 五维综合评分 + 可执行操作计划 + 风险提示
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List, Optional

import numpy as np
from fastapi import APIRouter, HTTPException, Query

from app.adapters import create_default_failover_manager, FailoverManager, DataType
from app.api.data import _fetch_quote_tencent, resolve_stock_symbol
from app.config import get_settings
from app.indicators import compute_all_indicators
from app.insight import (
    analyze_valuation,
    build_action_plan,
    build_cautions,
    build_narrative,
    build_risk_flags,
    compute_composite,
    compute_growth_score,
    compute_momentum_score,
    compute_quality_score,
    compute_risk_score,
    extract_financial_metrics,
    fetch_financial_abstract,
    safe_float,
)

logger = logging.getLogger(__name__)
router = APIRouter()
settings = get_settings()

_failover: Optional[FailoverManager] = None


def get_failover() -> FailoverManager:
    global _failover
    if _failover is None:
        _failover = create_default_failover_manager(tushare_token=settings.TUSHARE_TOKEN)
    return _failover


async def _code(symbol: str) -> str:
    """支持 6 位代码、带后缀代码与股票中文名。"""
    import re
    resolved = await resolve_stock_symbol(symbol)
    match = re.search(r"(\d{6})", resolved)
    if not match:
        raise HTTPException(status_code=404, detail=f"未找到股票: {symbol}")
    return match.group(1)


async def _resolve_price(symbol: str, lookback: int = 60) -> Dict[str, Any]:
    quote = await _fetch_quote_tencent(symbol)
    if quote and quote.get("price"):
        return {"price": float(quote["price"]), "name": quote.get("name", ""),
                "change_pct": float(quote.get("change_pct") or 0.0), "source": "tencent"}
    try:
        from datetime import datetime, timedelta
        end_date = datetime.now().strftime("%Y%m%d")
        start_date = (datetime.now() - timedelta(days=max(60, int(lookback * 1.6) + 30))).strftime("%Y%m%d")
        klines = await get_failover().fetch_with_failover(
            DataType.KLINE, symbol, frequency="D", start_date=start_date, end_date=end_date,
        )
        if klines:
            return {"price": float(klines[-1].close), "name": "", "change_pct": 0.0,
                    "source": "kline_close"}
    except Exception as exc:
        logger.warning(f"price fallback failed for {symbol}: {exc}")
    raise HTTPException(status_code=502, detail=f"无法获取 {symbol} 的实时价格")


@router.get("/valuation")
async def get_valuation(symbol: str = Query(..., description="股票代码或名称，如 300750.SZ / 宁德时代")):
    code = await _code(symbol)
    price_info = await _resolve_price(code)
    result = await analyze_valuation(code, price_info["price"])
    return {
        "symbol": code, "name": price_info["name"],
        "change_pct": price_info["change_pct"],
        "price_source": price_info["source"],
        **result.to_dict(),
    }


@router.get("/quality")
async def get_quality(symbol: str = Query(...)):
    code = await _code(symbol)
    rows = await fetch_financial_abstract(code)
    if not rows:
        raise HTTPException(status_code=502, detail="财务数据源暂不可用，请稍后重试")
    metrics = extract_financial_metrics(rows)
    if not metrics:
        raise HTTPException(status_code=422, detail="未能解析该股票的财务指标")
    quality = compute_quality_score(metrics)
    growth = compute_growth_score(metrics)
    return {
        "symbol": code,
        "latest_period": metrics.get("latest_period"),
        "metrics": {k: v for k, v in metrics.items() if k != "history"},
        "history": metrics.get("history", []),
        "quality": quality,
        "growth": growth,
    }


@router.get("/scorecard")
async def get_scorecard(symbol: str = Query(...), lookback: int = Query(250, ge=60, le=500)):
    code = await _code(symbol)
    price_info = await _resolve_price(code, lookback)
    price = price_info["price"]

    valuation, rows = await asyncio.gather(
        analyze_valuation(code, price), fetch_financial_abstract(code),
    )
    metrics = extract_financial_metrics(rows)
    quality = compute_quality_score(metrics) if metrics else {"score": None, "grade": "N/A", "components": {}}
    growth = compute_growth_score(metrics) if metrics else {"score": None, "grade": "N/A", "components": {}}

    momentum: Dict[str, Any] = {"score": None, "label": "数据不足", "components": {}}
    risk: Dict[str, Any] = {"score": None, "label": "数据不足", "components": {}}
    supports: List[float] = []
    resistances: List[float] = []
    ma60 = None
    try:
        from datetime import datetime, timedelta
        start_date = (datetime.now() - timedelta(days=int(lookback * 1.6) + 30)).strftime("%Y%m%d")
        klines = await get_failover().fetch_with_failover(
            DataType.KLINE, code, frequency="D", start_date=start_date,
            end_date=datetime.now().strftime("%Y%m%d"),
        )
        if klines:
            window = klines[-lookback:]
            closes = np.array([k.close for k in window], dtype=float)
            highs = np.array([k.high for k in window], dtype=float)
            lows = np.array([k.low for k in window], dtype=float)
            volumes = np.array([k.volume for k in window], dtype=float)
            indicators = compute_all_indicators(highs, lows, closes, volumes)
            momentum = compute_momentum_score(closes.tolist(), indicators)
            risk = compute_risk_score(closes.tolist())
            ma60 = safe_float(indicators.get("ma60"))
            if len(lows) >= 20:
                supports = [round(float(np.percentile(lows[-40:], 20)), 2)]
            if len(highs) >= 20:
                resistances = [round(float(np.percentile(highs[-40:], 80)), 2)]
    except Exception as exc:
        logger.warning(f"momentum/risk computation failed for {symbol}: {exc}")

    composite = compute_composite({
        "valuation": valuation.score if valuation else None,
        "quality": quality.get("score"),
        "growth": growth.get("score"),
        "momentum": momentum.get("score"),
        "risk": risk.get("score"),
    })
    plan = build_action_plan(
        price, valuation.fair_value, composite["score"], risk,
        supports, resistances, growth_score=growth.get("score"),
    )
    flags = build_risk_flags(valuation, metrics or {}, risk, price, ma60)
    divergence = bool(
        (valuation.score or 0) >= 70
        and (momentum.get("score") or 100) <= 40
    )
    cautions = build_cautions(valuation, momentum, composite, divergence)
    narrative = build_narrative(valuation, metrics or {}, quality, growth, momentum, risk, composite, plan)

    return {
        "symbol": code,
        "name": price_info["name"],
        "price": round(price, 2),
        "change_pct": price_info["change_pct"],
        "price_source": price_info["source"],
        "valuation": valuation.to_dict(),
        "metrics": {k: v for k, v in (metrics or {}).items() if k != "history"},
        "history": (metrics or {}).get("history", []),
        "quality": quality,
        "growth": growth,
        "momentum": momentum,
        "risk": risk,
        "composite": composite,
        "action_plan": plan,
        "risk_flags": flags,
        "cautions": cautions,
        "divergence": divergence,
        "narrative": narrative,
        "levels": {"support": supports, "resistance": resistances, "ma60": ma60},
    }
