# ============================================================================
# AI Stock Analysis Platform - Investment Insight Engine
# ============================================================================
"""估值分位、财务质量、动量/风险与综合评分卡。

设计原则与 app/risk_metrics.py 一致：所有分数与买卖价位都由确定性公式从
真实数据算出，LLM 只负责叙述，不参与数值生成。

数据源（均无需额外依赖）：
- stock_zh_valuation_baidu        : PE(TTM)/PB 五年历史 → 分位与合理估值区间
- stock_financial_abstract        : ROE/毛利率/增速/负债率等财务指标（多报告期）
- stock_history_dividend_detail   : 分红明细（派息单位为每 10 股）
- FailoverManager K 线            : 动量与风险指标
"""
from __future__ import annotations

import asyncio
import logging
import math
import re
import time
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from app.risk_metrics import simple_returns

logger = logging.getLogger(__name__)


# -- Helpers ------------------------------------------------------------------
def safe_float(value: Any) -> Optional[float]:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(f) or math.isinf(f):
        return None
    return f


def clamp(value: float, low: float = 0.0, high: float = 100.0) -> float:
    return max(low, min(high, value))


def percentile_rank(values: List[float], current: float) -> Optional[float]:
    """Share of history at or below `current`, as 0-100."""
    clean = [v for v in (safe_float(x) for x in values) if v is not None]
    if not clean or current is None:
        return None
    below = sum(1 for v in clean if v <= current)
    return round(below / len(clean) * 100, 1)


def quantile(values: List[float], q: float) -> Optional[float]:
    clean = [v for v in (safe_float(x) for x in values) if v is not None]
    if not clean:
        return None
    return round(float(np.quantile(clean, q)), 4)


# -- Valuation ----------------------------------------------------------------
VALUATION_WEIGHTS = {"pe_percentile": 30.0, "pb_percentile": 20.0, "dividend": 15.0, "absolute": 35.0}


@dataclass
class ValuationResult:
    current_price: float = 0.0
    pe: Optional[float] = None
    pb: Optional[float] = None
    pe_percentile: Optional[float] = None
    pb_percentile: Optional[float] = None
    pe_band: Dict[str, float] = field(default_factory=dict)
    pb_band: Dict[str, float] = field(default_factory=dict)
    pe_series: List[Dict[str, Any]] = field(default_factory=list)
    pb_series: List[Dict[str, Any]] = field(default_factory=list)
    fair_value: Dict[str, Any] = field(default_factory=dict)
    dividend_ttm_per_share: float = 0.0
    dividend_yield_pct: Optional[float] = None
    dividend_years: int = 0
    score: float = 50.0
    label: str = "数据不足"
    reasons: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "current_price": self.current_price,
            "pe": self.pe,
            "pb": self.pb,
            "pe_percentile": self.pe_percentile,
            "pb_percentile": self.pb_percentile,
            "pe_band": self.pe_band,
            "pb_band": self.pb_band,
            "pe_series": self.pe_series,
            "pb_series": self.pb_series,
            "fair_value": self.fair_value,
            "dividend_ttm_per_share": self.dividend_ttm_per_share,
            "dividend_yield_pct": self.dividend_yield_pct,
            "dividend_years": self.dividend_years,
            "score": self.score,
            "label": self.label,
            "reasons": self.reasons,
        }


def compute_valuation_score(pe: Optional[float], pb: Optional[float],
                            pe_pct: Optional[float], pb_pct: Optional[float],
                            dividend_yield: Optional[float]) -> Tuple[float, str, List[str]]:
    """Blend percentile, absolute level and dividend into a 0-100 valuation score."""
    score = 50.0
    reasons: List[str] = []

    if pe is None or pe <= 0:
        score -= 12
        reasons.append("PE 为负或缺失（亏损/微利），估值判断主要依赖 PB")
    else:
        if pe_pct is not None:
            score += (50 - pe_pct) * 0.45
            if pe_pct <= 20:
                reasons.append(f"PE {pe:.1f}x 处于近五年 {pe_pct:.0f}% 分位，历史低估区间")
            elif pe_pct >= 80:
                reasons.append(f"PE {pe:.1f}x 处于近五年 {pe_pct:.0f}% 分位，历史高位")
            else:
                reasons.append(f"PE {pe:.1f}x 处于近五年 {pe_pct:.0f}% 分位")
        if pe < 12:
            score += 10
        elif pe < 20:
            score += 5
        elif pe > 60:
            score -= 12
        elif pe > 40:
            score -= 6

    if pb is not None and pb > 0:
        if pb_pct is not None:
            score += (50 - pb_pct) * 0.30
        if pb < 1:
            score += 6
        elif pb > 10:
            score -= 10
        elif pb > 6:
            score -= 5

    if dividend_yield is not None and dividend_yield > 0:
        score += min(12.0, dividend_yield * 3.0)
        if dividend_yield >= 3:
            reasons.append(f"股息率 {dividend_yield:.2f}%，现金回报较好")
    else:
        score -= 3
        reasons.append("近 12 个月无现金分红，缺少股息安全垫")

    score = round(clamp(score), 1)
    if score >= 75:
        label = "低估"
    elif score >= 60:
        label = "偏低"
    elif score >= 42:
        label = "合理"
    elif score >= 28:
        label = "偏高"
    else:
        label = "高估"
    return score, label, reasons


def build_fair_value(price: float, current_multiple: Optional[float],
                     band: Dict[str, float], kind: str) -> Optional[Dict[str, Any]]:
    """Scale price by historical multiple quartiles to obtain a fair-value range."""
    if not price or price <= 0 or not current_multiple or current_multiple <= 0:
        return None
    p25, p50, p75 = band.get("p25"), band.get("p50"), band.get("p75")
    if not p25 or not p50 or not p75:
        return None
    low = price * p25 / current_multiple
    mid = price * p50 / current_multiple
    high = price * p75 / current_multiple
    upside = (mid / price - 1) * 100
    return {
        "basis": kind,
        "low": round(low, 2),
        "mid": round(mid, 2),
        "high": round(high, 2),
        "upside_pct": round(upside, 2),
    }


def compute_ttm_dividend(rows: List[Dict[str, Any]], price: float) -> Dict[str, Any]:
    """TTM dividend from Sina rows where 派息 is per 10 shares."""
    today = date.today()
    per_share: List[Tuple[date, float]] = []
    years = set()
    for row in rows or []:
        raw_date = row.get("除权除息日") or row.get("公告日期")
        payout = safe_float(row.get("派息"))
        if payout is None or payout <= 0:
            continue
        d = _parse_date(raw_date)
        if not d:
            continue
        years.add(d.year)
        per_share.append((d, payout / 10.0))

    ttm = sum(v for d, v in per_share if (today - d).days <= 366)
    ttm = round(ttm, 4)
    yield_pct = round(ttm / price * 100, 2) if price > 0 and ttm > 0 else None
    return {
        "dividend_ttm_per_share": ttm,
        "dividend_yield_pct": yield_pct,
        "dividend_years": len(years),
        "latest_per_share": round(max(per_share)[1], 4) if per_share else 0.0,
    }


def _parse_date(value: Any) -> Optional[date]:
    if value is None:
        return None
    try:
        if value != value:   # NaN / pandas NaT
            return None
    except (TypeError, ValueError):
        return None
    if isinstance(value, datetime):
        parsed = value.date()
        return parsed if parsed.year > 1900 else None
    if isinstance(value, date):
        return value if value.year > 1900 else None
    text = str(value).strip()[:10]
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%Y%m%d"):
        try:
            parsed = datetime.strptime(text, fmt).date()
            return parsed if parsed.year > 1900 else None
        except ValueError:
            continue
    return None


# -- Financial quality --------------------------------------------------------
QUALITY_WEIGHTS = {
    "roe": 20.0, "gross_margin": 12.0, "net_margin": 8.0,
    "debt_ratio": 10.0, "current_ratio": 5.0, "cash_quality": 10.0,
    "roe_trend": 5.0,
}
GROWTH_WEIGHTS = {"revenue_growth": 45.0, "profit_growth": 45.0, "growth_trend": 10.0}


def _normalize(value: Optional[float], full: float, zero: float = 0.0) -> float:
    if value is None:
        return 0.0
    return clamp((value - zero) / (full - zero), 0.0, 1.0)


def compute_quality_score(metrics: Dict[str, Any]) -> Dict[str, Any]:
    roe = metrics.get("roe_annual") if metrics.get("roe_annual") is not None else metrics.get("roe")
    sub = {
        "roe": _normalize(roe, 22.0),
        "gross_margin": _normalize(metrics.get("gross_margin"), 35.0),
        "net_margin": _normalize(metrics.get("net_margin"), 15.0),
        "debt_ratio": clamp((85.0 - (metrics.get("debt_ratio") or 85.0)) / 35.0, 0.0, 1.0),
        "current_ratio": clamp(((metrics.get("current_ratio") or 1.0) - 1.0) / 0.8, 0.0, 1.0),
        "cash_quality": _normalize(metrics.get("cash_quality"), 1.2),
        "roe_trend": {"up": 1.0, "flat": 0.5, "down": 0.0}.get(metrics.get("roe_trend") or "flat", 0.5),
    }
    score = sum(sub[k] * w for k, w in QUALITY_WEIGHTS.items())
    score = round(clamp(score), 1)
    return {"score": score, "components": {k: round(v * 100, 1) for k, v in sub.items()},
            "grade": _grade(score)}


def compute_growth_score(metrics: Dict[str, Any]) -> Dict[str, Any]:
    rg = metrics.get("revenue_growth")
    pg = metrics.get("profit_growth")
    sub = {
        "revenue_growth": _normalize(rg, 30.0, -10.0),
        "profit_growth": _normalize(pg, 35.0, -10.0),
        "growth_trend": {"up": 1.0, "flat": 0.5, "down": 0.0}.get(metrics.get("growth_trend") or "flat", 0.5),
    }
    score = sum(sub[k] * w for k, w in GROWTH_WEIGHTS.items())
    return {"score": round(clamp(score), 1),
            "components": {k: round(v * 100, 1) for k, v in sub.items()},
            "grade": _grade(score)}


def _grade(score: float) -> str:
    if score >= 85:
        return "A+"
    if score >= 72:
        return "A"
    if score >= 58:
        return "B"
    if score >= 42:
        return "C"
    return "D"


def extract_financial_metrics(abstract_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Turn stock_financial_abstract rows (选项/指标/period columns) into metrics.

    `abstract_rows` must already be plain dicts; period keys look like '20260630'.
    """
    if not abstract_rows:
        return {}
    periods = sorted(
        {k for row in abstract_rows for k in row if re.fullmatch(r"\d{8}", str(k))},
        reverse=True,
    )
    table: Dict[str, Dict[str, Any]] = {}
    for row in abstract_rows:
        name = str(row.get("指标", "")).strip()
        if name:
            table[name] = row

    def pick(names: List[str], period: str) -> Optional[float]:
        for n in names:
            if n in table:
                v = safe_float(table[n].get(period))
                if v is not None:
                    return v
        return None

    def series(name: str) -> List[Tuple[str, float]]:
        row = table.get(name)
        if not row:
            return []
        out = []
        for p in periods:
            v = safe_float(row.get(p))
            if v is not None:
                out.append((p, v))
        return out

    latest = periods[0] if periods else ""
    annual = [p for p in periods if p.endswith("1231")]

    roe_series = series("净资产收益率(ROE)")
    annual_roe = [(p, v) for p, v in roe_series if p.endswith("1231")]
    roe_annual = annual_roe[0][1] if annual_roe else safe_float(pick(["净资产收益率(ROE)"], latest))
    roe_trend = "flat"
    if len(annual_roe) >= 2:
        delta = annual_roe[0][1] - annual_roe[1][1]
        roe_trend = "up" if delta > 1.0 else "down" if delta < -1.0 else "flat"

    rg = safe_float(pick(["营业总收入增长率"], latest))
    pg = safe_float(pick(["归属母公司净利润增长率", "净利润增长率"], latest))
    growth_trend = "flat"
    growth_series = [(p, v) for p, v in series("归属母公司净利润增长率")]
    if len(growth_series) >= 3:
        if growth_series[0][1] > growth_series[1][1] > growth_series[2][1]:
            growth_trend = "up"
        elif growth_series[0][1] < growth_series[1][1] < growth_series[2][1]:
            growth_trend = "down"

    metrics = {
        "latest_period": latest,
        "annual_period": annual[0] if annual else None,
        "roe": safe_float(pick(["净资产收益率(ROE)"], latest)),
        "roe_annual": roe_annual,
        "gross_margin": safe_float(pick(["毛利率"], latest)),
        "net_margin": safe_float(pick(["销售净利率"], latest)),
        "revenue_growth": rg,
        "profit_growth": pg,
        "debt_ratio": safe_float(pick(["资产负债率"], latest)),
        "current_ratio": safe_float(pick(["流动比率"], latest)),
        "cash_quality": safe_float(pick(["经营活动净现金/归属母公司的净利润",
                                          "经营现金净流量与净利润的比率(%)"], latest)),
        "eps": safe_float(pick(["基本每股收益"], latest)),
        "bvps": safe_float(pick(["每股净资产"], latest)),
        "roe_trend": roe_trend,
        "growth_trend": growth_trend,
    }
    if metrics["cash_quality"] is not None and metrics["cash_quality"] > 10:
        metrics["cash_quality"] = metrics["cash_quality"] / 100.0
    metrics["history"] = [
        {"period": p, "roe": safe_float(table.get("净资产收益率(ROE)", {}).get(p)),
         "gross_margin": safe_float(table.get("毛利率", {}).get(p)),
         "revenue_growth": safe_float(table.get("营业总收入增长率", {}).get(p)),
         "profit_growth": safe_float(table.get("归属母公司净利润增长率", {}).get(p)),
         "debt_ratio": safe_float(table.get("资产负债率", {}).get(p))}
        for p in periods[:8]
    ]
    return metrics


# -- Momentum & risk ----------------------------------------------------------
def compute_momentum_score(closes: List[float], indicators: Dict[str, Any]) -> Dict[str, Any]:
    arr = np.asarray([c for c in closes if safe_float(c) is not None], dtype=float)
    if arr.size < 21:
        return {"score": 50.0, "components": {}, "label": "数据不足"}

    price = float(arr[-1])
    ma5 = safe_float(indicators.get("ma5"))
    ma20 = safe_float(indicators.get("ma20"))
    ma60 = safe_float(indicators.get("ma60"))
    rsi = safe_float(indicators.get("rsi_14"))
    macd_hist = safe_float(indicators.get("macd_hist"))

    ret20 = (price / arr[-21] - 1) * 100 if arr[-21] > 0 else 0.0
    ret60 = (price / arr[-61] - 1) * 100 if arr.size >= 61 and arr[-61] > 0 else 0.0

    alignment = 0.0
    if ma5 and ma20 and ma60:
        alignment = 1.0 if ma5 > ma20 > ma60 else 0.0 if ma5 < ma20 < ma60 else 0.5
    sub = {
        "ma_alignment": alignment,
        "ret20": _normalize(ret20, 15.0, -15.0),
        "ret60": _normalize(ret60, 30.0, -30.0),
        "macd": 1.0 if (macd_hist or 0) > 0 else 0.0,
        "rsi": _rsi_score(rsi),
        "above_ma20": 1.0 if (ma20 and price > ma20) else 0.0,
    }
    weights = {"ma_alignment": 0.30, "ret20": 0.22, "ret60": 0.18,
               "macd": 0.12, "rsi": 0.10, "above_ma20": 0.08}
    score = sum(sub[k] * w * 100 for k, w in weights.items())
    if score >= 75:
        label = "强势"
    elif score >= 55:
        label = "偏强"
    elif score >= 40:
        label = "震荡"
    else:
        label = "弱势"
    return {"score": round(clamp(score), 1),
            "components": {k: round(v * 100, 1) for k, v in sub.items()},
            "label": label, "ret20_pct": round(ret20, 2), "ret60_pct": round(ret60, 2)}


def _rsi_score(rsi: Optional[float]) -> float:
    if rsi is None:
        return 0.5
    if 45 <= rsi <= 70:
        return 1.0
    if 35 <= rsi < 45 or 70 < rsi <= 78:
        return 0.6
    if rsi < 30 or rsi > 85:
        return 0.15
    return 0.4


def compute_risk_score(closes: List[float]) -> Dict[str, Any]:
    arr = np.asarray([c for c in closes if safe_float(c) is not None], dtype=float)
    if arr.size < 30:
        return {"score": 50.0, "components": {}, "label": "数据不足"}

    rets = simple_returns(arr)
    ann_vol = float(np.std(rets, ddof=1)) * math.sqrt(252) * 100
    var95 = abs(float(np.quantile(rets, 0.05))) * 100
    peak = np.maximum.accumulate(arr)
    drawdown = float(np.min(arr / peak - 1.0)) * 100

    sub = {
        "volatility": 1.0 - _normalize(ann_vol, 65.0, 12.0),
        "max_drawdown": 1.0 - _normalize(abs(drawdown), 55.0, 5.0),
        "var95": 1.0 - _normalize(var95, 6.0, 0.8),
    }
    weights = {"volatility": 0.40, "max_drawdown": 0.35, "var95": 0.25}
    score = sum(sub[k] * w * 100 for k, w in weights.items())
    if score >= 75:
        label = "低风险"
    elif score >= 55:
        label = "中低风险"
    elif score >= 40:
        label = "中等风险"
    else:
        label = "高风险"
    return {
        "score": round(clamp(score), 1),
        "components": {k: round(v * 100, 1) for k, v in sub.items()},
        "label": label,
        "annual_volatility_pct": round(ann_vol, 2),
        "max_drawdown_pct": round(drawdown, 2),
        "daily_var95_pct": round(var95, 2),
    }


# -- Composite & action plan --------------------------------------------------
DIMENSION_WEIGHTS = {"valuation": 25.0, "quality": 25.0, "growth": 20.0,
                     "momentum": 15.0, "risk": 15.0}
DIMENSION_LABELS = {"valuation": "估值", "quality": "质量", "growth": "成长",
                    "momentum": "动量", "risk": "风险"}


def compute_composite(dimensions: Dict[str, Optional[float]]) -> Dict[str, Any]:
    total = 0.0
    used_weight = 0.0
    breakdown = {}
    for key, weight in DIMENSION_WEIGHTS.items():
        value = dimensions.get(key)
        if value is None:
            breakdown[key] = {"label": DIMENSION_LABELS[key], "score": None,
                              "weight": weight, "weighted": 0.0}
            continue
        weighted = value * weight / 100.0
        total += weighted
        used_weight += weight
        breakdown[key] = {"label": DIMENSION_LABELS[key], "score": round(value, 1),
                          "weight": weight, "weighted": round(weighted, 2)}
    score = round(total / used_weight * 100, 1) if used_weight else 0.0
    if score >= 80:
        rating, stance = "强烈看好", "优质标的，可重点跟踪"
    elif score >= 65:
        rating, stance = "积极关注", "基本面与估值匹配，逢低布局"
    elif score >= 50:
        rating, stance = "中性观察", "等待更明确的催化或估值改善"
    elif score >= 35:
        rating, stance = "谨慎", "存在明显短板，控制仓位"
    else:
        rating, stance = "回避", "多项指标偏弱，暂不介入"
    return {"score": score, "rating": rating, "stance": stance,
            "breakdown": breakdown, "coverage_weight": used_weight}


def build_action_plan(price: float, fair_value: Optional[Dict[str, Any]],
                      composite_score: float, risk: Dict[str, Any],
                      supports: List[float], resistances: List[float],
                      growth_score: Optional[float] = None) -> Dict[str, Any]:
    if not price or price <= 0:
        return {}

    fair_low = (fair_value or {}).get("low")
    fair_mid = (fair_value or {}).get("mid")
    fair_high = (fair_value or {}).get("high")

    if fair_low and fair_mid and price < fair_low:
        action = "分批建仓"
        entry_low, entry_high = price * 0.96, price * 1.02
    elif fair_mid and price > (fair_high or fair_mid):
        action = "高估观望"
        anchor = min(price * 0.90, (fair_mid or price) * 0.95)
        entry_low, entry_high = anchor, price * 0.96
    elif composite_score >= 65:
        action = "逢低布局"
        entry_low, entry_high = price * 0.92, min(price * 0.99, fair_mid or price)
    elif composite_score >= 50:
        action = "观察等待"
        entry_low, entry_high = price * 0.93, price * 0.98
    else:
        action = "回避"
        entry_low, entry_high = price * 0.85, price * 0.92

    entry_low = round(max(entry_low, price * 0.80), 2)
    entry_high = round(max(min(entry_high, price * 1.03), entry_low), 2)

    support = supports[0] if supports else price * 0.92
    daily_vol = (risk.get("annual_volatility_pct") or 30.0) / math.sqrt(252) / 100
    stop_distance = max(0.06, min(0.16, daily_vol * 4.0))
    stop_by_vol = price * (1 - stop_distance)
    stop_loss = min(stop_by_vol, support) if 0 < support < price else stop_by_vol
    stop_loss = round(max(stop_loss, price * 0.80), 2)

    growth_cap = 0.10 + clamp((growth_score if growth_score is not None else 50.0) - 40.0,
                              0.0, 60.0) / 60.0 * 0.30
    resistance = resistances[0] if resistances else price * 1.12
    raw_target = max(fair_high or price * 1.12, resistance if resistance > price else price * 1.12)
    target = min(raw_target, price * (1 + growth_cap))
    if composite_score >= 55:
        target = max(target, price * 1.05)
    else:
        target = min(target, price * 1.05)
    target = round(target, 2)

    risk_per_share = max(price - stop_loss, price * 0.01)
    reward_per_share = max(target - price, 0.0)
    rr = round(reward_per_share / risk_per_share, 2) if risk_per_share > 0 else 0.0

    base = clamp((composite_score - 40.0) / 60.0, 0.0, 1.0) * 25.0
    vol = risk.get("annual_volatility_pct") or 35.0
    vol_adj = clamp(30.0 / max(vol, 12.0), 0.35, 1.0)
    position_pct = round(min(base * vol_adj, 30.0), 1)
    if composite_score < 40 or action == "回避":
        position_pct = 0.0

    return {
        "action": action,
        "entry_zone": [round(entry_low, 2), round(entry_high, 2)],
        "stop_loss": stop_loss,
        "target": target,
        "risk_reward_ratio": rr,
        "position_pct": position_pct,
        "expected_upside_pct": round((target / price - 1) * 100, 2),
        "max_risk_pct": round((1 - stop_loss / price) * 100, 2),
    }


def build_risk_flags(valuation: ValuationResult, metrics: Dict[str, Any],
                     risk: Dict[str, Any], price: float, ma60: Optional[float]) -> List[str]:
    flags: List[str] = []
    if valuation.pe_percentile is not None and valuation.pe_percentile >= 80:
        flags.append(f"PE 处于近五年 {valuation.pe_percentile:.0f}% 分位，估值回调压力较大")
    if (metrics.get("revenue_growth") or 0) < 0:
        flags.append(f"营业收入同比下滑 {abs(metrics['revenue_growth']):.1f}%")
    if (metrics.get("profit_growth") or 0) < 0:
        flags.append(f"归母净利润同比下滑 {abs(metrics['profit_growth']):.1f}%")
    if (metrics.get("debt_ratio") or 0) > 70:
        flags.append(f"资产负债率 {metrics['debt_ratio']:.1f}%，杠杆偏高")
    if (metrics.get("cash_quality") or 1) < 0.8:
        flags.append(f"经营现金流/净利润仅 {metrics['cash_quality']:.2f}，盈利含金量待验证")
    if (metrics.get("current_ratio") or 2) < 1:
        flags.append(f"流动比率 {metrics['current_ratio']:.2f} < 1，短期偿债压力")
    if (risk.get("annual_volatility_pct") or 0) > 45:
        flags.append(f"年化波动率 {risk['annual_volatility_pct']:.1f}%，持仓波动较大")
    if ma60 and price and price < ma60:
        flags.append("股价位于 60 日均线下方，中期趋势偏弱")
    if valuation.dividend_years == 0:
        flags.append("近年无现金分红记录，缺少股息保护")
    return flags


def build_cautions(valuation: ValuationResult, momentum: Dict[str, Any],
                    composite: Dict[str, Any], divergence: bool) -> List[str]:
    cautions: List[str] = []
    band = valuation.pe_band or {}
    p50, p90 = band.get("p50"), band.get("p90")
    if p50 and p90 and p90 / max(p50, 1e-9) >= 2.5:
        cautions.append(
            f"近五年 PE 波动极大（中位数 {p50:.1f}x / P90 {p90:.1f}x），历史分位参考意义下降，目标价已按成长性约束。"
        )
    if divergence:
        cautions.append(
            "基本面评分与价格动量明显背离（左侧特征）：估值/成长占优但趋势偏弱，宜分批建仓并严格执行止损。"
        )
    if (momentum.get("score") or 0) < 30:
        cautions.append("当前处于弱势趋势，避免一次性重仓；等待放量企稳或均线转多再加大仓位。")
    if (valuation.pe_percentile is not None and valuation.pe_percentile <= 10
            and (composite.get("score") or 0) < 55):
        cautions.append("估值虽处历史低位，但综合评分不高，需警惕价值陷阱（基本面恶化导致的低估值）。")
    if not cautions:
        cautions.append("暂无显著背离信号；仍需关注季报披露、行业政策与流动性变化。")
    return cautions


def build_narrative(valuation: ValuationResult, metrics: Dict[str, Any],
                    quality: Dict[str, Any], growth: Dict[str, Any],
                    momentum: Dict[str, Any], risk: Dict[str, Any],
                    composite: Dict[str, Any], plan: Dict[str, Any]) -> List[str]:
    """Deterministic, data-grounded sentences — the constructive part of the report."""
    lines: List[str] = []
    if valuation.pe is not None and valuation.pe > 0:
        lines.append(
            f"估值：PE(TTM) {valuation.pe:.1f}x / PB {valuation.pb:.2f}x"
            + (f"，PE 位于近五年 {valuation.pe_percentile:.0f}% 分位" if valuation.pe_percentile is not None else "")
            + f"，综合估值评分 {valuation.score:.0f} 分（{valuation.label}）。"
        )
    else:
        pb_txt = f"{valuation.pb:.2f}x" if valuation.pb else "N/A"
        lines.append(f"估值：PE 失效（亏损或数据缺失），以 PB {pb_txt} 与股息率综合判断，评分 {valuation.score:.0f} 分。")

    q = quality.get("score") or 0
    roe_txt = f"ROE {metrics['roe_annual']:.1f}% " if metrics.get("roe_annual") is not None else ""
    margin_txt = f"毛利率 {metrics['gross_margin']:.1f}% " if metrics.get("gross_margin") is not None else ""
    lines.append(
        f"质量：{quality.get('grade')} 级（{q:.0f} 分），{roe_txt}{margin_txt}"
        + ("，盈利与资产结构扎实。" if q >= 70 else "，盈利与资产结构中规中矩。" if q >= 50 else "，盈利与资产结构存在短板。")
    )
    g = growth.get("score") or 0
    rg = metrics.get("revenue_growth")
    pg = metrics.get("profit_growth")
    growth_bits = []
    if rg is not None:
        growth_bits.append(f"营收同比 {rg:+.1f}%")
    if pg is not None:
        growth_bits.append(f"净利同比 {pg:+.1f}%")
    lines.append(
        f"成长：{growth.get('grade')} 级（{g:.0f} 分），"
        + ("、".join(growth_bits) if growth_bits else "增长数据不足")
        + ("，增长强劲。" if g >= 70 else "，增长动能一般。" if g >= 45 else "，增长承压。")
    )
    lines.append(
        f"动量：{momentum.get('label')}（{momentum.get('score', 0):.0f} 分），"
        f"近 20 日 {momentum.get('ret20_pct', 0):+.1f}%、近 60 日 {momentum.get('ret60_pct', 0):+.1f}%。"
    )
    lines.append(
        f"风险：{risk.get('label')}（{risk.get('score', 0):.0f} 分），"
        f"年化波动 {risk.get('annual_volatility_pct', 0):.1f}%，"
        f"最大回撤 {risk.get('max_drawdown_pct', 0):.1f}%。"
    )
    lines.append(
        f"结论：综合评分 {composite.get('score', 0):.0f} 分 → {composite.get('rating')}。"
        f"{composite.get('stance')}。"
    )
    if plan:
        lo, hi = plan.get("entry_zone", [0, 0])
        lines.append(
            f"操作建议：{plan.get('action')}，参考区间 {lo:.2f}–{hi:.2f}，"
            f"止损 {plan.get('stop_loss'):.2f}，目标 {plan.get('target'):.2f}，"
            f"风险收益比 {plan.get('risk_reward_ratio'):.2f}，建议仓位 ≤{plan.get('position_pct'):.1f}%。"
        )
    return lines
    lines.append(
        f"成长：{growth.get('grade')} 级（{g:.0f} 分），"
        + ("营收与利润保持较快增长" if (g or 0) >= 70 else "增长动能一般" if (g or 0) >= 45 else "增长承压")
        + "。"
    )
    lines.append(
        f"动量：{momentum.get('label')}（{momentum.get('score', 0):.0f} 分），"
        f"近 20 日 {momentum.get('ret20_pct', 0):+.1f}%、近 60 日 {momentum.get('ret60_pct', 0):+.1f}%。"
    )
    lines.append(
        f"风险：{risk.get('label')}（{risk.get('score', 0):.0f} 分），"
        f"年化波动 {risk.get('annual_volatility_pct', 0):.1f}%，"
        f"最大回撤 {risk.get('max_drawdown_pct', 0):.1f}%。"
    )
    lines.append(
        f"结论：综合评分 {composite.get('score', 0):.0f} 分 → {composite.get('rating')}。"
        f"{composite.get('stance')}。"
    )
    if plan:
        lo, hi = plan.get("entry_zone", [0, 0])
        lines.append(
            f"操作建议：{plan.get('action')}，参考区间 {lo:.2f}–{hi:.2f}，"
            f"止损 {plan.get('stop_loss'):.2f}，目标 {plan.get('target'):.2f}，"
            f"风险收益比 {plan.get('risk_reward_ratio'):.2f}，建议仓位 ≤{plan.get('position_pct'):.1f}%。"
        )
    return lines


# -- Data fetching (network) --------------------------------------------------
_cache: Dict[str, Tuple[float, Any]] = {}
_CACHE_TTL = 600.0


def _cached(key: str) -> Optional[Any]:
    item = _cache.get(key)
    if item and time.time() - item[0] < _CACHE_TTL:
        return item[1]
    return None


def _store(key: str, value: Any) -> Any:
    _cache[key] = (time.time(), value)
    return value


def _strip_code(symbol: str) -> str:
    match = re.search(r"(\d{6})", str(symbol))
    return match.group(1) if match else str(symbol).split(".")[0]


def _downsample(points: List[Dict[str, Any]], limit: int) -> List[Dict[str, Any]]:
    if len(points) <= limit:
        return points
    step = len(points) / float(limit)
    sampled = [points[int(i * step)] for i in range(limit)]
    if sampled[-1] is not points[-1]:
        sampled[-1] = points[-1]
    return sampled


async def fetch_valuation_history(code: str) -> Dict[str, Any]:
    cached = _cached(f"val:{code}")
    if cached is not None:
        return cached
    import akshare as ak

    loop = asyncio.get_event_loop()

    def _load(indicator: str) -> List[Dict[str, Any]]:
        try:
            df = ak.stock_zh_valuation_baidu(symbol=code, indicator=indicator, period="近五年")
            if df is None or df.empty or "value" not in df.columns:
                return []
            points = []
            for row in df.to_dict(orient="records"):
                value = safe_float(row.get("value"))
                if value is None or value <= 0:
                    continue
                points.append({"date": str(row.get("date"))[:10], "value": round(value, 4)})
            return _downsample(points, 180)
        except Exception as exc:
            logger.warning(f"valuation {indicator} failed for {code}: {exc}")
            return []

    pe_series, pb_series = await asyncio.gather(
        loop.run_in_executor(None, _load, "市盈率(TTM)"),
        loop.run_in_executor(None, _load, "市净率"),
    )
    result = {
        "pe_series": pe_series,
        "pb_series": pb_series,
        "pe_values": [p["value"] for p in pe_series],
        "pb_values": [p["value"] for p in pb_series],
    }
    return _store(f"val:{code}", result)


async def fetch_financial_abstract(code: str) -> List[Dict[str, Any]]:
    cached = _cached(f"fin:{code}")
    if cached is not None:
        return cached
    import akshare as ak

    loop = asyncio.get_event_loop()

    def _load() -> List[Dict[str, Any]]:
        try:
            df = ak.stock_financial_abstract(symbol=code)
            if df is None or df.empty:
                return []
            return df.to_dict(orient="records")
        except Exception as exc:
            logger.warning(f"financial abstract failed for {code}: {exc}")
            return []

    rows = await loop.run_in_executor(None, _load)
    return _store(f"fin:{code}", rows)


async def fetch_dividends(code: str) -> List[Dict[str, Any]]:
    cached = _cached(f"div:{code}")
    if cached is not None:
        return cached
    import akshare as ak

    loop = asyncio.get_event_loop()

    def _load() -> List[Dict[str, Any]]:
        try:
            df = ak.stock_history_dividend_detail(symbol=code, indicator="分红")
            if df is None or df.empty:
                return []
            return df.to_dict(orient="records")
        except Exception as exc:
            logger.warning(f"dividend fetch failed for {code}: {exc}")
            return []

    rows = await loop.run_in_executor(None, _load)
    return _store(f"div:{code}", rows)


async def analyze_valuation(code: str, price: float) -> ValuationResult:
    hist, dividends = await asyncio.gather(
        fetch_valuation_history(code),
        fetch_dividends(code),
    )
    pe_values = hist.get("pe_values") or []
    pb_values = hist.get("pb_values") or []

    pe = pe_values[-1] if pe_values else None
    pb = pb_values[-1] if pb_values else None
    pe_pct = percentile_rank(pe_values, pe) if pe is not None else None
    pb_pct = percentile_rank(pb_values, pb) if pb is not None else None

    pe_band = {f"p{int(q*100)}": quantile(pe_values, q) for q in (0.1, 0.25, 0.5, 0.75, 0.9)}
    pb_band = {f"p{int(q*100)}": quantile(pb_values, q) for q in (0.1, 0.25, 0.5, 0.75, 0.9)}
    pe_band = {k: v for k, v in pe_band.items() if v is not None}
    pb_band = {k: v for k, v in pb_band.items() if v is not None}

    div = compute_ttm_dividend(dividends, price)
    score, label, reasons = compute_valuation_score(
        pe, pb, pe_pct, pb_pct, div["dividend_yield_pct"],
    )

    fair_pe = build_fair_value(price, pe, pe_band, "PE")
    fair_pb = build_fair_value(price, pb, pb_band, "PB")
    fair = fair_pe or fair_pb or {}
    if fair_pe and fair_pb:
        fair["cross_check"] = {
            "pe_mid": fair_pe["mid"],
            "pb_mid": fair_pb["mid"],
            "consistent": abs(fair_pe["mid"] - fair_pb["mid"]) / max(price, 1e-9) < 0.25,
        }

    return ValuationResult(
        current_price=price, pe=pe, pb=pb,
        pe_percentile=pe_pct, pb_percentile=pb_pct,
        pe_band=pe_band, pb_band=pb_band,
        pe_series=hist.get("pe_series") or [],
        pb_series=hist.get("pb_series") or [],
        fair_value=fair,
        dividend_ttm_per_share=div["dividend_ttm_per_share"],
        dividend_yield_pct=div["dividend_yield_pct"],
        dividend_years=div["dividend_years"],
        score=score, label=label, reasons=reasons,
    )
