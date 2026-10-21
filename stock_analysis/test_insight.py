# ============================================================================
# Tests for app/insight.py - deterministic investment insight engine
# ============================================================================
from __future__ import annotations

import math
from datetime import date, timedelta

import numpy as np
import pytest

from app.insight import (
    ValuationResult,
    build_action_plan,
    build_fair_value,
    build_narrative,
    build_risk_flags,
    compute_composite,
    compute_growth_score,
    compute_momentum_score,
    compute_quality_score,
    compute_risk_score,
    compute_ttm_dividend,
    compute_valuation_score,
    extract_financial_metrics,
    percentile_rank,
    quantile,
    safe_float,
)


class TestHelpers:
    def test_safe_float_nan_and_none(self):
        assert safe_float("3.5") == 3.5
        assert safe_float(None) is None
        assert safe_float("abc") is None
        assert safe_float(float("nan")) is None
        assert safe_float(float("inf")) is None

    def test_percentile_rank(self):
        values = [10, 20, 30, 40, 50]
        assert percentile_rank(values, 30) == 60.0
        assert percentile_rank(values, 5) == 0.0
        assert percentile_rank(values, 50) == 100.0
        assert percentile_rank([], 10) is None
        assert percentile_rank([None, "x"], 10) is None

    def test_quantile_ignores_bad_values(self):
        assert quantile([1, 2, 3, 4, 5], 0.5) == 3.0
        assert quantile([], 0.5) is None


class TestValuation:
    def test_cheap_valuation_scores_high(self):
        score, label, reasons = compute_valuation_score(
            pe=8.0, pb=1.0, pe_pct=5.0, pb_pct=10.0, dividend_yield=4.0,
        )
        assert score >= 75
        assert label == "低估"
        assert any("历史低估" in r for r in reasons)
        assert any("股息率" in r for r in reasons)

    def test_expensive_valuation_scores_low(self):
        score, label, _ = compute_valuation_score(
            pe=90.0, pb=12.0, pe_pct=95.0, pb_pct=95.0, dividend_yield=0.0,
        )
        assert score < 30
        assert label == "高估"

    def test_loss_making_company_penalized(self):
        score, label, reasons = compute_valuation_score(
            pe=-5.0, pb=2.0, pe_pct=None, pb_pct=50.0, dividend_yield=None,
        )
        assert score < 50
        assert any("PE 为负" in r for r in reasons)
        assert any("无现金分红" in r for r in reasons)

    def test_fair_value_scales_with_quartiles(self):
        band = {"p25": 10.0, "p50": 20.0, "p75": 30.0}
        fair = build_fair_value(price=100.0, current_multiple=40.0, band=band, kind="PE")
        assert fair["low"] == 25.0
        assert fair["mid"] == 50.0
        assert fair["high"] == 75.0
        assert fair["upside_pct"] == -50.0

    def test_fair_value_missing_inputs(self):
        assert build_fair_value(0, 10, {"p25": 1, "p50": 2, "p75": 3}, "PE") is None
        assert build_fair_value(10, None, {"p25": 1, "p50": 2, "p75": 3}, "PE") is None
        assert build_fair_value(10, 10, {}, "PE") is None


class TestDividend:
    def test_ttm_and_units(self):
        today = date.today()
        rows = [
            {"除权除息日": today - timedelta(days=100), "派息": 20.0},
            {"除权除息日": today - timedelta(days=400), "派息": 10.0},
            {"除权除息日": today - timedelta(days=700), "派息": 5.0},
        ]
        result = compute_ttm_dividend(rows, price=100.0)
        assert result["dividend_ttm_per_share"] == 2.0
        assert result["dividend_yield_pct"] == 2.0
        assert result["dividend_years"] >= 2

    def test_no_dividends(self):
        result = compute_ttm_dividend([], price=50.0)
        assert result["dividend_ttm_per_share"] == 0.0
        assert result["dividend_yield_pct"] is None
        assert result["dividend_years"] == 0


def _abstract_rows() -> list[dict]:
    periods = ["20260630", "20251231", "20241231", "20231231"]
    def row(name, vals, option="常用指标"):
        d = {"选项": option, "指标": name}
        d.update({p: v for p, v in zip(periods, vals)})
        return d
    return [
        row("净资产收益率(ROE)", [12.08, 21.42, 20.55, 22.32]),
        row("毛利率", [23.93, 22.10, 19.50, 21.00]),
        row("销售净利率", [16.98, 18.12, 14.92, 11.66]),
        row("资产负债率", [63.65, 61.94, 65.24, 69.34]),
        row("流动比率", [1.56, 1.60, 1.61, 1.57]),
        row("经营活动净现金/归属母公司的净利润", [1.39, 1.73, 1.80, 1.99]),
        row("基本每股收益", [9.51, 18.20, 14.10, 12.30]),
        row("每股净资产", [81.99, 78.50, 65.00, 55.00]),
        row("营业总收入增长率", [54.80, 17.04, -9.70, 22.01], option="成长能力"),
        row("归属母公司净利润增长率", [41.98, 42.18, 15.50, 39.76], option="成长能力"),
    ]


class TestFinancialMetrics:
    def test_extract_metrics(self):
        metrics = extract_financial_metrics(_abstract_rows())
        assert metrics["latest_period"] == "20260630"
        assert metrics["annual_period"] == "20251231"
        assert metrics["roe_annual"] == 21.42
        assert metrics["gross_margin"] == 23.93
        assert metrics["revenue_growth"] == 54.80
        assert metrics["profit_growth"] == 41.98
        assert metrics["debt_ratio"] == 63.65
        assert metrics["roe_trend"] in {"up", "down", "flat"}
        assert len(metrics["history"]) == 4

    def test_extract_empty(self):
        assert extract_financial_metrics([]) == {}

    def test_quality_score_range_and_grade(self):
        metrics = extract_financial_metrics(_abstract_rows())
        quality = compute_quality_score(metrics)
        assert 0 <= quality["score"] <= 100
        assert quality["grade"] in {"A+", "A", "B", "C", "D"}
        assert quality["score"] > 50

    def test_low_quality_company(self):
        metrics = {
            "roe_annual": 1.0, "gross_margin": 5.0, "net_margin": 1.0,
            "debt_ratio": 85.0, "current_ratio": 0.6, "cash_quality": 0.1,
            "roe_trend": "down",
        }
        quality = compute_quality_score(metrics)
        assert quality["score"] < 30
        assert quality["grade"] == "D"

    def test_growth_score_fast_vs_shrinking(self):
        fast = compute_growth_score({
            "revenue_growth": 40.0, "profit_growth": 50.0, "growth_trend": "up",
        })
        shrinking = compute_growth_score({
            "revenue_growth": -20.0, "profit_growth": -30.0, "growth_trend": "down",
        })
        assert fast["score"] > 90
        assert shrinking["score"] < 15


class TestMomentumAndRisk:
    @staticmethod
    def _uptrend(n=120):
        return [100 * (1.004 ** i) for i in range(n)]

    @staticmethod
    def _downtrend(n=120):
        return [100 * (0.996 ** i) for i in range(n)]

    def test_uptrend_scores_higher(self):
        up = compute_momentum_score(self._uptrend(), {
            "ma5": 110, "ma20": 105, "ma60": 95, "rsi_14": 60, "macd_hist": 0.5,
        })
        down = compute_momentum_score(self._downtrend(), {
            "ma5": 90, "ma20": 95, "ma60": 105, "rsi_14": 35, "macd_hist": -0.5,
        })
        assert up["score"] > down["score"]
        assert up["ret20_pct"] > 0 > down["ret20_pct"]

    def test_momentum_insufficient_data(self):
        result = compute_momentum_score([1.0, 2.0], {})
        assert result["score"] == 50.0

    def test_risk_low_vs_high_volatility(self):
        rng = np.random.default_rng(42)
        calm = list(100 + np.cumsum(rng.normal(0, 0.3, 200)))
        wild = list(100 + np.cumsum(rng.normal(0, 4.0, 200)))
        calm_score = compute_risk_score(calm)
        wild_score = compute_risk_score(wild)
        assert calm_score["score"] > wild_score["score"]
        assert calm_score["annual_volatility_pct"] < wild_score["annual_volatility_pct"]
        assert wild_score["label"] in {"中等风险", "高风险"}

    def test_risk_insufficient_data(self):
        assert compute_risk_score([1, 2, 3])["score"] == 50.0


class TestComposite:
    def test_all_dimensions(self):
        result = compute_composite({
            "valuation": 80, "quality": 70, "growth": 60, "momentum": 50, "risk": 40,
        })
        assert 0 <= result["score"] <= 100
        assert result["coverage_weight"] == 100.0
        assert result["rating"] in {"强烈看好", "积极关注", "中性观察", "谨慎", "回避"}
        expected = (80 * .25 + 70 * .25 + 60 * .20 + 50 * .15 + 40 * .15)
        assert result["score"] == pytest.approx(expected, abs=0.1)

    def test_missing_dimension_renormalizes(self):
        result = compute_composite({
            "valuation": 80, "quality": 80, "growth": None, "momentum": None, "risk": None,
        })
        assert result["score"] == pytest.approx(80.0, abs=0.1)
        assert result["coverage_weight"] == 50.0
        assert result["breakdown"]["growth"]["score"] is None

    def test_all_missing(self):
        result = compute_composite({})
        assert result["score"] == 0.0
        assert result["coverage_weight"] == 0.0


class TestActionPlan:
    def test_cheap_stock_plan_is_consistent(self):
        plan = build_action_plan(
            price=100.0,
            fair_value={"low": 120.0, "mid": 140.0, "high": 160.0},
            composite_score=75.0,
            risk={"annual_volatility_pct": 30.0},
            supports=[92.0],
            resistances=[130.0],
        )
        assert plan["action"] == "分批建仓"
        assert plan["stop_loss"] < 100.0
        assert plan["target"] > plan["stop_loss"]
        assert plan["position_pct"] > 0
        assert plan["risk_reward_ratio"] > 0
        lo, hi = plan["entry_zone"]
        assert lo <= hi

    def test_entry_zone_stays_near_price(self):
        plan = build_action_plan(
            price=200.0,
            fair_value={"low": 60.0, "mid": 80.0, "high": 100.0},
            composite_score=45.0,
            risk={"annual_volatility_pct": 40.0},
            supports=[150.0], resistances=[220.0],
        )
        lo, hi = plan["entry_zone"]
        assert lo >= 200.0 * 0.80
        assert hi <= 200.0 * 1.03
        assert lo <= hi

    def test_growth_score_caps_target(self):
        slow = build_action_plan(
            price=100.0, fair_value={"low": 110, "mid": 150, "high": 200},
            composite_score=70.0, risk={"annual_volatility_pct": 25.0},
            supports=[95.0], resistances=[140.0], growth_score=40.0,
        )
        fast = build_action_plan(
            price=100.0, fair_value={"low": 110, "mid": 150, "high": 200},
            composite_score=70.0, risk={"annual_volatility_pct": 25.0},
            supports=[95.0], resistances=[140.0], growth_score=95.0,
        )
        assert slow["target"] <= 100.0 * 1.11
        assert fast["target"] >= slow["target"]
        assert fast["target"] <= 100.0 * 1.41

    def test_expensive_stock_waits(self):
        plan = build_action_plan(
            price=200.0,
            fair_value={"low": 100.0, "mid": 120.0, "high": 140.0},
            composite_score=70.0,
            risk={"annual_volatility_pct": 30.0},
            supports=[180.0],
            resistances=[220.0],
        )
        assert plan["action"] == "高估观望"
        assert plan["entry_zone"][1] < 200.0

    def test_weak_stock_no_position(self):
        plan = build_action_plan(
            price=50.0,
            fair_value={"low": 40.0, "mid": 45.0, "high": 55.0},
            composite_score=25.0,
            risk={"annual_volatility_pct": 60.0},
            supports=[48.0],
            resistances=[60.0],
        )
        assert plan["position_pct"] == 0.0
        assert plan["stop_loss"] < 50.0

    def test_no_price_returns_empty(self):
        assert build_action_plan(0, None, 50, {}, [], []) == {}


class TestRiskFlagsAndNarrative:
    def test_flags_capture_weaknesses(self):
        valuation = ValuationResult(pe=80.0, pb=10.0, pe_percentile=92.0,
                                    dividend_years=0, score=20.0, label="高估")
        metrics = {
            "revenue_growth": -12.0, "profit_growth": -30.0, "debt_ratio": 78.0,
            "cash_quality": 0.4, "current_ratio": 0.8,
        }
        risk = {"annual_volatility_pct": 55.0}
        flags = build_risk_flags(valuation, metrics, risk, price=10.0, ma60=12.0)
        text = " | ".join(flags)
        assert "估值回调压力" in text
        assert "营业收入同比下滑" in text
        assert "资产负债率" in text
        assert "60 日均线" in text
        assert len(flags) >= 6

    def test_narrative_contains_numbers(self):
        valuation = ValuationResult(pe=18.0, pb=2.5, pe_percentile=25.0,
                                    dividend_years=3, score=70.0, label="偏低")
        metrics = {"roe_annual": 21.4, "gross_margin": 23.9,
                   "revenue_growth": 17.0, "profit_growth": 42.0}
        quality = {"score": 72, "grade": "A"}
        growth = {"score": 68, "grade": "B"}
        momentum = {"score": 60, "label": "偏强", "ret20_pct": 3.2, "ret60_pct": 8.1}
        risk = {"score": 55, "label": "中低风险", "annual_volatility_pct": 32.0,
                "max_drawdown_pct": -28.0}
        composite = {"score": 66, "rating": "积极关注", "stance": "逢低布局"}
        plan = {"action": "逢低布局", "entry_zone": [95.0, 100.0], "stop_loss": 90.0,
                "target": 120.0, "risk_reward_ratio": 2.0, "position_pct": 15.0}
        lines = build_narrative(valuation, metrics, quality, growth, momentum,
                                risk, composite, plan)
        text = "\n".join(lines)
        assert "PE(TTM) 18.0x" in text
        assert "ROE 21.4%" in text
        assert "营收同比 +17.0%" in text
        assert "综合评分 66" in text
        assert "止损 90.00" in text
