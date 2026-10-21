# ============================================================================
# Tests - Technical indicators & direction forecast engine
# ============================================================================
"""纯函数测试：不触网、不依赖数据库。"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from app import ta
from app.forecast import (
    _to_df, blend_probability, build_signals, build_stock_forecast,
    conditional_outcome, key_levels, market_regime, multi_timeframe,
    relative_strength, scenario_plan, technical_score, timeframe_trend,
)
from app.industry import CSRC_TO_SW, SW_LEVEL1, _quadrants, industry_metrics


# -- Fixtures -----------------------------------------------------------------
def make_df(n: int = 300, drift: float = 0.004, vol: float = 0.012,
            seed: int = 7, start: float = 100.0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rets = rng.normal(drift, vol, n)
    close = start * np.cumprod(1 + rets)
    high = close * (1 + np.abs(rng.normal(0, 0.006, n)))
    low = close * (1 - np.abs(rng.normal(0, 0.006, n)))
    open_ = close * (1 + rng.normal(0, 0.004, n))
    dates = pd.bdate_range("2023-01-02", periods=n).strftime("%Y-%m-%d")
    return pd.DataFrame({
        "date": dates, "open": open_, "high": high, "low": low, "close": close,
        "volume": rng.integers(1_000_000, 5_000_000, n).astype(float),
        "amount": close * 1e6,
    })


@pytest.fixture()
def uptrend():
    return make_df(320, drift=0.006, vol=0.010, seed=1)


@pytest.fixture()
def downtrend():
    return make_df(320, drift=-0.006, vol=0.010, seed=2)


@pytest.fixture()
def bench():
    return make_df(320, drift=0.0005, vol=0.008, seed=9, start=4000)


# -- Indicators ---------------------------------------------------------------
def test_sma_matches_manual():
    s = pd.Series([1, 2, 3, 4, 5], dtype=float)
    assert ta.sma(s, 3).iloc[-1] == pytest.approx(4.0)
    assert math.isnan(ta.sma(s, 3).iloc[0])


def test_ema_recursive():
    s = pd.Series([1.0, 2.0, 3.0])
    out = ta.ema(s, 2)
    alpha = 2 / 3
    assert out.iloc[1] == pytest.approx(alpha * 2 + (1 - alpha) * 1)
    assert out.iloc[2] == pytest.approx(alpha * 3 + (1 - alpha) * out.iloc[1])


def test_macd_hist_is_double_gap(uptrend):
    m = ta.macd(uptrend["close"])
    gap = m["dif"] - m["dea"]
    assert (m["hist"] - gap * 2).abs().max() < 1e-9


def test_rsi_bounds_and_extremes():
    up = pd.Series(np.arange(1, 60, dtype=float))
    down = up[::-1].reset_index(drop=True)
    assert ta.rsi(up, 14).iloc[-1] > 95
    assert ta.rsi(down, 14).iloc[-1] < 5
    r = ta.rsi(up, 14).dropna()
    assert ((r >= 0) & (r <= 100)).all()


def test_kdj_j_definition():
    df = make_df(60)
    k = ta.kdj(df["high"], df["low"], df["close"])
    assert (k["j"] - (3 * k["k"] - 2 * k["d"])).abs().max() < 1e-9


def test_boll_bands_order(uptrend):
    b = ta.boll(uptrend["close"])
    last = -1
    assert b["low"].iloc[last] < b["mid"].iloc[last] < b["up"].iloc[last]


def test_atr_positive(uptrend):
    a = ta.atr(uptrend["high"], uptrend["low"], uptrend["close"])
    assert (a.dropna() > 0).all()


def test_obv_accumulates_on_up_days():
    close = pd.Series([10, 11, 10, 12], dtype=float)
    vol = pd.Series([100, 100, 100, 100], dtype=float)
    out = ta.obv(close, vol)
    assert out.iloc[-1] == pytest.approx(100)


def test_max_drawdown():
    assert ta.max_drawdown([100, 120, 60, 90]) == pytest.approx(50.0)


def test_resample_weekly_preserves_direction(uptrend):
    w = ta.resample_ohlc(uptrend, "W-FRI")
    assert len(w) < len(uptrend)
    assert w["close"].iloc[-1] == pytest.approx(uptrend["close"].iloc[-1])


# -- Technical score ----------------------------------------------------------
def test_uptrend_scores_higher_than_downtrend(uptrend, downtrend, bench):
    s_up = ta.snapshot(uptrend)
    s_dn = ta.snapshot(downtrend)
    assert technical_score(s_up, None)["score"] > technical_score(s_dn, None)["score"]
    assert technical_score(s_up, None)["score"] > 0
    assert technical_score(s_dn, None)["score"] < 0


def test_direction_labels_mapping(uptrend, downtrend):
    up = technical_score(ta.snapshot(uptrend), None)
    dn = technical_score(ta.snapshot(downtrend), None)
    assert up["direction"] in ("看涨", "偏多", "震荡")
    assert dn["direction"] in ("看跌", "偏空", "震荡")
    assert set(up["components"]) == {"trend", "momentum", "volume", "position", "relative"}


def test_score_is_bounded(uptrend):
    s = ta.snapshot(uptrend)
    out = technical_score(s, {"excess20": 999, "excess60": 999})
    assert -100 <= out["score"] <= 100


# -- Conditional probability --------------------------------------------------
def test_conditional_outcome_requires_history():
    out = conditional_outcome(make_df(60), horizon=5)
    assert out["available"] is False


def test_conditional_outcome_distribution(uptrend):
    out = conditional_outcome(uptrend, horizon=10)
    assert out["available"] is True
    assert out["samples"] > 0
    total = out["p_bull"] + out["p_base"] + out["p_bear"]
    assert abs(total - 100) < 0.5
    assert 0 <= out["p_up"] <= 100
    assert out["p25"] <= out["median"] <= out["p75"]


def test_blend_probability_without_samples():
    out = blend_probability(None, 50)
    assert out["probability"] is None
    assert out["confidence"] == "low"


def test_blend_probability_clamped():
    cond = {"available": True, "p_up": 99, "base_rate": 99, "samples": 5000}
    assert blend_probability(cond, 100)["probability"] <= 80
    cond2 = {"available": True, "p_up": 1, "base_rate": 1, "samples": 5000}
    assert blend_probability(cond2, -100)["probability"] >= 20


def test_blend_confidence_scales_with_samples():
    small = {"available": True, "p_up": 50, "base_rate": 50, "samples": 10}
    big = {"available": True, "p_up": 50, "base_rate": 50, "samples": 400}
    assert blend_probability(small, 0)["confidence"] == "low"
    assert blend_probability(big, 0)["confidence"] == "low"  # 无偏离 → 低置信


# -- Levels & scenarios -------------------------------------------------------
def test_key_levels_ordering(uptrend):
    s = ta.snapshot(uptrend)
    lv = key_levels(uptrend, s)
    price = lv["price"]
    assert all(x["price"] < price for x in lv["supports"])
    assert all(x["price"] > price for x in lv["resistances"])
    assert lv["suggested_stop"] < price
    if lv["supports"]:
        assert lv["suggested_stop"] <= lv["supports"][0]["price"]


def test_scenarios_probabilities_sum(uptrend):
    cond = conditional_outcome(uptrend, horizon=10)
    rows = scenario_plan(float(uptrend["close"].iloc[-1]), cond, 3.0)
    assert len(rows) == 3
    assert sum(r["probability"] for r in rows) == pytest.approx(100, abs=0.5)
    assert rows[0]["target"] > rows[1]["target"] > rows[2]["target"]


def test_scenario_cap_limits_extremes(uptrend):
    fake = {
        "available": True, "horizon": 10, "p_bull": 40, "p_base": 30,
        "p_bear": 30, "p25": -90, "median": 0, "p75": 90,
    }
    rows = scenario_plan(100.0, fake, 2.0)
    assert rows[0]["return_pct"] <= 20
    assert rows[2]["return_pct"] >= -20


# -- Multi timeframe & market -------------------------------------------------
def test_timeframe_trend_directions(uptrend, downtrend):
    assert timeframe_trend(uptrend)["direction"] == "看涨"
    assert timeframe_trend(downtrend)["direction"] == "看跌"


def test_multi_timeframe_has_three_periods(uptrend):
    out = multi_timeframe(uptrend)
    assert set(out["timeframes"]) == {"日线", "周线", "月线"}
    assert out["resonance"] in ("多头共振", "空头共振", "偏多共振", "偏空共振", "周期分歧")


def test_market_regime_states(bench):
    up = market_regime(make_df(300, drift=0.01, seed=3, start=4000))
    dn = market_regime(make_df(300, drift=-0.01, seed=4, start=4000))
    assert up["state"] == "进攻"
    assert dn["state"] == "防守"
    assert up["bias"] > 0 > dn["bias"]


def test_relative_strength_excess(uptrend, bench):
    rel = relative_strength(uptrend, bench)
    assert rel["available"] is True
    assert rel["excess20"] is not None
    assert len(rel["series"]) > 10


def test_relative_strength_insufficient_data():
    assert relative_strength(make_df(10), make_df(10))["available"] is False


# -- Aggregation --------------------------------------------------------------
def test_build_stock_forecast_shape(uptrend, bench):
    out = build_stock_forecast(uptrend, bench, horizon=10)
    assert out["available"] is True
    assert out["probability"]["probability"] is not None
    assert out["multi_timeframe"]["resonance"]
    assert isinstance(out["signals"], list)
    assert out["levels"]["price"] > 0
    assert out["technical"]["score"] == pytest.approx(
        out["technical"]["score"], abs=0.1)


def test_build_stock_forecast_rejects_short_history():
    out = build_stock_forecast(make_df(20), None)
    assert out["available"] is False


def test_signals_have_valid_shape(uptrend, bench):
    out = build_stock_forecast(uptrend, bench)
    for sig in out["signals"]:
        assert sig["direction"] in ("bull", "bear")
        assert sig["group"] and sig["text"]


def test_to_df_accepts_objects():
    class K:
        def __init__(self, d, c):
            self.date, self.open, self.high, self.low = d, c, c, c
            self.close, self.volume, self.amount = c, 1, 1

    df = _to_df([K("2024-01-02", 10), K("2024-01-03", 11)])
    assert list(df["close"]) == [10, 11]
    assert df["date"].iloc[0] == "2024-01-02"


# -- Industry -----------------------------------------------------------------
def test_industry_metrics_basic():
    df = pd.DataFrame({
        "date": pd.bdate_range("2025-01-01", periods=200).strftime("%Y-%m-%d"),
        "close": np.linspace(100, 150, 200),
        "high": np.linspace(101, 151, 200),
        "low": np.linspace(99, 149, 200),
    })
    m = industry_metrics(df)
    assert m["r20"] > 0 and m["r60"] > 0
    assert m["above_ma20"] is True
    assert 0 <= m["pos250"] <= 100


def test_quadrants_classification():
    rows = [
        {"code": "1", "name": "双强", "r20": 5, "r60": 10},
        {"code": "2", "name": "反弹", "r20": 5, "r60": -10},
        {"code": "3", "name": "双弱", "r20": -5, "r60": -10},
        {"code": "4", "name": "回调", "r20": -5, "r60": 10},
        {"code": "5", "name": "缺数据", "r20": None, "r60": 10},
    ]
    out = _quadrants(rows)
    assert len(out) == 4
    labels = {r["name"]: r["quadrant"] for r in out}
    assert labels == {"双强": 1, "反弹": 2, "双弱": 3, "回调": 4}


def test_sw_industry_list_is_complete():
    assert len(SW_LEVEL1) == 31
    codes = [c for c, _ in SW_LEVEL1]
    assert len(set(codes)) == 31
    assert all(v in {n for _, n in SW_LEVEL1} for v in CSRC_TO_SW.values())


def test_bench_returns_relative_strength():
    import pandas as pd
    from app.api.forecast import _bench_returns

    closes = [100.0] * 45 + [100.0 + i for i in range(1, 11)]
    df = pd.DataFrame({"close": closes})
    out = _bench_returns(df)
    assert out["r20"] > 0
    assert out["r60"] is None  # 仅 55 根，不足以计算 60 日收益
    assert _bench_returns(pd.DataFrame()) == {}


def _fake_payload(score=-50.0):
    return {
        "symbol": "300750", "as_of": "2026-10-20", "price": 259.08,
        "change_pct": -2.97, "horizon": 10,
        "technical": {
            "score": score,
            "direction": "看跌" if score < 0 else "看涨",
            "components": {
                "trend": {"score": -78.0, "weight": 0.3, "notes": []},
                "momentum": {"score": -75.0, "weight": 0.25, "notes": []},
                "volume": {"score": -20.0, "weight": 0.15, "notes": []},
                "position": {"score": 10.0, "weight": 0.15, "notes": []},
                "relative": {"score": -60.0, "weight": 0.15, "notes": []},
            },
            "snapshot": {
                "ma": {"ma20": 291.396, "ma60": 348.31},
                "returns": {"r20": -23.15},
                "pos_250": 1.2, "volume_ratio": 1.5075898,
            },
        },
        "probability": {"probability": 42.7, "method": "条件历史频率 50% + 技术评分 30% + 基准频率 20%"},
        "probabilities": [
            {"horizon": 10, "probability": 42.7, "samples": 38, "confidence": "low"},
        ],
        "levels": {
            "price": 259.08,
            "supports": [{"label": "20 日低点", "price": 254.55, "distance_pct": -1.75}],
            "resistances": [{"label": "MA20", "price": 291.4, "distance_pct": 12.47}],
            "suggested_stop": 246.21,
        },
        "scenarios": [
            {"key": "bear", "label": "悲观情景", "probability": 44.7,
             "target": 235.66, "return_pct": -9.04, "trigger": "跌破短期支撑且量能放大"},
        ],
        "multi_timeframe": {"resonance": "偏空共振", "summary": "多周期偏空"},
        "market": {"state": "防守", "note": "沪深300 位于 MA20 下方"},
        "industry": {"name": "电力设备", "code": "801730"},
        "relative": {"excess20": -15.83, "excess60": -21.2},
        "sector_relative": {"excess20": -4.82},
    }


def test_rule_explain_is_grounded_in_numbers():
    from app.api.forecast import _rule_explain

    out = _rule_explain(_fake_payload(-50.0))
    text = out["summary"]
    assert "看跌" in text
    assert "-23.15%" in text and "42.7%" in text
    assert "291.40" in text          # 四舍五入到 2 位
    joined = " ".join([text, *out["invalidation"], *out["watch"]])
    assert "1.51" in joined          # 量比保留 2 位
    assert "1.5075898" not in joined  # 不泄露长尾小数
    assert out["invalidation"] and "291.40" in out["invalidation"][0]
    assert any("254.55" in w for w in out["watch"])
    assert out["risk_note"]


def test_rule_explain_bullish_flips_invalidation():
    from app.api.forecast import _rule_explain

    out = _rule_explain(_fake_payload(50.0))
    assert "看涨" in out["summary"]
    assert "跌破" in out["invalidation"][0]
    assert any("289" in w or "291" in w for w in out["watch"])


def test_parse_legu_market_breadth():
    import pandas as pd
    from app.api.forecast import _parse_legu

    df = pd.DataFrame({
        "item": ["上涨", "下跌", "平盘", "涨停", "跌停", "真实涨停", "真实跌停", "活跃度", "统计日期"],
        "value": [3433, 1662, 122, 55, 7, 48, 6, "65.72%", "2026-10-20 15:00:00"],
    })
    out = _parse_legu(df)
    assert out["available"] is True
    assert out["up"] == 3433 and out["down"] == 1662
    assert out["limit_up"] == 55 and out["limit_down"] == 7
    assert out["activity"] == 65.72
    assert out["up_ratio"] == 65.8
    assert _parse_legu(pd.DataFrame())["available"] is False


def test_conditional_distribution_buckets():
    import numpy as np
    import pandas as pd

    from app.forecast import conditional_outcome

    rng = np.random.default_rng(7)
    n = 600
    ret = rng.normal(0, 0.02, n)
    close = pd.Series(100 * np.cumprod(1 + ret))
    df = pd.DataFrame({
        "date": pd.date_range("2023-01-02", periods=n, freq="B").strftime("%Y-%m-%d"),
        "open": close * 0.995, "high": close * 1.01, "low": close * 0.99,
        "close": close, "volume": rng.integers(1e6, 5e6, n),
    })
    cond = conditional_outcome(df, 10)
    assert cond["available"] is True
    dist = cond["distribution"]
    assert len(dist["labels"]) == len(dist["counts"]) == 6
    assert sum(dist["counts"]) == cond["samples"]
    # 近零区间占比与 p_base 同源：-σ~0 与 0~σ 合计约等于 p_base
    near_zero = dist["share"][2] + dist["share"][3]
    assert abs(near_zero - cond["p_base"]) < 0.5
