# ============================================================================
# AI Stock Analysis Platform - Direction Forecast Engine
# ============================================================================
"""个股涨跌研判引擎：技术打分 + 历史条件概率 + 关键价位 + 情景推演。

设计原则：
1. 所有结论由确定性公式从真实行情算出，LLM 不参与生成任何数字；
2. 概率来自「该股历史上相同技术状态下的未来收益分布」，属于条件频率，
   不是预测模型，响应中始终附带样本量与口径说明；
3. 数据不足时明确降级（例如次新股样本 < 60 根 K 线直接给出 low 置信度）。

研判框架（自上而下）：
    市场环境 → 行业相对强度 → 个股趋势/动量/量能/位置 → 关键价位 → 情景
"""
from __future__ import annotations

import asyncio
import logging
import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from app import ta

logger = logging.getLogger(__name__)


# -- Helpers ------------------------------------------------------------------
def _to_df(klines: Any) -> pd.DataFrame:
    """KlineData 列表 / DataFrame → 标准 DataFrame（升序）。"""
    if klines is None:
        return pd.DataFrame()
    if isinstance(klines, pd.DataFrame):
        df = klines.copy()
    else:
        rows = []
        for k in klines:
            if isinstance(k, dict):
                rows.append(k)
            else:
                rows.append({
                    "date": getattr(k, "date", None),
                    "open": getattr(k, "open", None),
                    "high": getattr(k, "high", None),
                    "low": getattr(k, "low", None),
                    "close": getattr(k, "close", None),
                    "volume": getattr(k, "volume", None),
                    "amount": getattr(k, "amount", None),
                })
        df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    for col in ("open", "high", "low", "close", "volume", "amount"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["close"]).sort_values("date").reset_index(drop=True)
    if "volume" not in df.columns:
        df["volume"] = 0.0
    return df


def _f(v: Any) -> Optional[float]:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if math.isnan(f) or math.isinf(f):
        return None
    return f


def _clip(v: float, lo: float = -100.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, v))


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


# -- 分项打分（每项 -100 ~ +100） ---------------------------------------------
def score_trend(s: Dict[str, Any]) -> Dict[str, Any]:
    """均线系统与趋势强度。"""
    price = s.get("price") or 0
    ma = s.get("ma") or {}
    score, notes = 0.0, []

    ma5, ma10, ma20, ma60, ma120 = (ma.get(f"ma{n}") for n in (5, 10, 20, 60, 120))

    if ma20 and ma60:
        if price > ma20 > ma60:
            score += 45; notes.append("价格站上 MA20 且 MA20 在 MA60 上方（多头排列）")
        elif price > ma20 and ma20 < ma60:
            score += 20; notes.append("价格站上 MA20，但 MA20 仍在 MA60 下方（反弹初期）")
        elif price < ma20 < ma60:
            score -= 45; notes.append("价格跌破 MA20 且 MA20 在 MA60 下方（空头排列）")
        elif price < ma20 and ma20 > ma60:
            score -= 25; notes.append("价格跌破 MA20，中期均线仍向上（回调）")
    if ma5 and ma10:
        if ma5 > ma10:
            score += 10
        else:
            score -= 10
    slope20 = s.get("ma20_slope") or 0
    score += _clip(slope20 * 12, -25, 25)
    if ma120:
        if price > ma120:
            score += 10
        else:
            score -= 10
    if s.get("adx") and s["adx"].get("adx") is not None:
        adx_v = s["adx"]["adx"]
        if adx_v >= 25:
            notes.append(f"ADX {adx_v:.1f}，趋势性较强")
    return {"score": _clip(score), "notes": notes}


def score_momentum(s: Dict[str, Any]) -> Dict[str, Any]:
    price = s.get("price") or 0
    macd = s.get("macd") or {}
    rsi = s.get("rsi14")
    score, notes = 0.0, []

    dif, dea, hist = macd.get("dif"), macd.get("dea"), macd.get("hist")
    if dif is not None and dea is not None:
        if dif > dea:
            score += 20; notes.append("MACD DIF 在 DEA 上方")
        else:
            score -= 20; notes.append("MACD DIF 在 DEA 下方")
        if hist is not None:
            score += _clip((hist / max(price, 1e-9)) * 4000, -20, 20)
        if dif > 0 and dea > 0:
            score += 10
        elif dif < 0 and dea < 0:
            score -= 10
    if rsi is not None:
        if rsi >= 75:
            score -= 18; notes.append(f"RSI {rsi:.0f} 超买，短线回吐风险")
        elif rsi >= 55:
            score += 18; notes.append(f"RSI {rsi:.0f} 偏强")
        elif rsi >= 45:
            score += 4
        elif rsi >= 30:
            score -= 14; notes.append(f"RSI {rsi:.0f} 偏弱")
        else:
            score += 8; notes.append(f"RSI {rsi:.0f} 超卖，存在反抽需求")
    ret20 = (s.get("returns") or {}).get("r20")
    if ret20 is not None:
        score += _clip(ret20 * 1.2, -25, 25)
    kdj = s.get("kdj") or {}
    if kdj.get("k") is not None and kdj.get("d") is not None:
        if kdj["k"] > kdj["d"] and kdj["k"] < 80:
            score += 8
        elif kdj["k"] < kdj["d"]:
            score -= 8
    return {"score": _clip(score), "notes": notes}


def score_volume(s: Dict[str, Any]) -> Dict[str, Any]:
    score, notes = 0.0, []
    vr = s.get("volume_ratio")
    ret1 = s.get("change_pct") or 0
    obv_slope = s.get("obv_slope") or 0
    ret20 = (s.get("returns") or {}).get("r20") or 0

    if vr is not None:
        if vr >= 1.5 and ret1 > 0:
            score += 25; notes.append(f"量比 {vr:.2f}，放量上涨（资金进攻）")
        elif vr >= 1.5 and ret1 < 0:
            score -= 25; notes.append(f"量比 {vr:.2f}，放量下跌（抛压释放）")
        elif vr <= 0.7:
            score -= 5; notes.append(f"量比 {vr:.2f}，缩量观望")
    score += _clip(obv_slope * 60, -25, 25)
    if ret20 * obv_slope > 0:
        score += 12; notes.append("量价同向（OBV 与价格趋势一致）")
    elif ret20 * obv_slope < 0:
        score -= 12; notes.append("量价背离（OBV 与价格趋势相反）")
    return {"score": _clip(score), "notes": notes}


def score_position(s: Dict[str, Any]) -> Dict[str, Any]:
    score, notes = 0.0, []
    pct_b = (s.get("boll") or {}).get("pct_b")
    pos = s.get("pos_250")
    price = s.get("price") or 0
    ma20 = (s.get("ma") or {}).get("ma20")
    atr_v = s.get("atr")

    if pct_b is not None:
        if pct_b > 1.0:
            score -= 18; notes.append("价格突破布林上轨，短期偏离过大")
        elif pct_b > 0.75:
            score += 10
        elif pct_b < 0.0:
            score += 12; notes.append("价格跌破布林下轨，超跌")
        elif pct_b < 0.25:
            score -= 10
    if pos is not None:
        if pos >= 90:
            score -= 12; notes.append(f"处于 250 日高位区（{pos:.0f}% 分位）")
        elif pos <= 10:
            score += 12; notes.append(f"处于 250 日低位区（{pos:.0f}% 分位）")
    if ma20 and atr_v:
        dist = (price - ma20) / atr_v
        score += _clip(-dist * 4, -16, 16)
    return {"score": _clip(score), "notes": notes}


def score_relative(rel: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not rel:
        return {"score": 0.0, "notes": []}
    score, notes = 0.0, []
    e20 = rel.get("excess20")
    e60 = rel.get("excess60")
    if e20 is not None:
        score += _clip(e20 * 3, -30, 30)
        notes.append(f"近 20 日相对{rel.get('bench_name', '基准')}超额 {e20:+.1f}%")
    if e60 is not None:
        score += _clip(e60 * 1.5, -30, 30)
    return {"score": _clip(score), "notes": notes}


WEIGHTS = {
    "trend": 0.30,
    "momentum": 0.25,
    "volume": 0.15,
    "position": 0.15,
    "relative": 0.15,
}

DIRECTION_LABELS = [
    (40, "看涨", "bullish"),
    (15, "偏多", "lean_bullish"),
    (-15, "震荡", "neutral"),
    (-40, "偏空", "lean_bearish"),
    (-101, "看跌", "bearish"),
]


def technical_score(s: Dict[str, Any], rel: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    parts = {
        "trend": score_trend(s),
        "momentum": score_momentum(s),
        "volume": score_volume(s),
        "position": score_position(s),
        "relative": score_relative(rel),
    }
    total = sum(parts[k]["score"] * WEIGHTS[k] for k in WEIGHTS)
    total = round(_clip(total), 1)
    label, key = "震荡", "neutral"
    for threshold, text, code in DIRECTION_LABELS:
        if total >= threshold:
            label, key = text, code
            break
    return {
        "score": total,
        "direction": label,
        "direction_key": key,
        "components": {
            k: {"score": round(v["score"], 1), "weight": WEIGHTS[k], "notes": v["notes"]}
            for k, v in parts.items()
        },
    }


# -- 历史条件概率 --------------------------------------------------------------
def _state_of(row: Dict[str, Any]) -> Tuple[int, int, int]:
    """(趋势状态, RSI 分桶, MACD 柱符号)，用于匹配历史相似状态。"""
    trend = 0
    if row["close"] > row["ma20"] and row["ma20"] > row["ma60"]:
        trend = 1
    elif row["close"] < row["ma20"] and row["ma20"] < row["ma60"]:
        trend = -1
    r = row["rsi"]
    rsi_bucket = 0 if r < 35 else 1 if r < 50 else 2 if r < 65 else 3
    return trend, rsi_bucket, 1 if row["hist"] > 0 else -1


def conditional_outcome(df: pd.DataFrame, horizon: int = 10) -> Dict[str, Any]:
    """统计历史上与当前技术状态相同/相近时，未来 horizon 日的收益分布。"""
    if df is None or len(df) < 120:
        return {"available": False, "reason": "历史样本不足 120 根 K 线"}

    close = df["close"].astype(float)
    ma20 = ta.sma(close, 20)
    ma60 = ta.sma(close, 60)
    rsi14 = ta.rsi(close, 14)
    hist = ta.macd(close)["hist"]

    work = pd.DataFrame({
        "close": close, "ma20": ma20, "ma60": ma60,
        "rsi": rsi14, "hist": hist,
    }).dropna()
    if len(work) < 120:
        return {"available": False, "reason": "有效样本不足"}

    fwd = close.shift(-horizon) / close - 1.0
    work["fwd"] = fwd.loc[work.index].values
    work = work.dropna(subset=["fwd"])
    if work.empty:
        return {"available": False, "reason": "无法计算未来收益"}

    cur = work.iloc[-1]
    cur_state = _state_of(cur)
    fwd_all = work["fwd"]

    # 逐级放宽匹配条件，保证样本量
    candidates = [
        (cur_state, "趋势+RSI+MACD 同态"),
        ((cur_state[0], cur_state[1], 0), "趋势+RSI 同态"),
        ((cur_state[0], 0, 0), "仅趋势同态"),
    ]
    chosen, match_desc = None, ""
    for state, desc in candidates:
        mask = work.apply(lambda r: _state_of(r) == state if state[2] != 0 else _state_of(r)[:2] == state[:2], axis=1)
        sample = work.loc[mask, "fwd"]
        if len(sample) >= 30:
            chosen, match_desc = sample, desc
            break
    if chosen is None:
        chosen, match_desc = fwd_all, "全样本（未找到足够相似状态）"

    sample = chosen.dropna()
    n = int(len(sample))
    atr_pct = float((ta.atr(df["high"], df["low"], close).iloc[-1]) / close.iloc[-1] * 100) if len(df) > 14 else 2.0
    band = max(atr_pct * math.sqrt(horizon / 5) / 100 * 0.8, 0.01)

    p_up = float((sample > 0).mean())
    p_bull = float((sample > band).mean())
    p_bear = float((sample < -band).mean())
    p_base = max(0.0, 1.0 - p_bull - p_bear)
    base_rate = float((fwd_all > 0).mean())

    # 收益分布直方图（以 ±1σ / ±2σ 分桶，σ 与上文 band 同源）
    edges = [-math.inf, -2 * band, -band, 0.0, band, 2 * band, math.inf]
    labels = ["≤-2σ", "-2σ~-σ", "-σ~0", "0~σ", "σ~2σ", ">2σ"]
    counts: List[int] = []
    for i in range(6):
        lo, hi = edges[i], edges[i + 1]
        counts.append(int(((sample >= lo) & (sample < hi)).sum()))

    return {
        "available": True,
        "horizon": horizon,
        "samples": n,
        "match": match_desc,
        "p_up": round(p_up * 100, 1),
        "p_bull": round(p_bull * 100, 1),
        "p_base": round(p_base * 100, 1),
        "p_bear": round(p_bear * 100, 1),
        "median": round(float(sample.median()) * 100, 2),
        "p25": round(float(sample.quantile(0.25)) * 100, 2),
        "p75": round(float(sample.quantile(0.75)) * 100, 2),
        "mean": round(float(sample.mean()) * 100, 2),
        "worst": round(float(sample.min()) * 100, 2),
        "best": round(float(sample.max()) * 100, 2),
        "base_rate": round(base_rate * 100, 1),
        "all_samples": int(len(fwd_all)),
        "distribution": {
            "labels": labels,
            "counts": counts,
            "band": round(band * 100, 2),
            "share": [round(c / n * 100, 1) if n else 0.0 for c in counts],
        },
    }


def blend_probability(cond: Optional[Dict[str, Any]], score: float, market_bias: float = 0.0) -> Dict[str, Any]:
    """条件频率 50% + 技术打分 30% + 无条件基准 20%，按样本量向 50% 收缩。"""
    if not cond or not cond.get("available"):
        return {"probability": None, "confidence": "low", "method": "样本不足，未给出概率"}

    prior = cond["p_up"] / 100.0
    model = _sigmoid((score + market_bias) / 28.0)
    base = cond["base_rate"] / 100.0
    raw = 0.5 * prior + 0.3 * model + 0.2 * base
    shrink = cond["samples"] / (cond["samples"] + 60.0)
    p = 0.5 + (raw - 0.5) * shrink
    p = max(0.2, min(0.8, p))
    confidence = "high" if (cond["samples"] >= 150 and abs(p - 0.5) >= 0.08) else \
                 "medium" if (cond["samples"] >= 60 and abs(p - 0.5) >= 0.04) else "low"
    return {
        "probability": round(p * 100, 1),
        "confidence": confidence,
        "method": "条件历史频率 50% + 技术评分 30% + 基准频率 20%（按样本量收缩）",
    }


# -- 关键价位 ------------------------------------------------------------------
def _swings(df: pd.DataFrame, left: int = 3, right: int = 3) -> Tuple[List[float], List[float]]:
    highs, lows = [], []
    h, l = df["high"].to_numpy(), df["low"].to_numpy()
    for i in range(left, len(df) - right):
        window_h = h[i - left:i + right + 1]
        window_l = l[i - left:i + right + 1]
        if h[i] == window_h.max():
            highs.append(float(h[i]))
        if l[i] == window_l.min():
            lows.append(float(l[i]))
    return highs, lows


def key_levels(df: pd.DataFrame, s: Dict[str, Any]) -> Dict[str, Any]:
    price = float(df["close"].iloc[-1])
    atr_v = s.get("atr") or price * 0.02
    highs, lows = _swings(df.tail(250))

    supports: List[Dict[str, Any]] = []
    resistances: List[Dict[str, Any]] = []

    for name, val in (
        ("MA20", (s.get("ma") or {}).get("ma20")),
        ("MA60", (s.get("ma") or {}).get("ma60")),
        ("布林下轨", (s.get("boll") or {}).get("low")),
    ):
        if val:
            (supports if val < price else resistances).append({"label": name, "price": float(val)})
    if (s.get("boll") or {}).get("up"):
        v = float(s["boll"]["up"])
        if v > price:
            resistances.append({"label": "布林上轨", "price": v})

    for v in lows:
        if v < price * 0.999:
            supports.append({"label": "前低", "price": v})
    for v in highs:
        if v > price * 1.001:
            resistances.append({"label": "前高", "price": v})

    # 摆动点识别需要右侧确认，最近几根 K 线不会成为 pivot，
    # 因此补充区间低点作为兜底支撑。
    for label, series in (("20 日低点", df["low"].tail(20)),
                          ("60 日低点", df["low"].tail(60)),
                          ("120 日低点", df["low"].tail(120))):
        if len(series):
            v = float(series.min())
            if v < price * 0.999:
                supports.append({"label": label, "price": v})

    def _dedupe(items, reverse):
        items = sorted(items, key=lambda x: x["price"], reverse=reverse)
        out: List[Dict[str, Any]] = []
        for it in items:
            if all(abs(it["price"] - o["price"]) > atr_v * 0.35 for o in out):
                out.append(it)
            if len(out) == 3:
                break
        for it in out:
            it["price"] = round(it["price"], 2)
            it["distance_pct"] = round((it["price"] / price - 1) * 100, 2)
        return out

    sup = _dedupe(supports, True)
    res = _dedupe(resistances, False)
    stop = (sup[0]["price"] - atr_v * 0.8) if sup else (price - atr_v * 2)
    return {
        "price": round(price, 2),
        "atr": round(atr_v, 3),
        "supports": sup,
        "resistances": res,
        "suggested_stop": round(max(stop, 0.01), 2),
        "stop_pct": round((max(stop, 0.01) / price - 1) * 100, 2),
    }


def scenario_plan(price: float, cond: Optional[Dict[str, Any]], atr_pct: float) -> List[Dict[str, Any]]:
    if not cond or not cond.get("available"):
        return []
    cap = max(atr_pct * math.sqrt(cond["horizon"] / 5) * 1.6, 3.0)
    rows = []
    for key, label, ret_key, prob_key in (
        ("bull", "乐观情景", "p75", "p_bull"),
        ("base", "中性情景", "median", "p_base"),
        ("bear", "悲观情景", "p25", "p_bear"),
    ):
        ret = cond.get(ret_key) or 0.0
        ret = max(-cap, min(cap, ret))
        rows.append({
            "key": key,
            "label": label,
            "probability": round(cond[prob_key], 1),
            "target": round(price * (1 + ret / 100), 2),
            "return_pct": round(ret, 2),
            "trigger": {
                "bull": "放量站稳 MA20 / 突破前高",
                "base": "维持区间震荡，量能平稳",
                "bear": "跌破短期支撑且量能放大",
            }[key],
        })
    return rows


# -- 多周期 --------------------------------------------------------------------
def timeframe_trend(df: pd.DataFrame) -> Dict[str, Any]:
    if df is None or len(df) < 25:
        return {"direction": "数据不足", "score": 0, "signal": "样本不足"}
    close = df["close"].astype(float)
    ma5, ma10, ma20 = ta.sma(close, 5), ta.sma(close, 10), ta.sma(close, 20)
    macd_hist = ta.macd(close)["hist"]
    price = float(close.iloc[-1])
    score = 0.0
    if not pd.isna(ma20.iloc[-1]):
        score += 25 if price > ma20.iloc[-1] else -25
    if not pd.isna(ma5.iloc[-1]) and not pd.isna(ma10.iloc[-1]):
        score += 20 if ma5.iloc[-1] > ma10.iloc[-1] else -20
    if not pd.isna(macd_hist.iloc[-1]):
        score += 20 if macd_hist.iloc[-1] > 0 else -20
    r20 = (price / float(close.iloc[-21]) - 1) * 100 if len(close) > 21 else 0.0
    score += max(-20, min(20, r20))
    direction = "看涨" if score >= 20 else "看跌" if score <= -20 else "震荡"
    bits = []
    if len(close) > 20 and not pd.isna(ma20.iloc[-1]):
        bits.append("站上 MA20" if price > ma20.iloc[-1] else "位于 MA20 下方")
    bits.append("MACD 柱翻红" if macd_hist.iloc[-1] > 0 else "MACD 柱为绿")
    bits.append(f"区间 {r20:+.1f}%")
    return {
        "direction": direction,
        "score": round(score, 1),
        "signal": " · ".join(bits),
    }


def multi_timeframe(df: pd.DataFrame) -> Dict[str, Any]:
    frames = {
        "日线": df,
        "周线": ta.resample_ohlc(df, "W-FRI"),
        "月线": ta.resample_ohlc(df, "ME"),
    }
    rows = {name: timeframe_trend(f) for name, f in frames.items()}
    scores = [r["score"] for r in rows.values()]
    agree = sum(1 for x in scores if x > 0) if sum(scores) > 0 else -sum(1 for x in scores if x < 0)
    bullish = sum(1 for x in scores if x >= 20)
    bearish = sum(1 for x in scores if x <= -20)
    if bullish == 3:
        resonance, text = "多头共振", "日/周/月三周期同向看涨"
    elif bearish == 3:
        resonance, text = "空头共振", "日/周/月三周期同向看跌"
    elif bullish >= 2:
        resonance, text = "偏多共振", "多数周期偏多，留意短周期背离"
    elif bearish >= 2:
        resonance, text = "偏空共振", "多数周期偏空，反弹需确认"
    else:
        resonance, text = "周期分歧", "周期方向不一致，趋势尚不明朗"
    return {"timeframes": rows, "resonance": resonance, "summary": text}


# -- 市场环境 ------------------------------------------------------------------
def market_regime(index_df: pd.DataFrame, name: str = "沪深300") -> Dict[str, Any]:
    df = _to_df(index_df)
    if df.empty or len(df) < 65:
        return {"available": False, "name": name}
    close = df["close"].astype(float)
    price = float(close.iloc[-1])
    ma20 = float(ta.sma(close, 20).iloc[-1])
    ma60 = float(ta.sma(close, 60).iloc[-1])
    r20 = (price / float(close.iloc[-21]) - 1) * 100
    r60 = (price / float(close.iloc[-61]) - 1) * 100
    vol = ta.annualized_volatility(close.tail(60))

    score = 0.0
    score += 30 if price > ma20 else -30
    score += 30 if price > ma60 else -30
    score += max(-25, min(25, r20 * 1.5))
    score += max(-15, min(15, r60 * 0.5))
    if score >= 30:
        state, bias = "进攻", 12.0
    elif score <= -30:
        state, bias = "防守", -12.0
    else:
        state, bias = "震荡", 0.0
    return {
        "available": True,
        "name": name,
        "price": round(price, 2),
        "change_pct": round((price / float(close.iloc[-2]) - 1) * 100, 2),
        "above_ma20": price > ma20,
        "above_ma60": price > ma60,
        "ret20": round(r20, 2),
        "ret60": round(r60, 2),
        "volatility": round(vol, 2),
        "score": round(score, 1),
        "state": state,
        "bias": bias,
        "note": f"{name}位于 MA20 {'上方' if price > ma20 else '下方'}、"
                f"MA60 {'上方' if price > ma60 else '下方'}，近 20 日 {r20:+.1f}%。",
    }


def relative_strength(stock_df: pd.DataFrame, bench_df: pd.DataFrame,
                      bench_name: str = "沪深300", limit: int = 250) -> Dict[str, Any]:
    a = _to_df(stock_df)
    b = _to_df(bench_df)
    if a.empty or b.empty or "date" not in a or "close" not in a or "date" not in b or "close" not in b:
        return {"available": False}
    a = a[["date", "close"]].rename(columns={"close": "stock"})
    b = b[["date", "close"]].rename(columns={"close": "bench"})
    m = a.merge(b, on="date", how="inner").sort_values("date")
    if len(m) < 25:
        return {"available": False}
    m = m.tail(limit).reset_index(drop=True)
    base_s, base_b = float(m["stock"].iloc[0]), float(m["bench"].iloc[0])
    m["stock_norm"] = m["stock"] / base_s * 100
    m["bench_norm"] = m["bench"] / base_b * 100
    m["rs"] = m["stock_norm"] / m["bench_norm"] * 100

    def _excess(n: int) -> Optional[float]:
        if len(m) <= n:
            return None
        s = float(m["stock"].iloc[-1]) / float(m["stock"].iloc[-1 - n]) - 1
        bch = float(m["bench"].iloc[-1]) / float(m["bench"].iloc[-1 - n]) - 1
        return round((s - bch) * 100, 2)

    rs_slope = ta.slope(m["rs"], 10)
    return {
        "available": True,
        "bench_name": bench_name,
        "series": [
            {"date": r["date"], "rs": round(float(r["rs"]), 2),
             "stock": round(float(r["stock_norm"]), 2),
             "bench": round(float(r["bench_norm"]), 2)}
            for _, r in m.iloc[::max(1, len(m) // 180)].iterrows()
        ],
        "excess20": _excess(20),
        "excess60": _excess(60),
        "rs_slope": round(rs_slope, 2),
        "outperform": (_excess(20) or 0) > 0,
    }


# -- 综合 ----------------------------------------------------------------------
def build_signals(s: Dict[str, Any], rel: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    signals: List[Dict[str, Any]] = []
    price = s.get("price") or 0
    ma = s.get("ma") or {}

    if ma.get("ma20") and ma.get("ma60"):
        if price > ma["ma20"] > ma["ma60"]:
            signals.append({"group": "趋势", "direction": "bull", "text": "均线多头排列（价 > MA20 > MA60）"})
        elif price < ma["ma20"] < ma["ma60"]:
            signals.append({"group": "趋势", "direction": "bear", "text": "均线空头排列（价 < MA20 < MA60）"})
    macd = s.get("macd") or {}
    if macd.get("hist") is not None:
        if macd["hist"] > 0 and macd.get("dif", 0) > macd.get("dea", 0):
            signals.append({"group": "动量", "direction": "bull", "text": "MACD 金叉且柱体为正"})
        elif macd["hist"] < 0:
            signals.append({"group": "动量", "direction": "bear", "text": "MACD 柱体为负"})
    rsi = s.get("rsi14")
    if rsi is not None:
        if rsi > 70:
            signals.append({"group": "超买超卖", "direction": "bear", "text": f"RSI14 {rsi:.0f} 进入超买区"})
        elif rsi < 30:
            signals.append({"group": "超买超卖", "direction": "bull", "text": f"RSI14 {rsi:.0f} 进入超卖区"})
    boll = s.get("boll") or {}
    if boll.get("pct_b") is not None:
        if boll["pct_b"] > 1:
            signals.append({"group": "波动", "direction": "bear", "text": "价格突破布林上轨，偏离均值"})
        elif boll["pct_b"] < 0:
            signals.append({"group": "波动", "direction": "bull", "text": "价格跌破布林下轨，超跌反抽"})
    vr = s.get("volume_ratio")
    chg = s.get("change_pct") or 0
    if vr is not None and vr >= 1.5:
        signals.append({
            "group": "量能", "direction": "bull" if chg > 0 else "bear",
            "text": f"量比 {vr:.2f}，{'放量上涨' if chg > 0 else '放量下跌'}",
        })
    r20 = (s.get("returns") or {}).get("r20")
    if r20 is not None and r20 >= 10:
        signals.append({"group": "动量", "direction": "bull", "text": f"20 日涨幅 {r20:+.1f}%，强势"})
    elif r20 is not None and r20 <= -10:
        signals.append({"group": "动量", "direction": "bear", "text": f"20 日跌幅 {r20:+.1f}%，弱势"})
    if s.get("high_20") and price >= s["high_20"] * 0.999:
        signals.append({"group": "突破", "direction": "bull", "text": "创 20 日新高"})
    if s.get("low_20") and price <= s["low_20"] * 1.001:
        signals.append({"group": "突破", "direction": "bear", "text": "创 20 日新低"})
    if rel and rel.get("excess20") is not None:
        e = rel["excess20"]
        signals.append({
            "group": "相对强度",
            "direction": "bull" if e > 0 else "bear",
            "text": f"相对{rel.get('bench_name', '基准')} 20 日超额 {e:+.1f}%",
        })
    return signals


def build_stock_forecast(
    df: pd.DataFrame,
    bench_df: Optional[pd.DataFrame] = None,
    sector_rel: Optional[Dict[str, Any]] = None,
    bench_name: str = "沪深300",
    horizon: int = 10,
) -> Dict[str, Any]:
    df = _to_df(df)
    if len(df) < 30:
        return {"available": False, "reason": "K 线数据不足 30 根，无法研判"}

    s = ta.snapshot(df)
    has_bench = bench_df is not None and not bench_df.empty
    bench = market_regime(bench_df, bench_name) if has_bench else {"available": False, "name": bench_name}
    rel = relative_strength(df, bench_df, bench_name) if has_bench else None

    tech = technical_score(s, rel)
    cond = conditional_outcome(df, horizon)
    prob = blend_probability(cond, tech["score"], (bench or {}).get("bias", 0.0) if bench else 0.0)
    levels = key_levels(df, s)
    mtf = multi_timeframe(df)
    scenarios = scenario_plan(float(df["close"].iloc[-1]), cond, s.get("atr_pct") or 2.0)
    signals = build_signals(s, rel)

    bull = sum(1 for x in signals if x["direction"] == "bull")
    bear = sum(1 for x in signals if x["direction"] == "bear")
    if mtf["resonance"] in ("多头共振",):
        tech["score"] = round(_clip(tech["score"] + 6), 1)
    elif mtf["resonance"] in ("空头共振",):
        tech["score"] = round(_clip(tech["score"] - 6), 1)

    return {
        "available": True,
        "price": s["price"],
        "change_pct": s["change_pct"],
        "horizon": horizon,
        "technical": {
            "score": tech["score"],
            "direction": tech["direction"],
            "direction_key": tech["direction_key"],
            "components": tech["components"],
            "snapshot": {k: v for k, v in s.items() if k != "series"},
        },
        "probability": prob,
        "conditional": cond,
        "levels": levels,
        "scenarios": scenarios,
        "multi_timeframe": mtf,
        "signals": signals,
        "signal_summary": {
            "bull": bull, "bear": bear,
            "balance": "偏多" if bull > bear else "偏空" if bear > bull else "均衡",
        },
        "market": bench,
        "relative": rel,
        "sector_relative": sector_rel,
        "disclaimer": "概率为历史条件频率与规则模型的合成结果，不构成投资建议。",
    }
