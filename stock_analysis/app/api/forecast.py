# ============================================================================
# AI Stock Analysis Platform - Forecast & Industry API
# ============================================================================
"""个股涨跌研判与行业分析的 HTTP 接口。

- GET /api/v1/forecast/stock      个股研判（技术/概率/关键位/情景/多周期/相对强度）
- GET /api/v1/forecast/market     市场环境（沪深300 趋势状态）
- GET /api/v1/forecast/industries 申万 31 个一级行业总览与轮动象限
- GET /api/v1/forecast/industry   单个行业详情（动量/估值/广度/成分股）
"""
from __future__ import annotations

import asyncio
import logging
import math
import re
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, Query

from app import industry as ind
from app import ta
from app.adapters import DataType
from app.forecast import (
    _to_df, blend_probability, build_stock_forecast, conditional_outcome,
    market_regime, relative_strength,
)

logger = logging.getLogger(__name__)
router = APIRouter()

_BENCH_CACHE: Dict[str, Any] = {"ts": 0.0, "df": None, "fail_ts": 0.0}
_BENCH_TTL = 600.0
_BENCH_FAIL_TTL = 60.0
_CODE_RE = re.compile(r"(\d{6})(?:\.(SH|SZ|BJ|sh|sz|bj))?")


async def _resolve_code(symbol: str) -> str:
    m = _CODE_RE.search(symbol or "")
    if m:
        return m.group(1)
    from app.api.data import resolve_stock_symbol
    resolved = await resolve_stock_symbol(symbol)
    m = _CODE_RE.search(resolved or "")
    if not m:
        raise HTTPException(status_code=404, detail=f"未找到股票: {symbol}")
    return m.group(1)


def _clean(obj: Any) -> Any:
    """递归转换 numpy / pandas 标量，避免 JSON 序列化失败。"""
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return None if not np.isfinite(obj) else float(obj)
    return obj


def _json_num(v: Any, nd: int = 3) -> Optional[float]:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if math.isnan(f) or math.isinf(f):
        return None
    return round(f, nd)


async def bench_history(symbol: str = "sh000300", name: str = "沪深300") -> pd.DataFrame:
    now = time.time()
    if _BENCH_CACHE["df"] is not None and now - _BENCH_CACHE["ts"] < _BENCH_TTL:
        return _BENCH_CACHE["df"]
    if _BENCH_CACHE["df"] is None and now - _BENCH_CACHE["fail_ts"] < _BENCH_FAIL_TTL:
        return pd.DataFrame()

    def _load():
        import akshare as ak
        return ak.stock_zh_index_daily(symbol=symbol)

    try:
        df = await asyncio.wait_for(asyncio.to_thread(_load), timeout=8.0)
        df = _to_df(df)
        _BENCH_CACHE.update({"ts": now, "df": df, "fail_ts": 0.0})
        return df
    except asyncio.TimeoutError:
        logger.warning("bench history timed out, using cached/empty")
        _BENCH_CACHE["fail_ts"] = now
        return _BENCH_CACHE["df"] if _BENCH_CACHE["df"] is not None else pd.DataFrame()
    except Exception as exc:
        logger.warning(f"bench history failed: {exc}")
        _BENCH_CACHE["fail_ts"] = now
        return _BENCH_CACHE["df"] if _BENCH_CACHE["df"] is not None else pd.DataFrame()


def _bench_returns(bench: pd.DataFrame) -> Dict[str, Optional[float]]:
    """基准指数的区间收益，用于把行业动量换算成相对强弱。"""
    out: Dict[str, Optional[float]] = {}
    if bench is None or bench.empty or "close" not in bench:
        return out
    c = bench["close"].astype(float)
    for n in (20, 60):
        out[f"r{n}"] = (
            round((float(c.iloc[-1]) / float(c.iloc[-1 - n]) - 1) * 100, 2)
            if len(c) > n else None
        )
    return out


async def _fetch_klines(code: str, days: int) -> pd.DataFrame:
    from app.api.data import get_failover

    mgr = get_failover()
    end = datetime.now().strftime("%Y%m%d")
    start = (datetime.now() - timedelta(days=days)).strftime("%Y%m%d")
    try:
        kl = await asyncio.wait_for(
            mgr.fetch_with_failover(
                DataType.KLINE, code, frequency="D", start_date=start, end_date=end,
            ),
            timeout=20.0,
        )
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail=f"行情获取超时: {code}")
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"行情获取失败: {exc}")
    df = _to_df(kl)
    if df.empty:
        raise HTTPException(status_code=422, detail=f"未获取到 {code} 的 K 线数据")
    return df


def _chart_payload(df: pd.DataFrame, limit: int = 250) -> Dict[str, Any]:
    d = df.tail(limit).reset_index(drop=True)
    close = d["close"].astype(float)
    vol = d["volume"].astype(float) if "volume" in d.columns else pd.Series([0.0] * len(d))
    ma20 = ta.sma(close, 20)
    ma60 = ta.sma(close, 60)
    boll = ta.boll(close)
    macd = ta.macd(close)
    rsi14 = ta.rsi(close, 14)

    def _arr(series: pd.Series, nd: int = 2) -> List[Optional[float]]:
        return [None if pd.isna(x) else round(float(x), nd) for x in series]

    return {
        "dates": d["date"].tolist(),
        "kline": [
            [round(float(o), 2), round(float(c), 2), round(float(l), 2), round(float(h), 2)]
            for o, c, l, h in zip(d["open"], d["close"], d["low"], d["high"])
        ],
        "volume": [int(v) if not pd.isna(v) else 0 for v in vol],
        "ma20": _arr(ma20),
        "ma60": _arr(ma60),
        "boll_up": _arr(boll["up"]),
        "boll_low": _arr(boll["low"]),
        "dif": _arr(macd["dif"], 4),
        "dea": _arr(macd["dea"], 4),
        "hist": _arr(macd["hist"], 4),
        "rsi14": _arr(rsi14, 1),
    }


async def _sector_for_stock(code: str):
    """行业映射与历史并行获取；失败时返回 (None, None)，不阻塞研判主流程。"""
    try:
        info = await asyncio.wait_for(ind.resolve_stock_industry(code), timeout=6.0)
    except asyncio.TimeoutError:
        logger.warning(f"sector resolve timed out for {code}")
        return None, None
    except Exception as exc:
        logger.warning(f"sector resolve failed for {code}: {exc}")
        return None, None
    if not info:
        return None, None

    try:
        hist = await asyncio.wait_for(
            ind.get_industry_history(info["code"], days=300), timeout=10.0,
        )
    except asyncio.TimeoutError:
        logger.warning(f"sector history timed out for {code}")
        return info, None
    except Exception as exc:
        logger.warning(f"sector history failed for {code}: {exc}")
        return info, None
    return info, hist


async def _compute_stock_payload(code: str, horizon: int, days: int) -> Dict[str, Any]:
    """个股研判的唯一计算入口，保证 /stock 与 /explain 的数字完全一致。"""
    (df, bench), (sector_info, hist) = await asyncio.gather(
        asyncio.gather(_fetch_klines(code, days), bench_history()),
        _sector_for_stock(code),
    )

    sector_rel = None
    if sector_info and hist is not None and not hist.empty and not bench.empty:
        sector_rel = relative_strength(hist, bench, f"申万{sector_info['name']}行业")
        sector_rel["industry"] = sector_info
        sector_rel["metrics"] = ind.industry_metrics(hist)

    payload = build_stock_forecast(df, bench, sector_rel=sector_rel, horizon=horizon)
    if not payload.get("available"):
        raise HTTPException(status_code=422, detail=payload.get("reason", "数据不足"))

    # 多周期概率（5/10/20 日），条件分布计算成本低，一次返回
    bias = (payload.get("market") or {}).get("bias", 0.0)
    score = payload["technical"]["score"]
    horizon_probs = []
    for h in (5, 10, 20):
        cond = conditional_outcome(df, h)
        prob = blend_probability(cond, score, bias)
        horizon_probs.append({
            "horizon": h,
            "probability": prob["probability"],
            "confidence": prob["confidence"],
            "samples": cond.get("samples") if cond.get("available") else 0,
            "p_bull": cond.get("p_bull") if cond.get("available") else None,
            "p_bear": cond.get("p_bear") if cond.get("available") else None,
            "median": cond.get("median") if cond.get("available") else None,
        })

    payload.update({
        "symbol": code,
        "probabilities": horizon_probs,
        "chart": _chart_payload(df),
        "industry": sector_info,
        "as_of": df["date"].iloc[-1],
    })
    return payload


@router.get("/accuracy")
async def signal_accuracy(
    symbol: str = Query(..., description="股票代码，如 300750.SZ"),
    horizon: int = Query(10, ge=3, le=30, description="前瞻交易日数"),
    step: int = Query(5, ge=1, le=30, description="调仓间隔（根K线）"),
    lookback: int = Query(250, ge=60, le=600, description="回看的K线根数"),
):
    """研判信号的历史准确率（walk-forward，无未来数据泄漏）。"""
    from app.signal_eval import evaluate_signal_accuracy

    code = await _resolve_code(symbol)
    try:
        df, bench = await asyncio.wait_for(
            asyncio.gather(_fetch_klines(code, 1500), bench_history()),
            timeout=30.0,
        )
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail=f"准确率回测数据获取超时: {code}")
    except HTTPException:
        raise
    except Exception as exc:
        logger.warning(f"accuracy fetch failed for {code}: {exc}")
        raise HTTPException(status_code=422, detail=f"行情数据获取失败：{exc}")

    result = evaluate_signal_accuracy(
        df, bench, horizon=horizon, step=step, lookback=lookback,
    )
    result["symbol"] = code
    result["as_of"] = df["date"].iloc[-1] if not df.empty else None
    return _clean(result)


@router.get("/stock")
async def stock_forecast(
    symbol: str = Query(..., description="股票代码，如 300750.SZ"),
    horizon: int = Query(10, ge=3, le=30, description="研判周期（交易日）"),
    days: int = Query(700, ge=120, le=1500, description="拉取的日历天数"),
):
    """个股涨跌研判：技术打分 + 历史条件概率 + 关键价位 + 情景推演。"""
    code = await _resolve_code(symbol)
    try:
        payload = await asyncio.wait_for(
            _compute_stock_payload(code, horizon, days), timeout=40.0,
        )
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail=f"研判超时，请稍后重试: {code}")
    except HTTPException:
        raise
    except Exception as exc:
        logger.warning(f"forecast failed for {code}: {exc}")
        raise HTTPException(status_code=422, detail=f"行情数据获取失败：{exc}")
    return _clean(payload)


def _explain_prompt(p: Dict[str, Any]) -> str:
    """把确定性研判结果压成一段事实清单，避免 LLM 自行编造数字。"""
    t = p.get("technical") or {}
    snap = t.get("snapshot") or {}
    ma = snap.get("ma") or {}
    prob_lines = " / ".join(
        f"{x['horizon']} 日 {x['probability']}%（样本 {x['samples']}，{x['confidence']}）"
        for x in p.get("probabilities") or []
    )
    mtf = p.get("multi_timeframe") or {}
    tf = mtf.get("timeframes") or {}
    tf_line = "｜".join(
        f"{k} {v.get('direction')}({v.get('score')})" for k, v in tf.items()
    )
    lv = p.get("levels") or {}
    res = " / ".join(f"{r['label']} {r['price']}" for r in lv.get("resistances") or []) or "无"
    sup = " / ".join(f"{r['label']} {r['price']}" for r in lv.get("supports") or []) or "无"
    scen = "；".join(
        f"{r['label']} {r['probability']}% → {r['target']}（{r['return_pct']}%）"
        for r in p.get("scenarios") or []
    )
    sig = "\n".join(f"- {x['group']}：{x['text']}" for x in p.get("signals") or [])
    mkt = p.get("market") or {}
    ind_info = p.get("industry") or {}
    sec = p.get("sector_relative") or {}
    rel = p.get("relative") or {}
    return f"""你是 A 股技术面分析师。以下研判由系统按确定性规则计算，**所有数字必须原样引用，禁止改动、禁止另算目标价或收益**。

## 标的
{p.get('symbol')} · 截至 {p.get('as_of')} · 收盘 {p.get('price')}（{p.get('change_pct')}%）

## 技术评分
综合 {t.get('score')}（{t.get('direction')}）｜分项 {t.get('parts')}
关键指标：MA20 {ma.get('ma20')}、MA60 {ma.get('ma60')}、RSI14 {snap.get('rsi14')}、MACD 柱 {(snap.get('macd') or {}).get('hist')}、ATR14 {snap.get('atr')}（{snap.get('atr_pct')}%）、量比 {snap.get('volume_ratio')}

## 概率
{prob_lines}
置信方法：{(p.get('probability') or {}).get('method')}
市场环境：{mkt.get('state')}｜{mkt.get('note')}

## 多周期
{tf_line} → {mtf.get('resonance')}（{mtf.get('summary')}）

## 关键价位
阻力 {res}｜支撑 {sup}｜建议止损 {lv.get('suggested_stop')}

## 情景
{scen}

## 信号
{sig}

## 行业与相对强度
行业 {ind_info.get('name')}（{ind_info.get('code')}）{'（近似映射）' if ind_info.get('approximate') else ''}
相对沪深300：20 日 {rel.get('excess20')}%、60 日 {rel.get('excess60')}%；相对本行业：20 日 {sec.get('excess20')}%

## 输出要求（严格 JSON，不要输出任何其他文字）
{{"summary": "3-5 句话，解释为什么得到这个结论，只引用上面的数字",
 "invalidation": ["1 个可验证的失效条件（含具体价位或指标阈值）", "另 1 个失效条件"],
 "watch": ["未来 1-2 周值得跟踪的数据点", "..."],
 "risk_note": "一句风险提示，不构成投资建议"}}"""


def _rule_explain(p: Dict[str, Any]) -> Dict[str, Any]:
    """模型不可用时的兜底叙述：只用已算好的数字拼句，不产生新信息。"""
    t = p.get("technical") or {}
    snap = t.get("snapshot") or {}
    ma = snap.get("ma") or {}
    parts = t.get("parts") or {}
    prob = p.get("probability") or {}
    probs = {x["horizon"]: x for x in p.get("probabilities") or []}
    lv = p.get("levels") or {}
    scen = {r["key"]: r for r in p.get("scenarios") or []}
    mtf = p.get("multi_timeframe") or {}
    mkt = p.get("market") or {}
    ind_info = p.get("industry") or {}
    sec = p.get("sector_relative") or {}
    rel = p.get("relative") or {}
    rets = snap.get("returns") or {}

    comps = t.get("components") or {}
    weakest = sorted(
        ((k, v.get("score") if isinstance(v, dict) else v) for k, v in comps.items()),
        key=lambda kv: (kv[1] is None, kv[1] or 0),
    )[:2]
    cn = {"trend": "趋势", "momentum": "动量", "volume": "量能",
          "position": "位置", "relative": "相对强度"}
    weakest_txt = "、".join(f"{cn.get(k, k)}（{v}）" for k, v in weakest if v is not None) or "各项"
    def _r(v, nd=2):
        try:
            return f"{float(v):.{nd}f}"
        except (TypeError, ValueError):
            return "-"

    lines = [
        f"技术评分 {_r(t.get('score'), 1)}（{t.get('direction')}），分项中最弱的是 {weakest_txt}，"
        f"20 日区间涨跌 {_r(rets.get('r20'))}%，250 日位置 {_r(snap.get('pos_250'), 0)}%。",
    ]
    h = p.get("horizon") or 10
    hp = probs.get(h) or {}
    conf_cn = {"high": "高", "medium": "中", "low": "低"}.get(hp.get("confidence"), "中")
    lines.append(
        f"历史上出现同类状态后，{h} 个交易日上涨概率 {_r(hp.get('probability'), 1)}%"
        f"（{hp.get('samples')} 个样本，{conf_cn}置信），"
        f"与打分的{('偏多' if (t.get('score') or 0) > 0 else '偏空')}方向"
        f"{'一致' if ((hp.get('probability') or 50) - 50) * (t.get('score') or 0) >= 0 else '出现背离'}。"
    )
    ma20 = ma.get("ma20")
    if ma20 and p.get("price"):
        dist = (p["price"] / ma20 - 1) * 100
        lines.append(
            f"价格位于 MA20（{_r(ma20)}）{'上方' if dist > 0 else '下方'} {abs(dist):.2f}%，"
            f"多周期为{mtf.get('resonance') or '数据不足'}，"
            f"大盘处于{mkt.get('state') or '未知'}状态。"
        )
    bear = scen.get("bear")
    if bear:
        lines.append(
            f"悲观情景（{bear.get('probability')}%）对应 {_r(bear.get('target'))}"
            f"（{_r(bear.get('return_pct'))}%），触发条件为{bear.get('trigger')}。"
        )
    pos = snap.get("pos_250")
    if pos is not None:
        if pos >= 85:
            lines.append(f"价格处于 250 日区间 {pos:.0f}% 的高位，追高的回撤风险高于均值。")
        elif pos <= 15:
            lines.append(f"价格处于 250 日区间 {pos:.0f}% 的低位，反弹弹性大但趋势尚未确认。")
    if sec.get("excess20") is not None or rel.get("excess20") is not None:
        lines.append(
            f"相对沪深300 20 日超额 {_r(rel.get('excess20'))}%，"
            f"相对{ind_info.get('name') or '所属行业'} 20 日超额 {_r(sec.get('excess20'))}%。"
        )

    supports = lv.get("supports") or []
    resistances = lv.get("resistances") or []
    bearish = (t.get("score") or 0) < 0
    invalidation = []
    watch = []
    if bearish:
        if resistances:
            invalidation.append(
                f"收盘站上 {resistances[0]['label']} {_r(resistances[0]['price'])}，弱势判断失效"
            )
        if supports:
            watch.append(
                f"收盘跌破 {supports[0]['label']} {_r(supports[0]['price'])} 且量比大于 1.2，弱势延续"
            )
    else:
        if supports:
            invalidation.append(
                f"收盘跌破 {supports[0]['label']} {_r(supports[0]['price'])}，强势判断失效"
            )
        if resistances:
            watch.append(
                f"收盘站上 {resistances[0]['label']} {_r(resistances[0]['price'])} 且量比大于 1.2，强势延续"
            )
    watch += [
        f"MA20（{_r(ma20)}）的得失",
        f"量比能否回到 1 附近（当前 {_r(snap.get('volume_ratio'))}）",
        "沪深300 与 MA20 的相对位置",
    ]
    return {
        "summary": "".join(lines),
        "invalidation": invalidation,
        "watch": watch,
        "risk_note": "以上为规则化技术解读，不含基本面与消息面，不构成投资建议。",
    }


@router.post("/explain")
async def explain_forecast(payload: Dict[str, Any]):
    """把确定性研判结果交给 LLM 做叙述，数字仍以上面的计算结果为准。"""
    symbol = str(payload.get("symbol") or "").strip()
    if not symbol:
        raise HTTPException(status_code=400, detail="缺少 symbol")
    horizon = int(payload.get("horizon") or 10)
    horizon = max(3, min(30, horizon))
    days = max(120, min(1500, int(payload.get("days") or 700)))
    code = await _resolve_code(symbol)

    result = await asyncio.wait_for(
        _compute_stock_payload(code, horizon, days), timeout=40.0,
    )

    from app.model_gateway import get_model_gateway
    from app.utils.json_extract import parse_llm_json

    try:
        gw = get_model_gateway()
        resp = await gw.chat(
            messages=[{"role": "user", "content": _explain_prompt(result)}],
            max_tokens=900,
            temperature=0.2,
        )
    except Exception as exc:
        logger.warning(f"forecast explain failed: {exc}")
        return {"available": False, "reason": f"模型调用失败：{exc}"}

    if not getattr(resp, "success", False) or not (resp.content or "").strip():
        fallback = _rule_explain(result)
        return {
            "available": True,
            "model": "规则引擎",
            "fallback": True,
            "reason": getattr(resp, "error", "") or "模型未返回内容",
            **fallback,
        }

    parsed = parse_llm_json(resp.content or "")
    if isinstance(parsed, dict) and parsed.get("summary"):
        return {
            "available": True,
            "model": getattr(resp, "model", ""),
            "fallback": False,
            "summary": str(parsed.get("summary") or "").strip(),
            "invalidation": [str(x) for x in (parsed.get("invalidation") or [])][:3],
            "watch": [str(x) for x in (parsed.get("watch") or [])][:4],
            "risk_note": str(parsed.get("risk_note") or "").strip(),
        }
    return {
        "available": True,
        "model": "规则引擎",
        "fallback": True,
        "reason": "模型输出无法解析",
        **_rule_explain(result),
    }


_LEGU_CACHE: Dict[str, Any] = {"ts": 0.0, "data": None}
_LEGU_TTL = 120.0


def _parse_legu(df: pd.DataFrame) -> Dict[str, Any]:
    """把「赚钱效应」表解析成结构化广度指标（纯函数，便于测试）。"""
    if df is None or df.empty or "item" not in df or "value" not in df:
        return {"available": False}
    raw = {str(r["item"]).strip(): r["value"] for _, r in df.iterrows()}

    def _int(key: str) -> Optional[int]:
        try:
            return int(float(raw.get(key)))
        except (TypeError, ValueError):
            return None

    def _pct(key: str) -> Optional[float]:
        try:
            return round(float(str(raw.get(key)).replace("%", "")), 2)
        except (TypeError, ValueError):
            return None

    up, down = _int("上涨"), _int("下跌")
    flat = _int("平盘")
    total = sum(v for v in (up, down, flat) if v) or None
    return {
        "available": up is not None and down is not None,
        "up": up, "down": down, "flat": flat,
        "limit_up": _int("涨停"), "limit_down": _int("跌停"),
        "real_limit_up": _int("真实涨停"), "real_limit_down": _int("真实跌停"),
        "activity": _pct("活跃度"),
        "up_ratio": round(up / total * 100, 1) if total and up is not None else None,
        "as_of": str(raw.get("统计日期") or ""),
    }


async def market_breadth() -> Dict[str, Any]:
    """全市场涨跌家数（仅供参考的情绪温度，失败时静默降级）。"""
    now = time.time()
    if _LEGU_CACHE["data"] is not None and now - _LEGU_CACHE["ts"] < _LEGU_TTL:
        return _LEGU_CACHE["data"]

    def _load():
        import akshare as ak
        return ak.stock_market_activity_legu()

    try:
        df = await asyncio.to_thread(_load)
        data = _parse_legu(df)
        _LEGU_CACHE.update({"ts": now, "data": data})
        return data
    except Exception as exc:
        logger.warning(f"market breadth failed: {exc}")
        return _LEGU_CACHE["data"] or {"available": False}


@router.get("/market")
async def market_overview():
    """市场环境：沪深300 趋势状态 + 主要指数当日表现。"""
    bench = await bench_history()
    regime = market_regime(bench)
    indices = []
    try:
        from app.api.data import get_market_indices
        rows = await get_market_indices()
        indices = [
            {"name": r.name, "price": r.price, "change_pct": r.change_pct}
            for r in rows
        ]
    except Exception as exc:
        logger.debug(f"index quotes failed: {exc}")
    breadth = await market_breadth()
    return {
        "regime": regime,
        "breadth": breadth,
        "indices": indices,
        "series": [
            {"date": r["date"], "close": round(float(r["close"]), 2)}
            for _, r in bench.tail(120).iterrows()
        ] if not bench.empty else [],
    }


@router.get("/industries")
async def industry_overview(force: bool = Query(False)):
    """申万 31 个一级行业：涨跌、估值、动量与轮动象限。"""
    data = await ind.get_industry_list(force=force)
    bench = await bench_history()
    br = _bench_returns(bench)
    for row in data.get("rows") or []:
        for n in (20, 60):
            rv, bv = row.get(f"r{n}"), br.get(f"r{n}")
            row[f"rs{n}"] = round(rv - bv, 2) if rv is not None and bv is not None else None
    data["bench"] = {"name": "沪深300", **br}
    return _clean(data)


@router.get("/industry")
async def industry_detail(
    code: str = Query(..., description="申万一级行业代码，如 801080（电子）"),
):
    """单个行业详情：指数走势、相对强弱、估值、广度与成分股。"""
    code = code.split(".")[0]
    if not code.isdigit():
        raise HTTPException(status_code=400, detail="行业代码须为 6 位数字")
    bench = await bench_history()
    detail = await ind.industry_detail(code, bench=bench)
    if not detail.get("series"):
        raise HTTPException(status_code=422, detail="该行业暂无历史数据（可能仍在拉取）")
    detail["bench_name"] = "沪深300"
    return _clean(detail)
