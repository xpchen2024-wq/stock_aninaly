# ============================================================================
# Walk-forward signal accuracy for the forecast engine
# ============================================================================
"""Measure how often the研判引擎's direction call was right.

The engine scores a stock from the data available *up to* a given date. Here we
replay that over history: at every rebalance date we truncate the K-line
series, recompute the score with only past data (no look-ahead), then compare
the predicted direction with the realized forward return.

    score > 0  -> predicts up
    score < 0  -> predicts down

Outputs
-------
- hit_rate_pct      : overall hit rate of the direction call
- by_bucket         : hit rate grouped by score magnitude (stronger signal →
                     higher hit rate is the expected shape)
- avg_forward_bull  : mean forward return after bullish signals
- avg_forward_bear  : mean forward return after bearish signals
- excess_pct        : mean forward return minus the benchmark's forward return
- timeline          : per-rebalance detail (date, score, forward return, hit)
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from app.forecast import build_stock_forecast

logger = logging.getLogger(__name__)

# 需要多少根K线才让多周期/相对强度指标有意义
MIN_BARS = 160
# 评分分档（绝对值）
BUCKETS = ((0, 15, "弱(0~15)"), (15, 30, "中(15~30)"),
           (30, 50, "强(30~50)"), (50, 999, "极强(50+)"))


def _bucket_label(score: float) -> str:
    a = abs(score)
    for lo, hi, label in BUCKETS:
        if lo <= a < hi:
            return label
    return BUCKETS[-1][2]


def _align_bench(bench: Optional[pd.DataFrame], dates) -> pd.DataFrame:
    """Benchmark sliced to the same window as the stock slice (no look-ahead)."""
    if bench is None or len(bench) == 0:
        return pd.DataFrame()
    b = bench.copy()
    b["date"] = b["date"].astype(str)
    end = str(dates[-1])
    b = b[b["date"] <= end]
    return b.tail(400).reset_index(drop=True)


def evaluate_signal_accuracy(
    df: pd.DataFrame,
    bench: Optional[pd.DataFrame] = None,
    horizon: int = 10,
    step: int = 5,
    lookback: int = 250,
    min_bars: int = MIN_BARS,
) -> Dict[str, Any]:
    """Walk-forward evaluation of the engine's direction calls.

    Parameters
    ----------
    df     : stock K-lines (ascending), must contain date/close/high/low/volume
    bench  : benchmark K-lines (沪深300) for excess-return comparison
    horizon: forward trading bars used to check the call
    step   : rebalance spacing in bars
    """
    if df is None or len(df) < min_bars + horizon + 1:
        return {
            "available": False,
            "reason": f"历史样本不足（需要至少 {min_bars + horizon + 1} 根K线）",
            "samples": 0,
        }

    df = df.reset_index(drop=True)
    dates = df["date"].astype(str).tolist()
    close = df["close"].astype(float).tolist()

    # 只回测最近 lookback 根，避免样本过多拖慢计算
    start_idx = max(min_bars, len(df) - horizon - lookback)
    records: List[Dict[str, Any]] = []

    for i in range(start_idx, len(df) - horizon, max(1, step)):
        slice_df = df.iloc[:i + 1].reset_index(drop=True)
        bench_slice = _align_bench(bench, slice_df["date"].astype(str).tolist())
        if bench_slice is None or len(bench_slice) == 0:
            bench_slice = None
        try:
            payload = build_stock_forecast(slice_df, bench_slice, horizon=horizon)
        except Exception as exc:  # 单次失败不影响整体
            logger.debug("walk-forward slice failed at %s: %s", dates[i], exc)
            continue
        if not payload.get("available"):
            continue

        score = float((payload.get("technical") or {}).get("score") or 0)
        if score == 0:
            continue
        fwd = (close[i + horizon] / close[i] - 1.0) * 100
        bench_fwd = None
        if bench_slice is not None and not bench_slice.empty:
            b_dates = bench_slice["date"].astype(str).tolist()
            b_close = bench_slice["close"].astype(float).tolist()
            try:
                b0 = b_close[b_dates.index(dates[i])] if dates[i] in b_dates else b_close[0]
                b1 = b_close[-1]
                bench_fwd = (b1 / b0 - 1.0) * 100
            except (ValueError, IndexError):
                bench_fwd = None

        pred_up = score > 0
        records.append({
            "date": dates[i],
            "score": round(score, 1),
            "bucket": _bucket_label(score),
            "direction": "bullish" if pred_up else "bearish",
            "forward_return_pct": round(fwd, 2),
            "benchmark_return_pct": round(bench_fwd, 2) if bench_fwd is not None else None,
            "hit": bool((pred_up and fwd > 0) or ((not pred_up) and fwd <= 0)),
        })

    if not records:
        return {"available": False, "reason": "没有可评估的调仓点", "samples": 0}

    hits = sum(1 for r in records if r["hit"])
    n = len(records)
    bull = [r for r in records if r["direction"] == "bullish"]
    bear = [r for r in records if r["direction"] == "bearish"]
    fwd_all = [r["forward_return_pct"] for r in records]
    bench_all = [r["benchmark_return_pct"] for r in records
                 if r["benchmark_return_pct"] is not None]
    excess = [r["forward_return_pct"] - r["benchmark_return_pct"]
              for r in records if r["benchmark_return_pct"] is not None]

    by_bucket: Dict[str, Dict[str, Any]] = {}
    for lo, hi, label in BUCKETS:
        sub = [r for r in records if r["bucket"] == label]
        if not sub:
            continue
        h = sum(1 for r in sub if r["hit"])
        by_bucket[label] = {
            "samples": len(sub),
            "hits": h,
            "hit_rate_pct": round(h / len(sub) * 100, 1),
            "avg_forward_pct": round(float(np.mean([r["forward_return_pct"] for r in sub])), 2),
        }

    return {
        "available": True,
        "horizon": horizon,
        "step": step,
        "samples": n,
        "hits": hits,
        "misses": n - hits,
        "hit_rate_pct": round(hits / n * 100, 1),
        "bull_samples": len(bull),
        "bear_samples": len(bear),
        "bull_hit_rate_pct": round(sum(1 for r in bull if r["hit"]) / len(bull) * 100, 1) if bull else None,
        "bear_hit_rate_pct": round(sum(1 for r in bear if r["hit"]) / len(bear) * 100, 1) if bear else None,
        "avg_forward_pct": round(float(np.mean(fwd_all)), 2),
        "avg_forward_bull_pct": round(float(np.mean([r["forward_return_pct"] for r in bull])), 2) if bull else None,
        "avg_forward_bear_pct": round(float(np.mean([r["forward_return_pct"] for r in bear])), 2) if bear else None,
        "avg_benchmark_pct": round(float(np.mean(bench_all)), 2) if bench_all else None,
        "avg_excess_pct": round(float(np.mean(excess)), 2) if excess else None,
        "by_bucket": by_bucket,
        "timeline": records[-60:],
        "period": {"start": records[0]["date"], "end": records[-1]["date"]},
    }
