# ============================================================================
# Tests - Walk-forward signal accuracy + decision evaluation helpers
# ============================================================================
"""纯函数测试：不触网、不依赖数据库。"""
from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from app.signal_eval import _bucket_label, evaluate_signal_accuracy


def make_df(n: int = 500, drift: float = 0.0008, vol: float = 0.015,
            seed: int = 11, start: float = 100.0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rets = rng.normal(drift, vol, n)
    close = start * np.cumprod(1 + rets)
    high = close * (1 + np.abs(rng.normal(0, 0.005, n)))
    low = close * (1 - np.abs(rng.normal(0, 0.005, n)))
    dates = pd.bdate_range("2023-01-02", periods=n).strftime("%Y-%m-%d")
    return pd.DataFrame({
        "date": dates, "open": close * 0.999, "high": high, "low": low,
        "close": close, "volume": rng.integers(1_000_000, 5_000_000, n).astype(float),
        "amount": close * 1e6,
    })


# -- signal accuracy ----------------------------------------------------------
def test_accuracy_reports_hit_rate_and_buckets():
    df = make_df(600)
    bench = make_df(600, seed=5)
    r = evaluate_signal_accuracy(df, bench, horizon=10, step=5, lookback=250)

    assert r["available"] is True
    assert r["samples"] > 10
    assert 0 <= r["hit_rate_pct"] <= 100
    assert r["hits"] + r["misses"] == r["samples"]
    assert r["by_bucket"], "必须按信号强度分档"
    for label, b in r["by_bucket"].items():
        assert b["samples"] > 0
        assert 0 <= b["hit_rate_pct"] <= 100
        assert b["hits"] <= b["samples"]
    # 看多/看空样本数之和等于总样本
    assert r["bull_samples"] + r["bear_samples"] == r["samples"]


def test_accuracy_is_deterministic():
    df = make_df(500, seed=3)
    a = evaluate_signal_accuracy(df, None, horizon=10, step=5, lookback=200)
    b = evaluate_signal_accuracy(df, None, horizon=10, step=5, lookback=200)
    assert a["samples"] == b["samples"]
    assert a["hit_rate_pct"] == b["hit_rate_pct"]
    assert a["timeline"] == b["timeline"]


def test_accuracy_no_lookahead_uses_past_only():
    """同一调仓点的评分不得因序列末端被截断而改变（无未来数据泄漏）。"""
    df = make_df(500, seed=8)
    full = evaluate_signal_accuracy(df, None, horizon=10, step=10, lookback=250)
    cut = df.iloc[:400].reset_index(drop=True)
    partial = evaluate_signal_accuracy(cut, None, horizon=10, step=10, lookback=250)
    f_map = {r["date"]: r for r in full["timeline"]}
    p_map = {r["date"]: r for r in partial["timeline"]}
    overlap = set(f_map) & set(p_map)
    assert overlap, "两次评估必须有重叠的调仓点"
    for d in overlap:
        assert f_map[d]["score"] == p_map[d]["score"], f"{d} 的评分受未来数据影响"
        assert f_map[d]["direction"] == p_map[d]["direction"]
        assert f_map[d]["forward_return_pct"] == p_map[d]["forward_return_pct"]


def test_accuracy_insufficient_history():
    r = evaluate_signal_accuracy(make_df(50), None)
    assert r["available"] is False
    assert r["samples"] == 0
    assert "不足" in r["reason"]


def test_bucket_labels():
    assert _bucket_label(5) == "弱(0~15)"
    assert _bucket_label(-20) == "中(15~30)"
    assert _bucket_label(40) == "强(30~50)"
    assert _bucket_label(-70) == "极强(50+)"


# -- decision evaluation helpers ---------------------------------------------
def test_as_naive_normalizes_aware_datetimes():
    from app.api.evaluation import _as_naive

    aware = datetime(2026, 10, 21, 8, 0, tzinfo=timezone.utc)
    naive = datetime(2026, 10, 21, 8, 0)
    assert _as_naive(aware) == naive
    assert _as_naive(naive) == naive
    assert _as_naive(aware).tzinfo is None


def test_pick_kline_window_handles_mixed_timezones():
    from app.api.evaluation import _pick_kline_window

    class K:
        def __init__(self, d, p):
            self.date = d
            self.price = p
            self.close = p

    klines = [K(datetime(2026, 1, 5), 10), K(datetime(2026, 1, 6), 11), K(datetime(2026, 1, 7), 12)]
    # aware decision date must not raise
    base, target = _pick_kline_window(klines, datetime(2026, 1, 5, tzinfo=timezone.utc), 1)
    assert base is not None and target is not None
    assert base.price == 10 and target.price == 11
    # not enough forward bars -> None
    assert _pick_kline_window(klines, datetime(2026, 1, 5), 10) == (None, None)
