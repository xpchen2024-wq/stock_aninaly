# ============================================================================
# AI Stock Analysis Platform - Technical Indicator Library
# ============================================================================
"""纯函数技术指标库（无网络、无第三方 TA 依赖）。

约定：所有函数接收按时间升序的 pandas.Series / numpy 数组，返回等长序列，
不足窗口的位置为 NaN。计算口径与国内行情软件保持一致：

- MACD : EMA12 - EMA26，DEA = EMA(DIF, 9)，柱体 = 2 × (DIF - DEA)
- RSI  : Wilder 平滑（同通达信 SMA(X, N, 1)）
- KDJ  : RSV → K/D 用 1/3 平滑，J = 3K - 2D
- BOLL : 20 日中轨 ± 2 倍标准差
- ATR  : Wilder 真实波幅
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd


def _s(values) -> pd.Series:
    if isinstance(values, pd.Series):
        return values.astype(float)
    return pd.Series(np.asarray(values, dtype=float))


def sma(values, window: int) -> pd.Series:
    return _s(values).rolling(window, min_periods=window).mean()


def ema(values, span: int) -> pd.Series:
    return _s(values).ewm(span=span, adjust=False).mean()


def wilder(values, window: int) -> pd.Series:
    """Wilder 平滑（等价于通达信 SMA(v, N, 1)）。"""
    return _s(values).ewm(alpha=1.0 / window, adjust=False).mean()


def macd(close, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict[str, pd.Series]:
    c = _s(close)
    dif = c.ewm(span=fast, adjust=False).mean() - c.ewm(span=slow, adjust=False).mean()
    dea = dif.ewm(span=signal, adjust=False).mean()
    hist = (dif - dea) * 2
    return {"dif": dif, "dea": dea, "hist": hist}


def rsi(close, window: int = 14) -> pd.Series:
    c = _s(close)
    delta = c.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1.0 / window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / window, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    out = 100 - 100 / (1 + rs)
    # 单边上涨时平均跌幅为 0，数学极限为 100（而非 NaN）
    out = out.mask((avg_loss == 0) & (avg_gain > 0), 100.0)
    return out.fillna(50.0)


def kdj(high, low, close, window: int = 9) -> Dict[str, pd.Series]:
    h = _s(high).rolling(window, min_periods=1).max()
    l = _s(low).rolling(window, min_periods=1).min()
    c = _s(close)
    rsv = (c - l) / (h - l).replace(0, np.nan) * 100
    rsv = rsv.fillna(50.0)
    k = rsv.ewm(alpha=1.0 / 3, adjust=False).mean()
    d = k.ewm(alpha=1.0 / 3, adjust=False).mean()
    j = 3 * k - 2 * d
    return {"k": k, "d": d, "j": j}


def boll(close, window: int = 20, k: float = 2.0) -> Dict[str, pd.Series]:
    c = _s(close)
    mid = c.rolling(window, min_periods=window).mean()
    std = c.rolling(window, min_periods=window).std(ddof=0)
    up = mid + k * std
    low = mid - k * std
    width = (up - low) / mid.replace(0, np.nan)
    pct_b = (c - low) / (up - low).replace(0, np.nan)
    return {"mid": mid, "up": up, "low": low, "width": width, "pct_b": pct_b}


def true_range(high, low, close) -> pd.Series:
    h, l, c = _s(high), _s(low), _s(close)
    prev = c.shift(1)
    return pd.concat([(h - l), (h - prev).abs(), (l - prev).abs()], axis=1).max(axis=1)


def atr(high, low, close, window: int = 14) -> pd.Series:
    return true_range(high, low, close).ewm(alpha=1.0 / window, adjust=False).mean()


def obv(close, volume) -> pd.Series:
    c, v = _s(close), _s(volume)
    direction = np.sign(c.diff()).fillna(0.0)
    return (direction * v).cumsum()


def adx(high, low, close, window: int = 14) -> Dict[str, pd.Series]:
    h, l, c = _s(high), _s(low), _s(close)
    up_move = h.diff()
    down_move = -l.diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    tr = true_range(h, l, c).ewm(alpha=1.0 / window, adjust=False).mean()
    plus_di = 100 * pd.Series(plus_dm, index=h.index).ewm(alpha=1.0 / window, adjust=False).mean() / tr.replace(0, np.nan)
    minus_di = 100 * pd.Series(minus_dm, index=h.index).ewm(alpha=1.0 / window, adjust=False).mean() / tr.replace(0, np.nan)
    dx = (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan) * 100
    return {
        "adx": dx.ewm(alpha=1.0 / window, adjust=False).mean(),
        "plus_di": plus_di,
        "minus_di": minus_di,
    }


def cci(high, low, close, window: int = 20) -> pd.Series:
    h, l, c = _s(high), _s(low), _s(close)
    tp = (h + l + c) / 3
    ma = tp.rolling(window, min_periods=window).mean()
    md = (tp - ma).abs().rolling(window, min_periods=window).mean()
    return (tp - ma) / (0.015 * md.replace(0, np.nan))


def williams_r(high, low, close, window: int = 14) -> pd.Series:
    h = _s(high).rolling(window, min_periods=window).max()
    l = _s(low).rolling(window, min_periods=window).min()
    c = _s(close)
    return -100 * (h - c) / (h - l).replace(0, np.nan)


def volume_ratio(volume, window: int = 5) -> pd.Series:
    v = _s(volume)
    return v / v.rolling(window, min_periods=1).mean().replace(0, np.nan)


def max_drawdown(closes) -> float:
    """区间最大回撤（正数，百分比）。"""
    c = _s(closes).dropna()
    if c.empty:
        return 0.0
    peak = c.cummax()
    dd = (c / peak - 1.0).min()
    return float(abs(dd) * 100)


def annualized_volatility(closes, periods: int = 252) -> float:
    c = _s(closes).dropna()
    if len(c) < 3:
        return 0.0
    ret = c.pct_change().dropna()
    return float(ret.std(ddof=1) * np.sqrt(periods) * 100)


def slope(values, window: int = 5) -> float:
    """最近 window 个点的线性回归斜率（按均值归一化为百分比/日）。"""
    v = _s(values).dropna().tail(window)
    if len(v) < 2:
        return 0.0
    x = np.arange(len(v), dtype=float)
    b = np.polyfit(x, v.to_numpy(), 1)[0]
    base = float(v.mean())
    if base == 0:
        return 0.0
    return float(b / base * 100)


def resample_ohlc(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """日线 → 周线/月线。df 需含 date/open/high/low/close/volume 列。"""
    if df is None or df.empty:
        return pd.DataFrame()
    work = df.copy()
    work["date"] = pd.to_datetime(work["date"])
    work = work.set_index("date").sort_index()
    agg = {
        "open": "first", "high": "max", "low": "min",
        "close": "last", "volume": "sum",
    }
    agg = {k: v for k, v in agg.items() if k in work.columns}
    out = work.resample(rule).agg(agg)
    return out.dropna(subset=["close"]).reset_index()


def snapshot(df: pd.DataFrame) -> Dict[str, Any]:
    """一次算出全部指标的最新值与关键中间量，供研判引擎使用。"""
    if df is None or len(df) < 5:
        return {}

    close = _s(df["close"])
    high = _s(df["high"])
    low = _s(df["low"])
    vol = _s(df["volume"]) if "volume" in df.columns else pd.Series(np.nan, index=close.index)

    ma = {n: sma(close, n) for n in (5, 10, 20, 60, 120, 250)}
    m = macd(close)
    r6, r14 = rsi(close, 6), rsi(close, 14)
    k = kdj(high, low, close)
    b = boll(close)
    a = atr(high, low, close)
    ox = adx(high, low, close)
    ob = obv(close, vol)
    vr = volume_ratio(vol)

    last = float(close.iloc[-1])
    prev = float(close.iloc[-2]) if len(close) > 1 else last

    def _v(series: Optional[pd.Series]) -> Optional[float]:
        if series is None or len(series) == 0:
            return None
        val = series.iloc[-1]
        return None if pd.isna(val) else float(val)

    def _ret(n: int) -> Optional[float]:
        if len(close) <= n:
            return None
        base = float(close.iloc[-1 - n])
        if base == 0:
            return None
        return round((last / base - 1) * 100, 2)

    atr_v = _v(a)
    high_250 = float(close.tail(250).max()) if len(close) >= 20 else float(close.max())
    low_250 = float(close.tail(250).min()) if len(close) >= 20 else float(close.min())
    span = high_250 - low_250
    pos_250 = (last - low_250) / span if span > 0 else 0.5

    return {
        "price": last,
        "prev_close": prev,
        "change_pct": round((last / prev - 1) * 100, 2) if prev else 0.0,
        "ma": {f"ma{n}": _v(s) for n, s in ma.items()},
        "ma20_slope": slope(ma[20].dropna(), 5) if len(ma[20].dropna()) >= 5 else 0.0,
        "ma60_slope": slope(ma[60].dropna(), 10) if len(ma[60].dropna()) >= 10 else 0.0,
        "macd": {key: _v(s) for key, s in m.items()},
        "rsi6": _v(r6),
        "rsi14": _v(r14),
        "kdj": {key: _v(s) for key, s in k.items()},
        "boll": {key: _v(s) for key, s in b.items()},
        "atr": atr_v,
        "atr_pct": round(atr_v / last * 100, 2) if atr_v and last else None,
        "adx": {key: _v(s) for key, s in ox.items()},
        "obv_slope": slope(ob.dropna(), 20) if len(ob.dropna()) >= 20 else 0.0,
        "volume_ratio": _v(vr),
        "volume_ma5": float(vol.tail(5).mean()) if len(vol.dropna()) >= 5 else None,
        "volume_ma20": float(vol.tail(20).mean()) if len(vol.dropna()) >= 20 else None,
        "returns": {f"r{n}": _ret(n) for n in (5, 20, 60, 120, 250)},
        "high_250": high_250,
        "low_250": low_250,
        "pos_250": round(float(pos_250) * 100, 1),
        "volatility": round(annualized_volatility(close.tail(120)), 2),
        "max_drawdown_120": round(max_drawdown(close.tail(120)), 2),
        "hist_volatility": float(close.pct_change().std(ddof=1)) if len(close) > 3 else 0.0,
        "bars": int(len(close)),
        "high_20": float(high.tail(20).max()),
        "low_20": float(low.tail(20).min()),
        "series": {
            "ma20": [None if pd.isna(x) else round(float(x), 3) for x in ma[20]],
            "ma60": [None if pd.isna(x) else round(float(x), 3) for x in ma[60]],
            "boll_up": [None if pd.isna(x) else round(float(x), 3) for x in b["up"]],
            "boll_low": [None if pd.isna(x) else round(float(x), 3) for x in b["low"]],
            "dif": [None if pd.isna(x) else round(float(x), 4) for x in m["dif"]],
            "dea": [None if pd.isna(x) else round(float(x), 4) for x in m["dea"]],
            "hist": [None if pd.isna(x) else round(float(x), 4) for x in m["hist"]],
            "rsi14": [None if pd.isna(x) else round(float(x), 2) for x in r14],
            "volume_ma5": [None if pd.isna(x) else round(float(x), 2)
                           for x in vol.rolling(5, min_periods=1).mean()],
        },
    }
