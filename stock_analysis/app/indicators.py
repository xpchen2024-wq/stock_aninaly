# ============================================================================
# AI Stock Analysis Platform - Technical Indicators (TA-Lib Wrapper)
# ============================================================================
from __future__ import annotations

import logging
from typing import Dict, Any, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def safe_talib(func, *args, **kwargs):
    """Safely call TA-Lib function, falling back to NaN on error."""
    try:
        import talib
        result = func(*args, **kwargs)
        return result
    except ImportError:
        logger.warning("TA-Lib not installed, returning NaN for indicator")
        return np.full(len(args[0]) if args else 0, np.nan)
    except Exception as e:
        logger.warning(f"TA-Lib error: {e}")
        return np.full(len(args[0]) if args else 0, np.nan)


def compute_all_indicators(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    volume: np.ndarray,
    open_: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Compute all technical indicators for AI analysis input.
    Returns a dict with indicator values for the latest data point.
    """
    result = {}

    # --- Trend Indicators ---
    result["ma5"] = _safe_last(safe_talib(talib_ma, close, 5))
    result["ma10"] = _safe_last(safe_talib(talib_ma, close, 10))
    result["ma20"] = _safe_last(safe_talib(talib_ma, close, 20))
    result["ma60"] = _safe_last(safe_talib(talib_ma, close, 60))

    # MACD
    macd, macd_signal, macd_hist = _macd(close)
    result["macd_dif"] = _safe_last(macd)
    result["macd_dea"] = _safe_last(macd_signal)
    result["macd_hist"] = _safe_last(macd_hist)
    result["macd_signal"] = "golden_cross" if (
        result["macd_dif"] is not None and result["macd_dea"] is not None
        and result["macd_dif"] > result["macd_dea"]
    ) else "death_cross"

    # --- Oscillator Indicators ---
    result["rsi_6"] = _safe_last(safe_talib(talib_rsi, close, 6))
    result["rsi_14"] = _safe_last(safe_talib(talib_rsi, close, 14))
    result["rsi_24"] = _safe_last(safe_talib(talib_rsi, close, 24))

    # KDJ
    result["kdj_k"], result["kdj_d"], result["kdj_j"] = _kdj(high, low, close)

    # CCI
    result["cci_14"] = _safe_last(safe_talib(talib_cci, high, low, close, 14))

    # --- Volatility Indicators ---
    bb_upper, bb_middle, bb_lower = _bollinger(close)
    result["bollinger_upper"] = _safe_last(bb_upper)
    result["bollinger_middle"] = _safe_last(bb_middle)
    result["bollinger_lower"] = _safe_last(bb_lower)
    if result["bollinger_lower"] is not None and result["bollinger_upper"] is not None:
        last_close = float(close[-1])
        result["bollinger_position"] = (
            (last_close - result["bollinger_lower"]) /
            (result["bollinger_upper"] - result["bollinger_lower"])
        ) if result["bollinger_upper"] != result["bollinger_lower"] else 0.5

    result["atr_14"] = _safe_last(safe_talib(talib_atr, high, low, close, 14))

    # --- Volume Indicators ---
    result["obv"] = _safe_last(safe_talib(talib_obv, close, volume))
    result["volume_ma5"] = _safe_last(safe_talib(talib_ma, volume, 5))
    result["volume_ratio"] = (
        float(volume[-1]) / result["volume_ma5"]
        if result["volume_ma5"] and result["volume_ma5"] > 0 else 1.0
    )

    # --- Round values ---
    for k, v in result.items():
        if isinstance(v, (float, np.floating)):
            result[k] = round(float(v), 4)

    return result


def _safe_last(arr: np.ndarray) -> Optional[float]:
    """Safely get the last non-NaN value from array."""
    if arr is None or len(arr) == 0:
        return None
    val = arr[-1]
    if isinstance(val, (np.floating, float)) and np.isnan(val):
        # Try second to last
        for i in range(len(arr) - 2, -1, -1):
            if not np.isnan(arr[i]):
                return float(arr[i])
        return None
    return float(val) if val is not None else None


def _macd(close: np.ndarray):
    try:
        import talib
        return talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
    except (ImportError, Exception):
        return np.full(len(close), np.nan), np.full(len(close), np.nan), np.full(len(close), np.nan)


def _kdj(high: np.ndarray, low: np.ndarray, close: np.ndarray):
    try:
        import talib
        k, d = talib.STOCH(high, low, close, fastk_period=9, slowk_period=3,
                           slowk_matype=0, slowd_period=3, slowd_matype=0)
        j = 3 * k - 2 * d
        return _safe_last(k), _safe_last(d), _safe_last(j)
    except (ImportError, Exception):
        return None, None, None


def _bollinger(close: np.ndarray):
    try:
        import talib
        return talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
    except (ImportError, Exception):
        return np.full(len(close), np.nan), np.full(len(close), np.nan), np.full(len(close), np.nan)


# -- Module-level wrappers for safe_talib dispatch ---------------------------
def talib_ma(data: np.ndarray, period: int) -> np.ndarray:
    try:
        import talib
        return talib.MA(data, timeperiod=period, matype=0)
    except (ImportError, Exception):
        return np.full(len(data), np.nan)


def talib_rsi(data: np.ndarray, period: int) -> np.ndarray:
    try:
        import talib
        return talib.RSI(data, timeperiod=period)
    except (ImportError, Exception):
        return np.full(len(data), np.nan)


def talib_cci(high, low, close, period: int) -> np.ndarray:
    try:
        import talib
        return talib.CCI(high, low, close, timeperiod=period)
    except (ImportError, Exception):
        return np.full(len(close), np.nan)


def talib_atr(high, low, close, period: int) -> np.ndarray:
    try:
        import talib
        return talib.ATR(high, low, close, timeperiod=period)
    except (ImportError, Exception):
        return np.full(len(close), np.nan)


def talib_obv(close, volume) -> np.ndarray:
    try:
        import talib
        return talib.OBV(close, volume.astype(float))
    except (ImportError, Exception):
        return np.full(len(close), np.nan)
