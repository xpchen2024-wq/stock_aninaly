# ============================================================================
# AI Stock Analysis Platform - Deterministic Risk Metrics
# ============================================================================
"""Pure-math risk computations.

Design principle (aligned with ai-hedge-fund): the LLM never invents trade
numbers. VaR/CVaR and position sizing are computed here from real K-line
history; the LLM only narrates.
"""
from __future__ import annotations

import logging
import math
from typing import Dict, Union

import numpy as np

logger = logging.getLogger(__name__)


def simple_returns(closes: np.ndarray) -> np.ndarray:
    closes = np.asarray(closes, dtype=float)
    if closes.size < 2:
        return np.array([])
    return (closes[1:] - closes[:-1]) / closes[:-1]


def historical_var_cvar(
    closes: np.ndarray,
    position_value: float,
    confidence_level: float = 0.95,
    horizon_days: int = 1,
) -> Dict[str, Union[float, int, bool]]:
    """Historical-simulation VaR / CVaR for a position, as positive loss amounts.

    - Returns are drawn from the empirical distribution of `closes`.
    - horizon_days > 1 applies sqrt(T) scaling (i.i.d. approximation).
    - Insufficient history (<30 samples) yields zeros plus an
      `insufficient_history` marker instead of a misleading number.
    """
    returns = simple_returns(np.asarray(closes, dtype=float))
    if returns.size < 30:
        logger.warning(
            f"historical_var_cvar: only {returns.size} return samples (<30), "
            "returning zeros with insufficient_history=True"
        )
        return {
            "var_value": 0.0,
            "cvar_value": 0.0,
            "confidence_level": confidence_level,
            "sample_size": int(returns.size),
            "insufficient_history": True,
        }

    alpha = 1.0 - confidence_level
    scale = math.sqrt(max(horizon_days, 1))
    var_return = -float(np.quantile(returns, alpha)) * scale
    tail = returns[returns <= np.quantile(returns, alpha)]
    cvar_return = -float(tail.mean()) * scale if tail.size else var_return

    # Clamp: losses cannot exceed the position itself under these assumptions
    var_value = max(min(var_return, 1.0), 0.0) * float(position_value)
    cvar_value = max(min(cvar_return, 1.0), var_value / max(float(position_value), 1e-9)) * float(position_value)
    if position_value <= 0:
        var_value = cvar_value = 0.0

    return {
        "var_value": round(var_value, 2),
        "cvar_value": round(cvar_value, 2),
        "confidence_level": confidence_level,
        "sample_size": int(returns.size),
        "insufficient_history": False,
    }


def position_size(
    confidence: float,
    price: float,
    stop_loss: float,
    capital: float = 1_000_000.0,
    risk_per_trade: float = 0.02,
) -> int:
    """Risk-based share count.

    risk_amount = capital * risk_per_trade scaled by LLM confidence;
    per-share risk = max(price - stop_loss, 1% of price) so that a missing or
    degenerate stop cannot produce unbounded size.
    """
    if price <= 0:
        return 0
    clamped_confidence = min(max(float(confidence), 0.1), 1.0)
    risk_amount = float(capital) * risk_per_trade * clamped_confidence
    per_share_risk = max(price - float(stop_loss), price * 0.01)
    return max(int(risk_amount // per_share_risk), 0)
