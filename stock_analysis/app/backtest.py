# ============================================================================
# AI Stock Analysis Platform - Strategy Backtest Engine
# ============================================================================
"""Deterministic backtesting over real K-lines.

Strategies are pure signal functions over price series; execution applies
position limits, fees, slippage and stop-loss/take-profit rules. The LLM is
never involved in trade simulation — results are reproducible.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

SUPPORTED_STRATEGIES = ("ma_cross", "macd_cross", "trend_follow")

# Minimum bars each strategy needs before signals are meaningful
STRATEGY_WARMUP = {
    "ma_cross": 25,       # longest lookback: MA20 + cross confirmation
    "macd_cross": 40,     # EMA26 + DEA9 convergence
    "trend_follow": 65,   # MA60 lookback
}


def _warmup_for(strategy: str) -> int:
    return STRATEGY_WARMUP.get(strategy, 65)


@dataclass
class BacktestParams:
    symbol: str = ""
    name: str = ""
    strategy: str = "ma_cross"
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    initial_capital: float = 1_000_000.0
    position_limit_pct: float = 20.0      # max % of equity per trade
    fee_rate: float = 0.0003              # per-side fee
    slippage_ticks: int = 1               # ticks paid per side
    tick_size: float = 0.01
    stop_loss_pct: float = 2.0            # 0 disables
    take_profit_pct: float = 15.0         # 0 disables


def _ma(values: np.ndarray, period: int) -> np.ndarray:
    out = np.full(len(values), np.nan)
    if len(values) >= period:
        csum = np.cumsum(np.insert(values, 0, 0.0))
        out[period - 1:] = (csum[period:] - csum[:-period]) / period
    return out


def _ema(values: np.ndarray, period: int) -> np.ndarray:
    out = np.full(len(values), np.nan)
    if len(values):
        k = 2.0 / (period + 1)
        out[0] = values[0]
        for i in range(1, len(values)):
            out[i] = values[i] * k + out[i - 1] * (1 - k)
    return out


def _macd(close: np.ndarray):
    ema12, ema26 = _ema(close, 12), _ema(close, 26)
    dif = ema12 - ema26
    dea = _ema(np.nan_to_num(dif, nan=0.0), 9)
    return dif, dea


def _signals(strategy: str, close: np.ndarray) -> np.ndarray:
    """Return array of {-1, 0, +1} target positions per bar (before warmup → 0)."""
    n = len(close)
    sig = np.zeros(n)
    warmup = _warmup_for(strategy)

    if strategy == "ma_cross":
        fast, slow = _ma(close, 5), _ma(close, 20)
        for i in range(warmup, n):
            f0, s0 = fast[i], slow[i]
            if np.isnan(f0) or np.isnan(s0) or np.isnan(fast[i - 1]) or np.isnan(slow[i - 1]):
                continue
            if fast[i - 1] <= slow[i - 1] and f0 > s0:
                sig[i] = 1
            elif fast[i - 1] >= slow[i - 1] and f0 < s0:
                sig[i] = -1

    elif strategy == "macd_cross":
        dif, dea = _macd(close)
        for i in range(warmup, n):
            if any(np.isnan(v) for v in (dif[i], dea[i], dif[i - 1], dea[i - 1])):
                continue
            if dif[i - 1] <= dea[i - 1] and dif[i] > dea[i]:
                sig[i] = 1
            elif dif[i - 1] >= dea[i - 1] and dif[i] < dea[i]:
                sig[i] = -1

    elif strategy == "trend_follow":
        ma20, ma60 = _ma(close, 20), _ma(close, 60)
        for i in range(warmup, n):
            if np.isnan(ma60[i]) or np.isnan(ma20[i]):
                continue
            sig[i] = 1 if ma20[i] > ma60[i] else -1

    else:
        raise ValueError(f"Unsupported strategy: {strategy}")
    return sig


def run_backtest(klines: list, params: BacktestParams) -> Dict[str, Any]:
    """Simulate a strategy over klines; returns metrics, trades and equity curve."""
    if params.strategy not in SUPPORTED_STRATEGIES:
        raise ValueError(f"Unsupported strategy: {params.strategy}. "
                         f"Supported: {SUPPORTED_STRATEGIES}")

    # Date window filter
    raw_count = len(list(klines))
    bars = list(klines)
    if params.start_date:
        bars = [k for k in bars if str(k.date)[:10] >= params.start_date]
    if params.end_date:
        bars = [k for k in bars if str(k.date)[:10] <= params.end_date]
    min_bars = _warmup_for(params.strategy) + 5
    if not bars:
        raise ValueError(
            f"所选区间 {params.start_date or '起点'} ~ {params.end_date or '终点'} "
            f"内没有K线数据（数据源实际覆盖 {raw_count} 根，"
            f"请调整回测区间）")
    if len(bars) < min_bars:
        raise ValueError(
            f"Insufficient K-line history: {len(bars)} bars "
            f"(need ≥ {min_bars} in the selected range)")

    dates = [str(k.date)[:10] for k in bars]
    open_ = np.array([float(k.open) for k in bars])
    close = np.array([float(k.close) for k in bars])
    high = np.array([float(k.high) for k in bars])
    low = np.array([float(k.low) for k in bars])

    sig = _signals(params.strategy, close)

    slip = params.slippage_ticks * params.tick_size
    fee = params.fee_rate

    cash = float(params.initial_capital)
    shares = 0
    entry_price = 0.0
    equity_curve: List[float] = []
    trades: List[Dict[str, Any]] = []

    def buy(i: int, price: float):
        nonlocal cash, shares, entry_price
        budget = min(cash, cash * params.position_limit_pct / 100.0 * 5.0)
        per_share_cost = price * (1 + fee) + slip
        qty = int(budget // per_share_cost)
        if qty <= 0:
            return
        cost = qty * per_share_cost
        if cost > cash:
            return
        cash -= cost
        shares, entry_price = qty, price
        trades.append({"date": dates[i], "side": "buy", "price": round(price, 3),
                       "quantity": qty})

    def sell(i: int, price: float, reason: str):
        nonlocal cash, shares, entry_price
        if shares <= 0:
            return
        proceeds = shares * price * (1 - fee) - shares * slip
        pnl_pct = (price / entry_price - 1.0) * 100 if entry_price else 0.0
        cash += proceeds
        trades.append({"date": dates[i], "side": "sell", "price": round(price, 3),
                       "quantity": shares, "pnl_pct": round(pnl_pct, 3),
                       "reason": reason})
        shares, entry_price = 0, 0.0

    for i in range(len(bars)):
        o, h, l, c = open_[i], high[i], low[i], close[i]

        # Manage open position first: stop-loss / take-profit intraday bounds
        if shares > 0:
            sl_price = entry_price * (1 - params.stop_loss_pct / 100.0)
            tp_price = entry_price * (1 + params.take_profit_pct / 100.0)
            if l <= sl_price:
                sell(i, min(o, sl_price), "stop_loss")
            elif params.take_profit_pct > 0 and h >= tp_price:
                sell(i, max(o, tp_price), "take_profit")

        act = sig[i]
        if act == 1 and shares == 0:
            buy(i, c + slip)
        elif act == -1 and shares > 0:
            sell(i, c - slip, "signal_exit")

        equity_curve.append(cash + shares * c)

    # Force-close at the end so PnL is realized
    if shares > 0:
        sell(len(bars) - 1, close[-1], "end_of_backtest")
        equity_curve[-1] = cash

    final_equity = equity_curve[-1] if equity_curve else float(params.initial_capital)
    total_return_pct = (final_equity / float(params.initial_capital) - 1) * 100
    years = max(len(bars) / 244.0, 1e-9)
    annualized = ((final_equity / float(params.initial_capital)) ** (1 / years) - 1) * 100

    eq = np.array(equity_curve, dtype=float)
    running_max = np.maximum.accumulate(eq)
    drawdowns = (eq - running_max) / running_max
    max_drawdown_pct = float(drawdowns.min() * 100) if len(eq) else 0.0

    daily_ret = np.diff(eq) / eq[:-1] if len(eq) > 1 else np.array([0.0])
    std = float(np.std(daily_ret))
    sharpe = float(np.mean(daily_ret) / std * math.sqrt(244)) if std > 0 else 0.0

    sells = [t for t in trades if t["side"] == "sell" and t.get("reason") != "end_of_backtest"]
    wins = sum(1 for t in sells if t.get("pnl_pct", 0) > 0)
    win_rate = (wins / len(sells) * 100) if sells else 0.0

    peak_equity = max(float(params.initial_capital), float(eq.max()) if len(eq) else params.initial_capital)

    # 月度收益（用于热力图）与年度统计
    equity_by_date = {d: float(e) for d, e in zip(dates, equity_curve)}
    monthly: Dict[str, float] = {}
    yearly: Dict[str, float] = {}
    for year in sorted({d[:4] for d in dates}):
        year_dates = [d for d in dates if d.startswith(year)]
        if not year_dates:
            continue
        start_eq = equity_by_date[year_dates[0]]
        end_eq = equity_by_date[year_dates[-1]]
        prev = start_eq
        for d in year_dates:
            month = d[:7]
            m_end = equity_by_date[d]
            if month not in monthly:
                monthly[month] = 0.0
            monthly[month] = (m_end / prev - 1) * 100
            prev = m_end
        yearly[year] = (end_eq / start_eq - 1) * 100

    return {
        "symbol": params.symbol,
        "name": params.name,
        "strategy": params.strategy,
        "start_date": dates[0],
        "end_date": dates[-1],
        "total_bars": len(bars),
        "metrics": {
            "initial_capital": float(params.initial_capital),
            "final_equity": round(final_equity, 2),
            "total_return_pct": round(total_return_pct, 3),
            "annualized_return_pct": round(annualized, 3),
            "max_drawdown_pct": round(max_drawdown_pct, 3),
            "sharpe_ratio": round(sharpe, 3),
            "win_rate_pct": round(win_rate, 2),
            "trade_count": len(sells),
        },
        "trades": trades,
        "equity_curve": [
            {"date": d, "equity": round(e, 2)}
            for d, e in zip(dates, equity_curve)
        ],
        "monthly_returns_pct": {k: round(v, 3) for k, v in sorted(monthly.items())},
        "yearly_returns_pct": {k: round(v, 3) for k, v in sorted(yearly.items())},
    }
