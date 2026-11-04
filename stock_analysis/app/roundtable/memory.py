# ============================================================================
# 投研圆桌 · 决策记忆与反思闭环
# ============================================================================
"""对应 TradingAgents 的 memory log：

1. 每次圆桌结束 → 落一条决策记忆（方向/信心/入场价/持有期）；
2. 下一次分析开跑时，后台结算已到期的记忆：取真实收盘价，算绝对收益与
   相对沪深300的超额，再让「反思教练」写一段教训；
3. 组合经理在终审前读取「同标的最近决策 + 其他标的近期教训」。

结算按真实交易日计数（K 线根数），而非自然日，避免节假日把持有期算错。
"""
from __future__ import annotations

import asyncio
import logging
from datetime import date, datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import pandas as pd
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.adapters import DataType
from app.models import RoundtableMemory, now_utc
from app.roundtable.agents import Reflector
from app.roundtable.state import RoundtableConfig
from app.roundtable.toolkit import to_df

logger = logging.getLogger(__name__)

_BENCH_SYMBOL = "sh000300"
_BENCH_NAME = "沪深300"


def _f(v: Any) -> Optional[float]:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if f != f or f in (float("inf"), float("-inf")):
        return None
    return f


async def _load_series(failover, symbol: str, trade_date: date) -> pd.DataFrame:
    try:
        raw = await asyncio.wait_for(
            failover.fetch_with_failover(
                DataType.KLINE, symbol,
                start_date=(trade_date - timedelta(days=30)).strftime("%Y%m%d"),
                end_date=datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%d"),
            ), timeout=20.0
        )
        return to_df(raw)
    except Exception as exc:
        logger.warning(f"[memory] kline load failed {symbol}: {exc}")
        return pd.DataFrame()


async def _load_bench() -> pd.DataFrame:
    def _load():
        import akshare as ak
        return ak.stock_zh_index_daily(symbol=_BENCH_SYMBOL)

    try:
        raw = await asyncio.wait_for(asyncio.to_thread(_load), timeout=12.0)
        return to_df(raw)
    except Exception as exc:
        logger.info(f"[memory] bench unavailable: {exc}")
        return pd.DataFrame()


async def save_decision(db: AsyncSession, *, session_id: str, symbol: str,
                        name: str, trade_date: date, decision: Dict[str, Any],
                        entry_price: Optional[float], horizon_days: int) -> None:
    """把本次终审写入记忆日志（未结算状态）。"""
    action = str(decision.get("action") or "hold").lower()
    target = trade_date + timedelta(days=max(horizon_days, 1) * 2)
    row = RoundtableMemory(
        session_id=session_id,
        symbol=symbol,
        name=name or symbol,
        trade_date=trade_date,
        signal=action,
        confidence=_f(decision.get("confidence")),
        entry_price=_f(entry_price) or _f(decision.get("entry_price")),
        horizon_days=horizon_days,
        target_date=target,
        settled=False,
    )
    db.add(row)
    await db.flush()


async def load_memory_context(db: AsyncSession, symbol: str,
                              same_limit: int = 3,
                              lesson_limit: int = 4,
                              as_of: Optional[datetime] = None) -> str:
    """拼装注入组合经理的历史记忆文本。"""
    blocks: List[str] = []

    same_query = select(RoundtableMemory).where(RoundtableMemory.symbol == symbol)
    if as_of is not None:
        same_query = same_query.where(RoundtableMemory.created_at < as_of)
    same = (await db.execute(
        same_query
        .order_by(RoundtableMemory.trade_date.desc())
        .limit(same_limit)
    )).scalars().all()

    if same:
        lines = []
        for m in same:
            head = f"- {m.trade_date} 决策 {m.signal}（信心 {m.confidence if m.confidence is not None else '—'}）"
            known_settlement = m.settled and (
                as_of is None or (m.settled_at is not None and m.settled_at < as_of)
            )
            if known_settlement and m.realized_return_pct is not None:
                head += f" → 持有{m.horizon_days}日实际 {m.realized_return_pct:+.2f}%，超额 {m.alpha_pct if m.alpha_pct is not None else '—'}%"
            else:
                head += " → 尚未结算"
            lines.append(head)
            if known_settlement and m.reflection:
                lines.append(f"  复盘：{m.reflection[:240]}")
        blocks.append("### 同标的最近决策\n" + "\n".join(lines))

    lesson_query = select(RoundtableMemory).where(
            RoundtableMemory.symbol != symbol,
            RoundtableMemory.settled.is_(True),
            RoundtableMemory.reflection.is_not(None),
        )
    if as_of is not None:
        lesson_query = lesson_query.where(RoundtableMemory.settled_at < as_of)
    others = (await db.execute(
        lesson_query
        .order_by(RoundtableMemory.settled_at.desc())
        .limit(lesson_limit)
    )).scalars().all()

    if others:
        lines = [
            f"- {m.name}({m.symbol}) {m.signal} 于 {m.trade_date}："
            f"{m.realized_return_pct:+.2f}%，超额 {m.alpha_pct if m.alpha_pct is not None else '—'}%；"
            f"{m.reflection[:200]}"
            for m in others
        ]
        blocks.append("### 其他标的的近期教训\n" + "\n".join(lines))

    return "\n\n".join(blocks)


def compute_settlement(df: pd.DataFrame, bench_df: pd.DataFrame,
                       trade_date: str, horizon_days: int,
                       entry_price: Optional[float] = None) -> Optional[Dict[str, Any]]:
    """按真实交易日结算一条决策。

    返回 ``None`` 表示「持有期还没走完 / 数据不足」，调用方应保持未结算状态。
    交易日不足时不结算，避免用自然日糊弄节假日。
    """
    if df is None or df.empty or horizon_days < 1:
        return None
    base = df[df["date"] <= trade_date]
    # An entry price alone cannot prove the returned window covers the horizon.
    if base.empty:
        return None
    after = df[df["date"] > trade_date].reset_index(drop=True)
    if len(after) < horizon_days:
        return None

    entry = _f(entry_price)
    if not entry:
        entry = _f(base["close"].iloc[-1]) if not base.empty else None
    exit_date = after["date"].iloc[horizon_days - 1]
    exit_price = _f(after["close"].iloc[horizon_days - 1])
    if not entry or not exit_price:
        return None

    ret = round((exit_price / entry - 1) * 100, 2)
    bench_ret = None
    if bench_df is not None and not bench_df.empty:
        b_exit = bench_df[bench_df["date"] == exit_date]
        b_base = bench_df[bench_df["date"] <= trade_date]
        if not b_exit.empty and not b_base.empty:
            b0 = _f(b_base["close"].iloc[-1])
            b1 = _f(b_exit["close"].iloc[-1])
            if b0 and b1:
                bench_ret = round((b1 / b0 - 1) * 100, 2)

    return {
        "entry_price": entry,
        "exit_price": exit_price,
        "realized_return_pct": ret,
        "benchmark": _BENCH_NAME,
        "benchmark_return_pct": bench_ret,
        "alpha_pct": round(ret - bench_ret, 2) if bench_ret is not None else None,
    }


async def settle_due(db: AsyncSession, failover, gateway,
                     config: RoundtableConfig, limit: int = 3) -> Dict[str, Any]:
    """结算所有已到持有期但未结算的记忆。返回统计，供前端展示。"""
    rows = (await db.execute(
        select(RoundtableMemory)
        .where(RoundtableMemory.settled.is_(False))
        .order_by(RoundtableMemory.trade_date.asc())
        .limit(limit)
    )).scalars().all()

    stats = {"checked": len(rows), "settled": 0, "skipped": 0, "failed": 0}
    if not rows:
        return stats

    bench_df = await _load_bench()
    reflector = Reflector(gateway, config)

    for row in rows:
        try:
            df = await _load_series(failover, row.symbol, row.trade_date)
            if df.empty:
                stats["failed"] += 1
                continue
            td = row.trade_date.isoformat() if isinstance(row.trade_date, date) else str(row.trade_date)
            outcome = compute_settlement(df, bench_df, td, row.horizon_days, row.entry_price)
            if outcome is None:
                stats["skipped"] += 1
                continue

            reflection = await reflector.reflect(
                row.symbol, row.name, row.signal,
                outcome["entry_price"], outcome["exit_price"],
                outcome["realized_return_pct"], outcome["alpha_pct"],
                rationale=f"{row.signal}，信心 {row.confidence}",
            )

            row.entry_price = outcome["entry_price"]
            row.realized_return_pct = outcome["realized_return_pct"]
            row.benchmark = outcome["benchmark"]
            row.benchmark_return_pct = outcome["benchmark_return_pct"]
            row.alpha_pct = outcome["alpha_pct"]
            row.reflection = reflection
            row.settled = True
            row.settled_at = now_utc()
            stats["settled"] += 1
        except Exception as exc:  # 单条失败不影响整轮
            logger.warning(f"[memory] settle failed session={row.session_id}: {exc}")
            stats["failed"] += 1

    await db.flush()
    return stats


def serialize_memory(row: RoundtableMemory) -> Dict[str, Any]:
    return {
        "session_id": row.session_id,
        "symbol": row.symbol,
        "name": row.name,
        "trade_date": row.trade_date.isoformat() if row.trade_date else None,
        "signal": row.signal,
        "confidence": row.confidence,
        "entry_price": row.entry_price,
        "horizon_days": row.horizon_days,
        "target_date": row.target_date.isoformat() if row.target_date else None,
        "realized_return_pct": row.realized_return_pct,
        "benchmark": row.benchmark,
        "benchmark_return_pct": row.benchmark_return_pct,
        "alpha_pct": row.alpha_pct,
        "reflection": row.reflection,
        "settled": bool(row.settled),
        "settled_at": row.settled_at.isoformat() if row.settled_at else None,
        "created_at": row.created_at.isoformat() if row.created_at else None,
    }
