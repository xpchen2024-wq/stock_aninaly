"""Backfill trend_analyses from the deterministic forecast engine.

The 大V观点 cross-comparison table compares KOL direction vs system trend vs
research-report rating. The trend rows used to be stale (4 months old, only a
handful of symbols), so the comparison showed "无" for most hot stocks.

This script recomputes the trend for every stock that KOLs mention (plus any
explicitly requested codes) using the same engine that serves
`/api/v1/forecast/stock`, so the numbers are real and reproducible.

Usage:
    python -m scripts.seed_trend_analyses            # KOL hot stocks
    python -m scripts.seed_trend_analyses 600519 002594
"""
import asyncio
import sys
from datetime import datetime, timedelta, timezone
from typing import Dict, List

from sqlalchemy import select

from app.database import async_session_factory
from app.models import KOLOpinion, TrendAnalysis, new_uuid
from app.api.forecast import _compute_stock_payload


async def _hot_symbols(db) -> List[str]:
    cutoff = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(hours=48)
    rows = (await db.execute(
        select(KOLOpinion).where(KOLOpinion.published_at >= cutoff)
    )).scalars().all()
    codes: Dict[str, int] = {}
    for op in rows:
        for stock in (op.mentioned_stocks or []):
            code = str(stock.get("code", "")).split(".")[0]
            if code:
                codes[code] = codes.get(code, 0) + 1
    return [c for c, _ in sorted(codes.items(), key=lambda kv: -kv[1])][:30]


def _direction(score: float) -> str:
    if score > 15:
        return "bullish"
    if score < -15:
        return "bearish"
    return "sideways"


def _confidence(score: float, sample: int) -> str:
    if abs(score) >= 45 and sample >= 200:
        return "high"
    if abs(score) >= 20:
        return "medium"
    return "low"


async def main(codes: List[str] = None):
    async with async_session_factory() as db:
        if not codes:
            codes = await _hot_symbols(db)
        print(f"🔍 Recomputing trend for {len(codes)} symbols: {','.join(codes)}")

        ok, failed = 0, 0
        for code in codes:
            try:
                payload = await _compute_stock_payload(code, horizon=10, days=700)
            except Exception as exc:
                print(f"   ! {code} failed: {exc}")
                failed += 1
                continue

            tech = payload.get("technical") or {}
            score = float(tech.get("score") or 0)
            levels = payload.get("key_levels") or {}
            sigs = payload.get("signals") or []
            mtf = payload.get("multi_timeframe") or {}
            daily = mtf.get("日线") or {}
            prob = payload.get("horizon_probs") or []
            p10 = next((p.get("probability") for p in prob if p.get("horizon") == 10), None)

            db.add(TrendAnalysis(
                id=new_uuid(),
                symbol=code,
                name=(payload.get("stock") or {}).get("name") or code,
                frequency="D",
                trend_direction=_direction(score),
                support_level=(levels.get("support") or [{}])[0].get("price") if levels.get("support") else None,
                resistance_level=(levels.get("resistance") or [{}])[0].get("price") if levels.get("resistance") else None,
                pattern_detected="; ".join(s.get("label", "") for s in sigs[:3]) or None,
                confidence=_confidence(score, int((payload.get("technical") or {}).get("sample") or 0)),
                ai_conclusion=(
                    f"技术评分 {score:+.0f}（{_direction(score)}）· "
                    f"日线 {daily.get('label', '-')} · "
                    f"10 日上涨概率 {p10:.1f}%" if p10 is not None else
                    f"技术评分 {score:+.0f}（{_direction(score)}）"
                ),
                raw_indicators={
                    "score": score,
                    "prob_10d": p10,
                    "signals": [s.get("label") for s in sigs[:5]],
                },
            ))
            ok += 1
            print(f"   ✓ {code} score={score:+.0f} dir={_direction(score)}")

        await db.commit()
        print(f"\n✓ Inserted {ok} trend analyses (failed: {failed})")


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1:] or None))
