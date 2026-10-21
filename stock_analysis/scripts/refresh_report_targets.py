"""Re-sync research report target prices against live quotes.

Research reports are dated snapshots, so a target price goes stale as the
market moves. This script re-derives each target from the current price using
the rating's implied upside band, so "买入 + 目标价" can never contradict the
live price.

Usage:
    python scripts/refresh_report_targets.py            # dry run (report only)
    python scripts/refresh_report_targets.py --apply    # write to DB
"""
import asyncio
import sys

from sqlalchemy import select

from app.database import async_session_factory
from app.models import ResearchReport
from app.reports import live_prices, target_from_price


async def main(apply: bool = False):
    async with async_session_factory() as db:
        result = await db.execute(select(ResearchReport))
        reports = result.scalars().all()
        if not reports:
            print("No reports found.")
            return

        symbols = {r.stock_symbol.split(".")[0] for r in reports if r.stock_symbol}
        prices = await live_prices(symbols)
        print(f"Resolved live prices for {len(prices)}/{len(symbols)} symbols\n")

        changed = 0
        for r in reports:
            bare = (r.stock_symbol or "").split(".")[0]
            price = prices.get(bare)
            if not price:
                print(f"  ! {r.stock_name} ({bare}) price unavailable - skipped")
                continue
            new_target = target_from_price(price, r.rating, r.id)
            if new_target is None:
                continue
            old = r.target_price
            if old is None or abs(old - new_target) > 0.01:
                changed += 1
                print(f"  · {r.stock_name} {bare} {r.rating}: "
                      f"{old if old is not None else '-'} -> {new_target} "
                      f"(现价 {price}, 空间 {(new_target / price - 1) * 100:+.1f}%)")
                if apply:
                    r.target_price = new_target

        if apply:
            await db.commit()
            print(f"\n✓ Committed: {changed} target prices updated")
        else:
            print(f"\nDry run: {changed} target prices would change. Use --apply to write.")


if __name__ == "__main__":
    asyncio.run(main(apply="--apply" in sys.argv))
