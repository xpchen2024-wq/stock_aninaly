"""Check current KOL data state."""
import asyncio
from app.database import async_session_factory
from app.models import KOL, KOLOpinion, KOLConsensus, HotTopic
from sqlalchemy import select, func


async def show():
    async with async_session_factory() as db:
        kols = (await db.execute(
            select(KOL).where(KOL.is_active == True).order_by(KOL.platform, KOL.rank_position)
        )).scalars().all()
        print(f"KOLs: {len(kols)}")
        for k in kols[:5]:
            print(f"  {k.platform} | {k.nickname} | id={k.id[:8]}... | followers={k.followers_count}")
        op_count = await db.scalar(select(func.count(KOLOpinion.id)))
        print(f"Opinions total: {op_count}")
        cons_count = await db.scalar(select(func.count(KOLConsensus.id)))
        print(f"Consensus total: {cons_count}")
        ht_count = await db.scalar(select(func.count(HotTopic.id)))
        print(f"HotTopics total: {ht_count}")
        # Get all KOLs with id, platform, nickname
        print("\nAll KOLs:")
        for k in kols:
            print(f"  {k.platform} | {k.nickname} | id={k.id}")


asyncio.run(show())
