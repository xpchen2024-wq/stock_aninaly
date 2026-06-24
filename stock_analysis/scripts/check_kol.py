"""Check KOL opinion data freshness."""
import asyncio
from datetime import datetime, timedelta
from sqlalchemy import select, func
from app.database import async_session_factory
from app.models import KOLOpinion


async def check():
    async with async_session_factory() as db:
        total = (await db.execute(select(func.count(KOLOpinion.id)))).scalar()
        cutoff = datetime.now(datetime.timezone.utc) - timedelta(hours=48)
        recent = (await db.execute(
            select(func.count(KOLOpinion.id)).where(KOLOpinion.collected_at >= cutoff)
        )).scalar()
        print(f'Total KOL opinions: {total}')
        print(f'Recent (48h): {recent}')
        most = (await db.execute(select(KOLOpinion).order_by(KOLOpinion.collected_at.desc()).limit(3))).scalars().all()
        for o in most:
            print(f'  - collected_at={o.collected_at} | {o.summary[:50] if o.summary else ""}')


asyncio.run(check())
