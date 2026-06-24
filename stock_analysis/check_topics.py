"""Check why hot-topics returns empty."""
import asyncio
from datetime import datetime
from app.database import async_session_factory
from app.models import HotTopic
from sqlalchemy import select


async def show():
    async with async_session_factory() as db:
        now = datetime.utcnow()
        topics = (await db.execute(
            select(HotTopic).order_by(HotTopic.heat_index.desc())
        )).scalars().all()
        print(f"HotTopics: {len(topics)} (now={now})")
        for t in topics:
            print(f"  {t.topic_name[:30]:30s} | heat={t.heat_index} | expires={t.expires_at} | generated={t.generated_at}")


asyncio.run(show())
