# ============================================================================
# AI Stock Analysis Platform - Database Core
# ============================================================================
from __future__ import annotations

from contextlib import asynccontextmanager

from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.pool import NullPool
from app.config import get_settings

settings = get_settings()

engine = create_async_engine(
    settings.DATABASE_URL,
    echo=settings.APP_DEBUG,
    pool_size=20,
    max_overflow=10,
    pool_pre_ping=True,
)

async_session_factory = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


class Base(DeclarativeBase):
    """Base class for all ORM models."""
    pass


async def get_db() -> AsyncSession:
    """FastAPI dependency for database sessions."""
    async with async_session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


async def init_db():
    """Initialize database - create all tables."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def close_db():
    """Close database connections."""
    await engine.dispose()


@asynccontextmanager
async def task_session():
    """Celery 任务专用数据库会话。

    Celery prefork worker 中每个任务都通过 ``asyncio.run()`` 新建事件循环，而上面的
    全局 engine 的连接池在第一次使用后即绑定在那个 loop 上。下一个任务在新 loop 中
    复用池中连接会抛出：

        RuntimeError: Task ... got Future ... attached to a different loop

    因此任务侧单独使用 NullPool 引擎：连接按需新建、用完即关，不跨 loop 复用。
    API 侧继续使用带连接池的全局 engine —— uvicorn 全程只有一个事件循环，复用是安全的。
    """
    task_engine = create_async_engine(
        settings.DATABASE_URL,
        poolclass=NullPool,
        echo=settings.APP_DEBUG,
    )
    factory = async_sessionmaker(
        task_engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )
    try:
        async with factory() as session:
            yield session
    finally:
        await task_engine.dispose()
