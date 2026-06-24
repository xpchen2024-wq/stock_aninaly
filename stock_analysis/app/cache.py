# ============================================================================
# AI Stock Analysis Platform - Redis Cache
# ============================================================================
from __future__ import annotations

import json
import logging
from typing import Optional, Any, Dict

import redis.asyncio as aioredis
from app.config import get_settings

logger = logging.getLogger(__name__)

settings = get_settings()

# Global Redis connection pool
_redis_pool: Optional[aioredis.Redis] = None


async def get_redis() -> aioredis.Redis:
    """Get or create Redis connection."""
    global _redis_pool
    if _redis_pool is None:
        _redis_pool = aioredis.from_url(
            settings.REDIS_URL,
            encoding="utf-8",
            decode_responses=True,
        )
        await _redis_pool.ping()
        logger.info("Redis connection established")
    return _redis_pool


async def close_redis():
    """Close Redis connection."""
    global _redis_pool
    if _redis_pool:
        await _redis_pool.close()
        _redis_pool = None


class CacheManager:
    """Unified cache operations with TTL support."""

    @staticmethod
    async def get(key: str) -> Optional[str]:
        r = await get_redis()
        return await r.get(key)

    @staticmethod
    async def set(key: str, value: str, ttl: int = 60):
        r = await get_redis()
        await r.set(key, value, ex=ttl)

    @staticmethod
    async def get_json(key: str) -> Optional[dict]:
        val = await CacheManager.get(key)
        return json.loads(val) if val else None

    @staticmethod
    async def set_json(key: str, data: dict, ttl: int = 60):
        await CacheManager.set(key, json.dumps(data, ensure_ascii=False), ttl)

    @staticmethod
    async def delete(key: str):
        r = await get_redis()
        await r.delete(key)

    @staticmethod
    async def exists(key: str) -> bool:
        r = await get_redis()
        return await r.exists(key) > 0

    @staticmethod
    async def ttl(key: str) -> int:
        r = await get_redis()
        return await r.ttl(key)


# -- Domain-specific cache helpers --------------------------------------------

async def cache_realtime_quote(symbol: str, data: dict, ttl: int = 60):
    """Cache real-time stock quote."""
    await CacheManager.set_json(f"quote:{symbol}", data, ttl)


async def get_cached_quote(symbol: str) -> Optional[dict]:
    """Get cached real-time stock quote."""
    return await CacheManager.get_json(f"quote:{symbol}")


async def cache_model_configs(configs: list[dict], ttl: int = 300):
    """Cache AI model configurations."""
    await CacheManager.set_json("model_configs", configs, ttl)


async def get_cached_model_configs() -> Optional[list]:
    """Get cached AI model configurations."""
    return await CacheManager.get_json("model_configs")


async def cache_hot_topics(topics: list[dict], ttl: int = 900):
    """Cache hot topics list."""
    await CacheManager.set_json("hot_topics", topics, ttl)


async def get_cached_hot_topics() -> Optional[list]:
    """Get cached hot topics."""
    return await CacheManager.get_json("hot_topics")


async def cache_kol_heat_scores(scores: dict, ttl: int = 1800):
    """Cache KOL opinion heat scores."""
    await CacheManager.set_json("kol_heat_scores", scores, ttl)


async def get_cached_kol_heat_scores() -> Optional[dict]:
    """Get cached KOL heat scores."""
    return await CacheManager.get_json("kol_heat_scores")
