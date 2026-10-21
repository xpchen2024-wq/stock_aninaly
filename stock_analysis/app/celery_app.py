# ============================================================================
# AI Stock Analysis Platform - Celery Configuration
# ============================================================================
from __future__ import annotations

import logging
from celery import Celery
from celery.signals import worker_init
from app.config import get_settings

settings = get_settings()

logger = logging.getLogger(__name__)
logger.info(f"Celery broker: {settings.CELERY_BROKER_URL}")

celery_app = Celery(
    "ai_stock_analysis",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND,
)

celery_app.conf.update(
    # Serialization
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="Asia/Shanghai",
    enable_utc=True,

    # Task routing
    task_routes={
        "crawl_news": {"queue": "crawler"},
        "tag_news_topics": {"queue": "analysis"},
        "compute_hot_topics": {"queue": "analysis"},
        "crawl_kol_opinions": {"queue": "crawler"},
        "generate_kol_consensus": {"queue": "analysis"},
        "cleanup_expired_data": {"queue": "maintenance"},
        "update_watchlist_cache": {"queue": "data"},
    },

    # Beat schedule (periodic tasks)
    beat_schedule={
        # HT-001/HT-003: News crawl every 15 minutes
        "crawl-news": {
            "task": "crawl_news",
            "schedule": 15 * 60,
        },
        # HT-002: 为新采集的新闻打题材标签（与采集同频，15 分钟）
        "tag-news-topics": {
            "task": "tag_news_topics",
            "schedule": 15 * 60,
        },
        # HT-002/HT-003: Hot topics every 15 minutes
        "compute-hot-topics": {
            "task": "compute_hot_topics",
            "schedule": 15 * 60,
        },
        # KV-001: KOL opinions every 30 minutes
        "crawl-kol-opinions": {
            "task": "crawl_kol_opinions",
            "schedule": 30 * 60,
        },
        # KV-005/AC-008: KOL consensus daily at 18:00
        "generate-kol-consensus": {
            "task": "generate_kol_consensus",
            "schedule": 60 * 60 * 18,
        },
        # Cleanup daily at 02:00
        "cleanup-expired": {
            "task": "cleanup_expired_data",
            "schedule": 60 * 60 * 2,
        },
        # WL-004: Watchlist cache daily at 18:00 (after market close)
        "update-watchlist-cache": {
            "task": "update_watchlist_cache",
            "schedule": 60 * 60 * 18,
        },
    },
)

celery_app.autodiscover_tasks(["app"])


# -- Worker 启动时装载模型配置 -----------------------------------------------
# 任务进程通过 get_model_gateway() 取网关，若不装载则网关为空，AI 相关任务全失效。
@worker_init.connect
def _load_model_configs_on_worker_init(**kwargs):
    """Worker 子进程初始化时从数据库装载模型配置。"""
    import asyncio
    from app.model_gateway import load_configs_from_db
    try:
        count = asyncio.run(load_configs_from_db())
        logger.info(f"Worker model configs loaded: {count}")
    except Exception as exc:
        logger.error(f"Worker failed to load model configs: {exc}", exc_info=True)
