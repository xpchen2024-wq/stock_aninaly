# ============================================================================
# AI Stock Analysis Platform - Global Markets API
# ============================================================================
"""全球市场速览：美股 / 日韩科技股，用于 Dashboard 顶部一行行情条。

- GET /api/v1/market/global        ：行情速览（服务端缓存 55s，前端 60s 轮询）
- GET /api/v1/market/global?refresh=1：强制刷新（调试用）
"""
from __future__ import annotations

import logging

from fastapi import APIRouter, Query

from app.global_markets import get_global_markets

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/global")
async def global_markets(refresh: bool = Query(False, description="强制跳过缓存")):
    """美股 + 日韩科技股 最新涨跌（Yahoo Finance，非交易时段为最近收盘价）。"""
    return await get_global_markets(force=refresh)
