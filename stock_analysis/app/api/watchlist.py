# ============================================================================
# AI Stock Analysis Platform - Watchlist API (WL-001 ~ WL-005)
# ============================================================================
from __future__ import annotations

import logging
from typing import Optional, List
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, delete

from app.database import get_db
from app.models import WatchlistGroup, WatchlistItem, User
from app.api.auth import get_current_user

logger = logging.getLogger(__name__)
router = APIRouter()


# -- Schemas --
class GroupCreate(BaseModel):
    name: str
    description: Optional[str] = None
    sort_order: int = 0


class GroupResponse(BaseModel):
    id: str
    name: str
    description: Optional[str]
    sort_order: int


class ItemAdd(BaseModel):
    symbol: str
    name: str
    market: Optional[str] = None
    group_id: Optional[str] = None
    notes: Optional[str] = None


class ItemResponse(BaseModel):
    id: str
    symbol: str
    name: str
    market: Optional[str]
    group_id: Optional[str]
    notes: Optional[str]
    added_at: str


class SearchResponse(BaseModel):
    symbol: str
    name: str
    market: Optional[str]


class ExportResponse(BaseModel):
    format: str
    count: int
    data: list


# -- Routes --
@router.get("/groups", response_model=List[GroupResponse])
async def list_groups(
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """List watchlist groups (WL-002)."""
    logger.info(f"Listing watchlist groups: user_id={user.id}")
    result = await db.execute(
        select(WatchlistGroup)
        .where(WatchlistGroup.user_id == str(user.id))
        .order_by(WatchlistGroup.sort_order)
    )
    groups = result.scalars().all()
    logger.info(f"Watchlist groups found: {len(groups)}")
    return [
        GroupResponse(
            id=str(g.id), name=g.name, description=g.description,
            sort_order=g.sort_order,
        )
        for g in groups
    ]


@router.post("/groups", response_model=GroupResponse, status_code=201)
async def create_group(
    req: GroupCreate,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    logger.info(f"Creating watchlist group: name={req.name}, user_id={user.id}")
    group = WatchlistGroup(
        id=str(uuid4()), user_id=str(user.id),
        name=req.name, description=req.description,
        sort_order=req.sort_order,
    )
    db.add(group)
    await db.flush()
    logger.info(f"Watchlist group created: id={group.id}, name={group.name}")
    return GroupResponse(
        id=str(group.id), name=group.name,
        description=group.description, sort_order=group.sort_order,
    )


@router.delete("/groups/{group_id}", status_code=204)
async def delete_group(
    group_id: str,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    logger.info(f"Deleting watchlist group: id={group_id}, user_id={user.id}")
    result = await db.execute(
        select(WatchlistGroup).where(
            WatchlistGroup.id == group_id,
            WatchlistGroup.user_id == str(user.id),
        )
    )
    group = result.scalar_one_or_none()
    if not group:
        logger.warning(f"Watchlist group not found: id={group_id}")
        raise HTTPException(status_code=404, detail="Group not found")
    logger.info(f"Deleting watchlist group: name={group.name}")
    await db.delete(group)


@router.get("/items", response_model=List[ItemResponse])
async def list_items(
    group_id: Optional[str] = Query(None),
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """List watchlist items (WL-001)."""
    logger.info(f"Listing watchlist items: user_id={user.id}, group_id={group_id or 'all'}")
    query = select(WatchlistItem).where(WatchlistItem.user_id == str(user.id))
    if group_id:
        query = query.where(WatchlistItem.group_id == group_id)
    query = query.order_by(WatchlistItem.added_at.desc())

    result = await db.execute(query)
    items = result.scalars().all()
    logger.info(f"Watchlist items found: {len(items)}")
    return [
        ItemResponse(
            id=str(i.id), symbol=i.symbol, name=i.name,
            market=i.market, group_id=str(i.group_id) if i.group_id else None,
            notes=i.notes,
            added_at=i.added_at.isoformat() if i.added_at else "",
        )
        for i in items
    ]


@router.post("/items", response_model=ItemResponse, status_code=201)
async def add_item(
    req: ItemAdd,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Add stock to watchlist (WL-001, WL-004 triggers cache)."""
    logger.info(f"Adding to watchlist: symbol={req.symbol}, name={req.name}, user_id={user.id}")
    existing = await db.execute(
        select(WatchlistItem).where(
            WatchlistItem.user_id == str(user.id),
            WatchlistItem.symbol == req.symbol,
        )
    )
    if existing.scalar_one_or_none():
        logger.warning(f"Stock already in watchlist: symbol={req.symbol}")
        raise HTTPException(status_code=409, detail="Stock already in watchlist")

    item = WatchlistItem(
        id=str(uuid4()), user_id=str(user.id),
        group_id=req.group_id, symbol=req.symbol,
        name=req.name, market=req.market, notes=req.notes,
    )
    db.add(item)
    await db.flush()
    logger.info(f"Watchlist item added: id={item.id}, symbol={item.symbol}")
    return ItemResponse(
        id=str(item.id), symbol=item.symbol, name=item.name,
        market=item.market, group_id=str(item.group_id) if item.group_id else None,
        notes=item.notes,
        added_at=item.added_at.isoformat() if item.added_at else "",
    )


@router.delete("/items/{item_id}", status_code=204)
async def remove_item(
    item_id: str,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    logger.info(f"Removing watchlist item: id={item_id}, user_id={user.id}")
    result = await db.execute(
        select(WatchlistItem).where(
            WatchlistItem.id == item_id,
            WatchlistItem.user_id == str(user.id),
        )
    )
    item = result.scalar_one_or_none()
    if not item:
        logger.warning(f"Watchlist item not found: id={item_id}")
        raise HTTPException(status_code=404, detail="Item not found")
    logger.info(f"Removing watchlist item: symbol={item.symbol}")
    await db.delete(item)


@router.get("/search", response_model=List[SearchResponse])
async def search_stocks(
    q: str = Query(..., min_length=1, description="Code or name"),
    limit: int = Query(10, le=50),
):
    """Search stocks by code or name (WL-001: fuzzy match)."""
    logger.info(f"Searching stocks: q={q}, limit={limit}")
    # Simple built-in search; in production would query a stock reference table
    import akshare as ak
    try:
        df = ak.stock_zh_a_spot_em()
        if df is None or df.empty:
            return []
        q_lower = q.lower()
        mask = (
            df["代码"].str.contains(q_lower, na=False) |
            df["名称"].str.contains(q, na=False)
        )
        matched = df[mask].head(limit)
        result_list = [
            SearchResponse(
                symbol=str(row["代码"]), name=str(row["名称"]), market=None,
            )
            for _, row in matched.iterrows()
        ]
        logger.info(f"Stock search results: q={q}, count={len(result_list)}")
        return result_list
    except Exception as e:
        logger.warning(f"Stock search failed: q={q}, error={e}")
        return []


@router.get("/export", response_model=ExportResponse)
async def export_watchlist(
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Export watchlist (WL-005)."""
    logger.info(f"Exporting watchlist: user_id={user.id}")
    result = await db.execute(
        select(WatchlistItem).where(WatchlistItem.user_id == str(user.id))
    )
    items = result.scalars().all()
    data = [
        {
            "symbol": i.symbol, "name": i.name,
            "market": i.market, "notes": i.notes,
        }
        for i in items
    ]
    logger.info(f"Watchlist exported: count={len(data)}")
    return ExportResponse(format="json", count=len(data), data=data)
