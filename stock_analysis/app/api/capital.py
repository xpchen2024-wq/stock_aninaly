# ============================================================================
# AI Stock Analysis Platform - Capital Flow & Shareholder Intelligence API
# ============================================================================
"""资金 & 筹码数据：主力资金流、基金持股变动、十大股东（含国家队识别）。

Data sources (akshare):
- stock_main_stock_holder      : 主要股东历史（含持股比例，多报告期）
- stock_report_fund_hold       : 基金持股季度统计（全市场 → 过滤个股）
- stock_individual_fund_flow   : 个股资金流向（东财，偶发不可用 → 明确降级）

国家队识别：按股东名称关键词匹配汇金/证金/大基金/社保/养老金等。
"""
from __future__ import annotations

import asyncio
import logging
import time
from datetime import date, timedelta
from typing import Any, Dict, List, Optional, Tuple

import requests
from fastapi import APIRouter, HTTPException, Query

logger = logging.getLogger(__name__)
router = APIRouter()

# 东财主力持仓接口：季度数据，按报告期缓存全市场快照（6 小时）
_FUND_API = "http://data.eastmoney.com/dataapi/zlsj/list"
_FUND_MAP_CACHE: Dict[Tuple[str, str], Tuple[float, Dict[str, Dict[str, Any]]]] = {}
_FUND_HOLD_TTL = 6 * 3600.0


def _fetch_fund_map(period: str, org_type: str = "1") -> Dict[str, Dict[str, Any]]:
    """单报告期全市场主力持仓快照 → {股票代码: 原始记录}。

    直接调用东财 JSON 接口：akshare 的 stock_report_fund_hold 依赖固定列序，
    东财字段调整后会出现列错位（持股数被读成市值），因此这里自行解析。
    """
    key = (period, org_type)
    cached = _FUND_MAP_CACHE.get(key)
    if cached and time.time() - cached[0] < _FUND_HOLD_TTL:
        return cached[1]
    date_str = f"{period[:4]}-{period[4:6]}-{period[6:]}"
    out: Dict[str, Dict[str, Any]] = {}
    page, pages = 1, 1
    while page <= pages and page <= 40:
        params = {
            "date": date_str, "type": org_type, "zjc": "0",
            "sortField": "HOULD_NUM", "sortDirec": "1",
            "pageNum": str(page), "pageSize": "500",
            "p": str(page), "pageNo": str(page),
        }
        payload = requests.get(_FUND_API, params=params, timeout=20).json()
        pages = int(payload.get("pages") or 1)
        for row in payload.get("data") or []:
            code = str(row.get("SECURITY_CODE") or "").strip().zfill(6)
            if code and code != "000000":
                out[code] = row
        page += 1
    _FUND_MAP_CACHE[key] = (time.time(), out)
    return out

# 国家队 / 主权类资金关键词
NATIONAL_FUND_KEYWORDS = [
    "中央汇金", "汇金", "中国证券金融", "证金公司", "证金资管",
    "国家集成电路产业投资基金", "大基金一期", "大基金二期", "大基金三期",
    "集成电路产业投资", "社保基金", "全国社会保障基金",
    "养老保险基金", "基本养老保险", "国新投资", "国新控股",
    "中投公司", "中国烟草", "外管局",
]


def _strip_suffix(symbol: str) -> str:
    """300750.SZ / 300750 -> 300750; also extract code from '300750.SZ 宁德时代'."""
    raw = symbol.strip().split()[0]
    return raw.split(".")[0]


def _market_of(symbol: str) -> str:
    raw = symbol.strip().split()[0].upper()
    if raw.endswith(".SH") or raw.startswith("6"):
        return "sh"
    return "sz"


def _is_national_fund(holder_name: str) -> Optional[str]:
    for kw in NATIONAL_FUND_KEYWORDS:
        if kw in holder_name:
            return kw
    return None


def _latest_periods(n: int = 6) -> List[str]:
    """Recent n quarterly report dates (YYYYMMDD, newest first) that are ≤ today."""
    today = date.today()
    candidates: List[str] = []
    for yy in (today.year, today.year - 1):
        for mm, day in ((12, 31), (9, 30), (6, 30), (3, 31)):
            d = date(yy, mm, day)
            if d <= today:
                candidates.append(d.strftime("%Y%m%d"))
    return sorted(set(candidates), reverse=True)[:n]


@router.get("/shareholders")
async def get_shareholders(symbol: str = Query(..., description="如 300750.SZ 或 300750")):
    """主要股东：最新报告期 Top10 + 国家队识别 + 相邻报告期持股变动。"""
    import akshare as ak

    code = _strip_suffix(symbol)
    logger.info(f"Capital shareholders query: {code}")

    def _load():
        return ak.stock_main_stock_holder(stock=code)

    try:
        df = await asyncio.wait_for(asyncio.to_thread(_load), timeout=25.0)
    except Exception as e:
        logger.warning(f"main_stock_holder failed for {code}: {e}")
        raise HTTPException(status_code=502, detail=f"股东数据获取失败: {e}")

    if df is None or df.empty:
        raise HTTPException(status_code=422, detail="未查询到该股票的股东数据")

    # 统一日期为 YYYY-MM-DD（源数据可能是 date/datetime/Timestamp 混合）
    def _norm_date(v):
        s = str(v)[:10]
        return s
    df["截至日期"] = df["截至日期"].map(_norm_date)
    latest_period = df["截至日期"].max()
    prev_df = df[df["截至日期"] < latest_period]
    prev_period = prev_df["截至日期"].max() if not prev_df.empty else None

    def _opt_float(v) -> Optional[float]:
        try:
            f = float(v)
            return None if f != f else f   # NaN → 未知
        except (TypeError, ValueError):
            return None

    def _row_to_item(row) -> Dict[str, Any]:
        name = str(row["股东名称"])
        national_kw = _is_national_fund(name)
        shares = _opt_float(row.get("持股数量"))
        ratio = _opt_float(row.get("持股比例"))
        return {
            "name": name,
            "shares": shares,
            "ratio_pct": ratio,
            "nature": str(row.get("股本性质", "") or ""),
            "is_national": national_kw is not None,
            "national_tag": national_kw,
            "change_shares": None,
            "change_ratio_pts": None,
            "change_note": None,
        }

    latest_rows = df[df["截至日期"] == latest_period].head(10)
    holders = [_row_to_item(r) for _, r in latest_rows.iterrows()]

    # 相邻报告期变动（同名称股东）
    if prev_period:
        import math
        prev_map = {
            str(r["股东名称"]): _opt_float(r.get("持股数量"))
            for _, r in prev_df[prev_df["截至日期"] == prev_period].iterrows()
        }
        for h in holders:
            if h["name"] not in prev_map:
                h["change_note"] = "新进前十"
                continue
            prev_shares = prev_map.get(h["name"])
            # 当前期持股数缺失时不计算变动，避免把“未知”误报为清仓
            if h["shares"] is not None and prev_shares is not None:
                h["change_shares"] = int(h["shares"] - prev_shares)
                if prev_shares > 0:
                    h["change_ratio_pts"] = round(
                        h["shares"] / prev_shares - 1, 4)
            else:
                h["change_note"] = "数据缺失"

    national_holders = [h for h in holders if h["is_national"]]
    total_ratio = sum(h["ratio_pct"] or 0 for h in holders)

    total_holder_count = None
    if "股东总数" in df.columns and len(latest_rows):
        try:
            v = float(df[df["截至日期"] == latest_period]["股东总数"].iloc[0])
            total_holder_count = int(v) if v == v else None  # NaN guard
        except (ValueError, TypeError, IndexError):
            total_holder_count = None

    return {
        "symbol": code,
        "period": latest_period,
        "prev_period": prev_period,
        "holders": holders,
        "national_summary": {
            "count": len(national_holders),
            "holders": [
                {"name": h["name"], "tag": h["national_tag"],
                 "shares": h["shares"], "ratio_pct": h["ratio_pct"],
                 "change_shares": h["change_shares"]}
                for h in national_holders
            ],
        },
        "top10_total_ratio_pct": round(total_ratio, 2),
        "total_holder_count": total_holder_count,
    }


@router.get("/fund-holdings")
async def get_fund_holdings(
    symbol: str = Query(...),
    periods: int = Query(4, ge=1, le=8, description="最近 N 个报告期"),
):
    """基金持股变动趋势：逐报告期的家数/持股数/市值/增减仓。"""
    code = _strip_suffix(symbol).zfill(6)
    period_list = list(reversed(_latest_periods(periods)))  # oldest → newest
    results: List[Dict[str, Any]] = []

    snapshots: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for p in period_list:
        try:
            snapshots[p] = await asyncio.to_thread(_fetch_fund_map, p)
        except Exception as e:
            logger.warning(f"fund_hold {p} failed: {e}")
            snapshots[p] = {}

    # 报告期披露存在滞后：尚未完整披露的期次只覆盖少量个股，
    # 用“相对最大快照规模的 50%”作为披露完整性门槛，避免误判为“未持有”。
    sizes = [len(v) for v in snapshots.values() if v]
    floor = 0.5 * max(sizes) if sizes else 0

    for p in period_list:
        snap = snapshots.get(p) or {}
        if len(snap) < floor:
            continue
        row = snap.get(code)
        if not row:
            results.append({
                "period": p, "fund_count": 0, "total_shares": 0,
                "market_value": 0.0, "change_flag": "未持有",
                "change_shares": None, "change_pct": None,
            })
            continue
        results.append({
            "period": p,
            "fund_count": int(_safe_float(row.get("HOULD_NUM")) or 0),
            "total_shares": int(_safe_float(row.get("TOTAL_SHARES")) or 0),
            "market_value": _safe_float(row.get("HOLD_VALUE")) or 0.0,
            "change_flag": str(row.get("HOLDCHA") or ""),
            "change_shares": _safe_float(row.get("HOLDCHA_NUM")),
            "change_pct": _safe_float(row.get("HOLDCHA_RATIO")),
        })

    if not results:
        raise HTTPException(status_code=422, detail="近几个报告期均无该股票的基金持股数据")

    latest = results[-1]
    national_view = None
    return {
        "symbol": code,
        "latest": latest,
        "history": results,
        "trend_note": _fund_trend_note(results),
    }


def _safe_float(v):
    try:
        f = float(v)
        return f if f == f else None   # NaN guard
    except (TypeError, ValueError):
        return None


def _fund_trend_note(history: List[Dict[str, Any]]) -> str:
    """根据最近两期生成一句趋势解读。"""
    if len(history) < 2:
        return history[0]["change_flag"] if history else ""
    prev, cur = history[-2], history[-1]
    d_funds = cur["fund_count"] - prev["fund_count"]
    d_shares = None
    if cur["total_shares"] and prev["total_shares"]:
        d_shares = cur["total_shares"] - prev["total_shares"]
    bits = []
    bits.append(f"持有基金 {prev['fund_count']}→{cur['fund_count']} 家"
                f"（{'+' if d_funds >= 0 else ''}{d_funds}）")
    if d_shares is not None:
        pct = (cur["total_shares"] / prev["total_shares"] - 1) * 100 if prev["total_shares"] else 0
        bits.append(f"持股量 {'+' if d_shares >= 0 else ''}{d_shares:,} 股（{pct:+.1f}%）")
    bits.append(f"最新口径：{cur['change_flag'] or '—'}")
    return "；".join(bits)


@router.get("/money-flow")
async def get_money_flow(
    symbol: str = Query(...),
    days: int = Query(10, ge=1, le=100),
):
    """个股主力资金流向。东财接口偶发不可用 → status=unavailable 明确降级。"""
    import akshare as ak

    code = _strip_suffix(symbol)
    market = _market_of(symbol)
    logger.info(f"Capital money-flow query: {code} market={market}")

    def _load():
        return ak.stock_individual_fund_flow(stock=code, market=market)

    try:
        df = await asyncio.wait_for(asyncio.to_thread(_load), timeout=20.0)
    except Exception as e:
        logger.warning(f"money_flow failed for {code}: {e}")
        return {
            "status": "unavailable",
            "message": "资金流数据源暂时不可用（东财接口限流或网络异常），请稍后重试",
            "symbol": code,
        }

    if df is None or df.empty:
        return {"status": "unavailable", "message": "数据源未返回资金流数据", "symbol": code}

    col_map = {
        "日期": "date", "收盘价": "close", "涨跌幅": "change_pct",
        "主力净流入-净额": "main_net", "主力净流入-净占比": "main_net_pct",
        "超大单净流入-净额": "super_net", "大单净流入-净额": "big_net",
        "中单净流入-净额": "mid_net", "小单净流入-净额": "small_net",
    }
    rows = []
    df = df.tail(days)
    for _, r in df.iterrows():
        item = {}
        for src, dst in col_map.items():
            if src in r.index:
                v = r[src]
                try:
                    item[dst] = round(float(v), 4)
                except (TypeError, ValueError):
                    item[dst] = str(v)
        rows.append(item)

    recent_main_sum = sum(r.get("main_net", 0) or 0 for r in rows if isinstance(r.get("main_net"), (int, float)))
    up_days = sum(1 for r in rows if isinstance(r.get("main_net"), (int, float)) and r["main_net"] > 0)

    return {
        "status": "ok",
        "symbol": code,
        "days": len(rows),
        "rows": rows,
        "summary": {
            "recent_main_net_total": round(recent_main_sum, 2),
            "main_inflow_days": up_days,
            "note": f"近 {len(rows)} 日主力净流入合计 {'+' if recent_main_sum >= 0 else ''}"
                    f"{recent_main_sum / 1e8:.2f} 亿元，其中 {up_days} 天为净流入",
        },
    }


@router.get("/national-funds")
async def get_national_fund_presence(symbol: str = Query(...)):
    """跨报告期的国家队持仓追踪（出现过的历史期次与持股变化）。"""
    import akshare as ak

    code = _strip_suffix(symbol)

    def _load():
        return ak.stock_main_stock_holder(stock=code)

    try:
        df = await asyncio.wait_for(asyncio.to_thread(_load), timeout=25.0)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"股东数据获取失败: {e}")

    if df is None or df.empty:
        raise HTTPException(status_code=422, detail="未查询到股东数据")

    df["截至日期"] = df["截至日期"].astype(str).map(lambda v: str(v)[:10])
    records = []
    for _, r in df.iterrows():
        name = str(r["股东名称"])
        tag = _is_national_fund(name)
        if tag:
            shares = _safe_float(r.get("持股数量"))
            ratio = _safe_float(r.get("持股比例"))
            if shares is None and ratio is None:
                continue
            records.append({
                "period": r["截至日期"],
                "name": name,
                "tag": tag,
                "shares": shares or 0.0,
                "ratio_pct": ratio or 0.0,
                "announce_date": str(r.get("公告日期", "") or ""),
            })

    by_tag: Dict[str, List[Dict[str, Any]]] = {}
    for rec in sorted(records, key=lambda x: x["period"], reverse=True):
        by_tag.setdefault(rec["tag"], []).append(rec)

    return {
        "symbol": code,
        "has_national_presence": bool(records),
        "funds": [
            {"tag": tag, "appearances": len(recs), "timeline": recs[:8]}
            for tag, recs in by_tag.items()
        ],
    }
