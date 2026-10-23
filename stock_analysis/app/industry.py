# ============================================================================
# AI Stock Analysis Platform - Industry (Shenwan) Analysis Engine
# ============================================================================
"""申万一级行业分析：行业动量、相对强弱、广度、估值与轮动定位。

数据源：
- sw_index_first_info         : 31 个一级行业的成分数 / PE / PB / 股息率
- index_realtime_sw           : 行业实时涨跌与成交额
- index_hist_sw               : 行业指数历史（约 7s/次，落盘缓存 1 天）
- index_component_sw          : 行业成分股与权重（落盘缓存 7 天）
- FailoverManager K 线        : 成分股行情（用于计算广度）

设计：历史与成分数据按天落盘到 .cache/sw/，首次访问后台预热，
未就绪时返回 pending=True，前端轮询补齐。
"""
from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import math
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

try:
    import fcntl
except ImportError:  # pragma: no cover - Windows
    fcntl = None  # type: ignore[assignment]

import numpy as np
import pandas as pd

from app import ta

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
CACHE_ROOT = ROOT / ".cache" / "sw"
CACHE_ROOT.mkdir(parents=True, exist_ok=True)

# 申万 2021 版一级行业（接口不可用时的兜底清单）
SW_LEVEL1: List[Tuple[str, str]] = [
    ("801010", "农林牧渔"), ("801030", "基础化工"), ("801040", "钢铁"),
    ("801050", "有色金属"), ("801080", "电子"), ("801110", "家用电器"),
    ("801120", "食品饮料"), ("801130", "纺织服饰"), ("801140", "轻工制造"),
    ("801150", "医药生物"), ("801160", "公用事业"), ("801170", "交通运输"),
    ("801180", "房地产"), ("801200", "商贸零售"), ("801210", "社会服务"),
    ("801230", "综合"), ("801710", "建筑材料"), ("801720", "建筑装饰"),
    ("801730", "电力设备"), ("801740", "国防军工"), ("801750", "计算机"),
    ("801760", "传媒"), ("801770", "通信"), ("801780", "银行"),
    ("801790", "非银金融"), ("801880", "汽车"), ("801890", "机械设备"),
    ("801950", "煤炭"), ("801960", "石油石化"), ("801970", "环保"),
    ("801980", "美容护理"),
]
SW_NAME = dict(SW_LEVEL1)

# 证监会行业大类 → 申万一级（精确映射未就绪时的近似兜底）
CSRC_TO_SW = {
    "农业": "农林牧渔", "林业": "农林牧渔", "畜牧业": "农林牧渔", "渔业": "农林牧渔",
    "农副食品加工业": "农林牧渔",
    "石油和天然气开采业": "石油石化", "开采专业及辅助性活动": "石油石化",
    "煤炭开采和洗选业": "煤炭",
    "黑色金属矿采选业": "钢铁", "黑色金属冶炼和压延加工业": "钢铁",
    "有色金属矿采选业": "有色金属", "有色金属冶炼和压延加工业": "有色金属",
    "非金属矿物制品业": "建筑材料", "非金属矿采选业": "建筑材料",
    "化学原料和化学制品制造业": "基础化工", "化学纤维制造业": "基础化工",
    "橡胶和塑料制品业": "基础化工", "石油加工、炼焦和核燃料加工业": "石油石化",
    "医药制造业": "医药生物",
    "电气机械和器材制造业": "电力设备",
    "计算机、通信和其他电子设备制造业": "电子",
    "通用设备制造业": "机械设备", "专用设备制造业": "机械设备",
    "金属制品业": "机械设备", "仪器仪表制造业": "机械设备",
    "铁路、船舶、航空航天和其他运输设备制造业": "国防军工",
    "汽车制造业": "汽车",
    "酒、饮料和精制茶制造业": "食品饮料", "食品制造业": "食品饮料",
    "纺织业": "纺织服饰", "纺织服装、服饰业": "纺织服饰",
    "皮革、毛皮、羽毛及其制品和制鞋业": "纺织服饰",
    "木材加工和木、竹、藤、棕、草制品业": "轻工制造",
    "家具制造业": "轻工制造", "造纸和纸制品业": "轻工制造",
    "印刷和记录媒介复制业": "轻工制造", "文教、工美、体育和娱乐用品制造业": "轻工制造",
    "电力、热力生产和供应业": "公用事业", "燃气生产和供应业": "公用事业",
    "水的生产和供应业": "公用事业",
    "道路运输业": "交通运输", "铁路运输业": "交通运输", "航空运输业": "交通运输",
    "水上运输业": "交通运输", "装卸搬运和运输代理业": "交通运输",
    "仓储业": "交通运输", "邮政业": "交通运输",
    "房地产业": "房地产", "房屋建筑业": "建筑装饰",
    "土木工程建筑业": "建筑装饰", "建筑安装业": "建筑装饰",
    "建筑装饰和其他建筑业": "建筑装饰",
    "批发业": "商贸零售", "零售业": "商贸零售",
    "住宿业": "社会服务", "餐饮业": "社会服务",
    "互联网和相关服务": "传媒", "软件和信息技术服务业": "计算机",
    "电信、广播电视和卫星传输服务": "通信",
    "新闻和出版业": "传媒", "广播、电视、电影和影视录音制作业": "传媒",
    "文化艺术业": "传媒", "体育": "社会服务",
    "货币金融服务": "银行", "资本市场服务": "非银金融",
    "保险业": "非银金融", "其他金融业": "非银金融",
    "研究和试验发展": "医药生物", "专业技术服务业": "机械设备",
    "生态保护和环境治理业": "环保", "公共设施管理业": "环保",
    "综合": "综合",
}

_MEM: Dict[str, Tuple[float, Any]] = {}
_MEM_TTL = 300.0
_PREWARM_LOCK = threading.Lock()
_BASE_LOCK = threading.Lock()
_BASE_FETCH_LOCK = threading.Lock()
_PREWARMING = {"hist": False, "map": False, "realtime": False}

# 实时清单接口 sw_index_first_info + index_realtime_sw 合计约 6s，是首屏
# 加载的最大瓶颈。过期后先用旧数据响应，后台异步刷新。
BASE_TTL_SECONDS = 300.0
# 行业历史为日线，2 天内都算新鲜；过期数据仍可用于展示。
HIST_TTL_SECONDS = 86400 * 2
# 成分股广度缓存 2 小时；过期后先返回旧值，后台重算。
BREADTH_TTL_SECONDS = 7200.0
# 广度只取权重最高的前 N 只，减少上游行情请求。
BREADTH_SAMPLE = 20
# 行业列表计算结果缓存 15s，避免轮询时重复解析 31 份历史 CSV。
LIST_MEM_TTL = 15.0
_BREADTH_TASKS: Dict[str, asyncio.Task] = {}


@contextlib.contextmanager
def _cache_file_lock(name: str, blocking: bool = True) -> Iterator[bool]:
    """跨进程文件锁（多 uvicorn worker 时只让一个进程刷新上游数据）。"""
    if fcntl is None:
        yield True
        return
    path = CACHE_ROOT / f".{name}.lock"
    try:
        handle = open(path, "a+")
    except OSError:
        yield True
        return
    acquired = False
    try:
        flags = fcntl.LOCK_EX if blocking else fcntl.LOCK_EX | fcntl.LOCK_NB
        try:
            fcntl.flock(handle.fileno(), flags)
            acquired = True
        except OSError:
            if blocking:
                logger.warning(f"industry file lock {name} failed")
        yield acquired
    finally:
        if acquired:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            except OSError:
                pass
        handle.close()


# -- 通用缓存 ------------------------------------------------------------------
def _read_json(name: str, ttl: float) -> Optional[Any]:
    path = CACHE_ROOT / name
    if not path.exists():
        return None
    if time.time() - path.stat().st_mtime > ttl:
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _read_json_stale(name: str, ttl: float) -> Tuple[Optional[Any], bool]:
    """读 JSON 缓存：过期也返回旧数据，并标记 stale（stale-while-revalidate）。"""
    path = CACHE_ROOT / name
    if not path.exists():
        return None, False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None, False
    stale = (time.time() - path.stat().st_mtime) > ttl
    return payload, stale


def _write_json(name: str, payload: Any) -> None:
    path = CACHE_ROOT / name
    tmp = path.with_suffix(".tmp")
    try:
        tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, path)
    except OSError as exc:
        logger.warning(f"industry cache write failed: {exc}")


def _read_csv(name: str, ttl: float) -> Optional[pd.DataFrame]:
    path = CACHE_ROOT / name
    if not path.exists() or time.time() - path.stat().st_mtime > ttl:
        return None
    try:
        return pd.read_csv(path)
    except (OSError, pd.errors.ParserError):
        return None


def _read_csv_stale(name: str, ttl: float) -> Tuple[Optional[pd.DataFrame], bool]:
    """读缓存：过期也返回旧数据，并标记 stale（stale-while-revalidate）。

    申万行业历史每个行业约 1.5MB / 5s，31 个行业全量刷新需要数十秒。
    宁可先给用户略旧的数据，也不要让页面干等——后台会异步刷新。
    """
    path = CACHE_ROOT / name
    if not path.exists():
        return None, False
    try:
        df = pd.read_csv(path)
    except (OSError, pd.errors.ParserError):
        return None, False
    if df is None or df.empty:
        return None, False
    stale = (time.time() - path.stat().st_mtime) > ttl
    return df, stale


def _write_csv(name: str, df: pd.DataFrame) -> None:
    path = CACHE_ROOT / name
    tmp = path.with_suffix(".tmp")
    try:
        df.to_csv(tmp, index=False)
        os.replace(tmp, path)
    except OSError as exc:
        logger.warning(f"industry csv write failed: {exc}")


def _mem_get(key: str, ttl: float = _MEM_TTL) -> Optional[Any]:
    item = _MEM.get(key)
    if item and time.time() - item[0] < ttl:
        return item[1]
    return None


def _invalidate_list_mem() -> None:
    _MEM.pop("industry:list", None)


def _mem_put(key: str, value: Any) -> Any:
    _MEM[key] = (time.time(), value)
    return value


# -- 行业清单与实时行情 --------------------------------------------------------
def _fetch_base_sync(refresh: bool = False) -> Dict[str, Any]:
    """同步拉取行业清单 + 实时行情（约 6s）。

    进程内 + 跨进程加锁并二次检查缓存：多个 worker/请求同时触发时只拉取
    一次，其余等待后直接复用刚写入的结果，避免重复访问上游。
    """
    with _BASE_FETCH_LOCK:
        with _cache_file_lock("base"):
            # force 时也容忍 10s 内的结果：另一 worker 刚刷新过就复用
            ttl = 10.0 if refresh else BASE_TTL_SECONDS
            fresh, stale = _read_json_stale("base.json", ttl)
            if fresh is not None and not stale:
                return fresh

            import akshare as ak

            info = ak.sw_index_first_info()
            try:
                rt = ak.index_realtime_sw(symbol="一级行业")
            except Exception as exc:  # 实时接口偶发失败，退化为仅有估值
                logger.warning(f"index_realtime_sw failed: {exc}")
                rt = None

            rt_map: Dict[str, Dict[str, Any]] = {}
            if rt is not None and not rt.empty:
                for _, r in rt.iterrows():
                    c = str(r.get("指数代码", "")).split(".")[0]
                    prev = float(r.get("昨收盘") or 0) or 0
                    last = float(r.get("最新价") or 0) or 0
                    rt_map[c] = {
                        "change_pct": round((last / prev - 1) * 100, 2) if prev else None,
                        "amount": float(r.get("成交额") or 0),
                        "price": last,
                    }
            else:
                # 实时接口失败时保留上一次的行情，避免页面瞬间丢失涨跌幅
                prev_base = _read_json("base.json", 86400 * 7) or {}
                rt_map = prev_base.get("realtime") or {}

            base = {
                "updated": datetime.now().strftime("%H:%M:%S"),
                "info": [
                    {"code": str(r.get("行业代码", "")).split(".")[0],
                     "name": str(r.get("行业名称", "")),
                     "count": int(r.get("成份个数") or 0),
                     "pe": _num(r.get("TTM(滚动)市盈率")),
                     "pb": _num(r.get("市净率")),
                     "div_yield": _num(r.get("静态股息率"))}
                    for _, r in info.iterrows()
                ],
                "realtime": rt_map,
            }
            _write_json("base.json", base)
            _invalidate_list_mem()
            return base


def _refresh_base_async() -> None:
    """后台刷新行业清单与实时行情，单飞（single-flight），不阻塞请求。"""
    if not _BASE_LOCK.acquire(blocking=False):
        return

    def _run():
        try:
            _fetch_base_sync()
        except Exception as exc:
            logger.warning(f"industry base refresh failed: {exc}")
        finally:
            _BASE_LOCK.release()

    threading.Thread(target=_run, daemon=True).start()


def _build_industry_rows(base: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[str]]:
    rows: List[Dict[str, Any]] = []
    stale_codes: List[str] = []
    for item in base.get("info", []):
        live = (base.get("realtime") or {}).get(item["code"], {})
        metric, stale_hist = _industry_metrics_from_cache(item["code"])
        metric = metric or {}
        if stale_hist:
            stale_codes.append(item["code"])
        rows.append({
            **item,
            "name": item["name"] or SW_NAME.get(item["code"], item["code"]),
            "change_pct": live.get("change_pct"),
            "amount": live.get("amount"),
            "price": live.get("price"),
            **metric,
        })
    return rows, stale_codes


async def get_industry_list(force: bool = False) -> Dict[str, Any]:
    """31 个一级行业：实时涨跌 + 估值 + 已缓存的动量指标。

    清单/实时行情缓存 5 分钟，过期后先用旧数据响应、后台异步刷新
    （stale-while-revalidate），避免首屏等待上游接口约 6s。动量指标从
    本地历史文件即时计算，并在后台线程执行，不阻塞事件循环。
    """
    if not force:
        cached = _mem_get("industry:list", LIST_MEM_TTL)
        if cached is not None:
            return cached

    base, stale_base = (None, False) if force else _read_json_stale(
        "base.json", BASE_TTL_SECONDS
    )
    if base is None:
        try:
            base = await asyncio.to_thread(_fetch_base_sync, force)
            stale_base = False
        except Exception as exc:
            logger.warning(f"sw_index_first_info failed: {exc}")
            base = _read_json("base.json", 86400 * 7)
            if not base:
                _refresh_base_async()
                return {"rows": [], "pending": False, "refreshing": True,
                        "error": str(exc)}
            stale_base = True
    elif stale_base:
        _refresh_base_async()

    rows, stale_codes = await asyncio.to_thread(_build_industry_rows, base)
    missing = [r["code"] for r in rows if r.get("r20") is None]
    # 缺失的（首次拉取）与过期的（stale）都交给后台刷新，接口立即返回
    refresh = list({*missing, *stale_codes})
    if refresh:
        _prewarm_histories(refresh)
    result = {
        "updated": base.get("updated") or datetime.now().strftime("%H:%M:%S"),
        "stale": bool(base.get("stale")) or stale_base or bool(stale_codes),
        "refreshing": bool(refresh) or stale_base,
        "pending": bool(missing),
        "progress": {"ready": len(rows) - len(missing), "total": len(rows)},
        "rows": rows,
        "quadrants": _quadrants(rows),
    }
    _mem_put("industry:list", result)
    return result


def warm_cache() -> None:
    """服务启动时后台预热：行业清单与缺失的历史数据。

    首屏不再触发上游接口；即便预热尚未完成，请求也会立即返回。
    """
    base, stale_base = _read_json_stale("base.json", BASE_TTL_SECONDS)
    if base is None or stale_base:
        _refresh_base_async()
    base = base or _read_json("base.json", 86400 * 7) or {}
    codes = [item["code"] for item in base.get("info", [])] or [c for c, _ in SW_LEVEL1]
    missing = []
    for code in codes:
        df = _read_csv(_hist_path(code), HIST_TTL_SECONDS)
        if df is None or len(df) < 25:
            missing.append(code)
    if missing:
        _prewarm_histories(missing)


def _num(v: Any) -> Optional[float]:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(f) else round(f, 2)


# -- 行业指数历史 --------------------------------------------------------------
def _hist_path(code: str) -> str:
    return f"hist_{code}.csv"


async def get_industry_history(code: str, days: int = 400) -> pd.DataFrame:
    """行业历史（stale-while-revalidate）：过期数据照常返回，后台刷新。

    单行业上游接口约 5s，过期同步拉取会让详情页久等，因此先给旧数据。
    """
    df, stale = _read_csv_stale(_hist_path(code), HIST_TTL_SECONDS)
    if df is None:
        try:
            df = await asyncio.wait_for(
                asyncio.to_thread(_fetch_history_sync, code), timeout=8.0,
            )
        except asyncio.TimeoutError:
            logger.warning(f"industry history fetch timed out for {code}")
            df = None
    elif stale:
        _prewarm_histories([code])
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    return df.tail(days).reset_index(drop=True)


def _fetch_history_sync(code: str, retries: int = 2) -> Optional[pd.DataFrame]:
    import akshare as ak
    for attempt in range(retries + 1):
        try:
            raw = ak.index_hist_sw(symbol=code, period="day")
            if raw is None or raw.empty:
                return None
            df = raw.rename(columns={"日期": "date", "收盘": "close", "开盘": "open",
                                     "最高": "high", "最低": "low", "成交量": "volume",
                                     "成交额": "amount"})
            keep = [c for c in ("date", "open", "high", "low", "close", "volume", "amount")
                    if c in df.columns]
            df = df[keep].tail(1000)
            _write_csv(_hist_path(code), df)
            return df
        except Exception as exc:
            logger.debug(f"index_hist_sw {code} attempt {attempt} failed: {exc}")
            time.sleep(1.5 + attempt)
    return None


def _industry_metrics_from_cache(code: str) -> Tuple[Optional[Dict[str, Any]], bool]:
    """读行业指标缓存，返回 (指标, 是否过期)。过期数据照常返回，由后台刷新。"""
    df, stale = _read_csv_stale(_hist_path(code), HIST_TTL_SECONDS)
    if df is None or len(df) < 25:
        return None, True
    return industry_metrics(df), stale


def _prewarm_histories(codes: List[str]) -> None:
    if not codes:
        return
    if not _PREWARM_LOCK.acquire(blocking=False):
        return
    _PREWARMING["hist"] = True

    def _run():
        try:
            # 跨进程单飞：多 worker 时只让一个进程刷新历史
            with _cache_file_lock("hist", blocking=False) as acquired:
                if not acquired:
                    return
                with ThreadPoolExecutor(max_workers=4) as pool:
                    list(pool.map(lambda c: _fetch_history_sync(c), codes))
        except Exception as exc:
            logger.warning(f"industry prewarm failed: {exc}")
        finally:
            _invalidate_list_mem()
            _PREWARMING["hist"] = False
            _PREWARM_LOCK.release()

    threading.Thread(target=_run, daemon=True).start()


# -- 指标计算 ------------------------------------------------------------------
def industry_metrics(df: pd.DataFrame) -> Dict[str, Any]:
    """行业指数的动量、波动、回撤与区间位置。"""
    d = df.copy()
    for col in ("close", "high", "low"):
        if col in d.columns:
            d[col] = pd.to_numeric(d[col], errors="coerce")
    d = d.dropna(subset=["close"])
    close = d["close"].astype(float)
    if len(close) < 2:
        return {}

    def _ret(n: int) -> Optional[float]:
        if len(close) <= n:
            return None
        base = float(close.iloc[-1 - n])
        return round((float(close.iloc[-1]) / base - 1) * 100, 2) if base else None

    hi = float(close.tail(250).max()) if len(close) >= 60 else float(close.max())
    lo = float(close.tail(250).min()) if len(close) >= 60 else float(close.min())
    pos = (float(close.iloc[-1]) - lo) / (hi - lo) * 100 if hi > lo else 50.0
    vol = ta.annualized_volatility(close.tail(60))

    return {
        "r5": _ret(5),
        "r20": _ret(20),
        "r60": _ret(60),
        "r120": _ret(120),
        "vol": round(vol, 2),
        "max_dd": round(ta.max_drawdown(close.tail(120)), 2),
        "pos250": round(pos, 1),
        "above_ma20": bool(len(close) >= 20 and float(close.iloc[-1]) >
                           float(ta.sma(close, 20).iloc[-1])),
        "above_ma60": bool(len(close) >= 60 and float(close.iloc[-1]) >
                           float(ta.sma(close, 60).iloc[-1])),
    }


def relative_to_bench(hist: pd.DataFrame, bench: pd.DataFrame) -> Dict[str, Any]:
    if hist is None or bench is None or hist.empty or bench.empty:
        return {}
    a = hist[["date", "close"]].rename(columns={"close": "sector"})
    b = bench[["date", "close"]].rename(columns={"close": "bench"})
    a["date"] = pd.to_datetime(a["date"]).dt.strftime("%Y-%m-%d")
    b["date"] = pd.to_datetime(b["date"]).dt.strftime("%Y-%m-%d")
    m = a.merge(b, on="date", how="inner").sort_values("date").tail(180)
    if len(m) < 25:
        return {}
    m = m.reset_index(drop=True)
    base_s, base_b = float(m["sector"].iloc[0]), float(m["bench"].iloc[0])
    m["rs"] = (m["sector"] / base_s) / (m["bench"] / base_b) * 100

    def _excess(n: int) -> Optional[float]:
        if len(m) <= n:
            return None
        s = float(m["sector"].iloc[-1]) / float(m["sector"].iloc[-1 - n]) - 1
        bch = float(m["bench"].iloc[-1]) / float(m["bench"].iloc[-1 - n]) - 1
        return round((s - bch) * 100, 2)

    return {
        "series": [{"date": r["date"], "rs": round(float(r["rs"]), 2)}
                   for _, r in m.iloc[::max(1, len(m) // 120)].iterrows()],
        "rs20": _excess(20),
        "rs60": _excess(60),
        "rs_slope": round(ta.slope(m["rs"], 10), 2),
    }


def _quadrants(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for r in rows:
        r20, r60 = r.get("r20"), r.get("r60")
        if r20 is None or r60 is None:
            continue
        if r20 >= 0 and r60 >= 0:
            q, label = 1, "双强·领涨"
        elif r20 >= 0 and r60 < 0:
            q, label = 2, "短强·反弹"
        elif r20 < 0 and r60 >= 0:
            q, label = 4, "短弱·回调"
        else:
            q, label = 3, "双弱·退潮"
        out.append({
            "code": r["code"], "name": r["name"], "x": r20, "y": r60,
            "quadrant": q, "quadrant_label": label,
            "size": r.get("amount") or 0, "change_pct": r.get("change_pct"),
        })
    return out


# -- 成分股与广度 --------------------------------------------------------------
async def get_constituents(code: str, limit: int = 60) -> List[Dict[str, Any]]:
    name = f"cons_{code}.json"
    cached = _read_json(name, 86400 * 7)
    if cached is None:
        def _load():
            import akshare as ak
            return ak.index_component_sw(symbol=code)
        try:
            df = await asyncio.to_thread(_load)
            cached = [
                {"code": str(r["证券代码"]).zfill(6), "name": str(r["证券名称"]),
                 "weight": _num(r.get("最新权重")) or 0.0}
                for _, r in df.iterrows()
            ]
            cached.sort(key=lambda x: x["weight"], reverse=True)
            _write_json(name, cached)
        except Exception as exc:
            logger.warning(f"index_component_sw {code} failed: {exc}")
            return []
    return cached[:limit]


def _schedule_breadth_refresh(code: str, limit: int) -> None:
    """在事件循环内后台重算广度，单行业单飞，不阻塞请求。"""
    running = _BREADTH_TASKS.get(code)
    if running is not None and not running.done():
        return
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return

    async def _run():
        try:
            # 跨进程单飞：同一行业只让一个 worker 重算
            with _cache_file_lock(f"breadth_{code}", blocking=False) as acquired:
                if acquired:
                    await _compute_breadth(code, limit)
        except Exception as exc:
            logger.warning(f"industry breadth refresh {code} failed: {exc}")

    task = loop.create_task(_run())
    _BREADTH_TASKS[code] = task

    def _cleanup(t: asyncio.Task) -> None:
        if _BREADTH_TASKS.get(code) is t:
            _BREADTH_TASKS.pop(code, None)

    task.add_done_callback(_cleanup)


async def get_industry_breadth(code: str, limit: int = BREADTH_SAMPLE) -> Dict[str, Any]:
    """成分股广度：站上 MA20/MA60 比例、20 日中位涨幅、今日涨跌家数。

    缓存过期时先返回旧数据，后台重算（stale-while-revalidate），避免详情
    页为 30 只成分股行情等待数秒。
    """
    cached, stale = _read_json_stale(f"breadth_{code}.json", BREADTH_TTL_SECONDS)
    if cached is not None:
        if stale:
            _schedule_breadth_refresh(code, limit)
        return cached
    return await _compute_breadth(code, limit)


async def _compute_breadth(code: str, limit: int) -> Dict[str, Any]:
    cache_name = f"breadth_{code}.json"
    cons = await get_constituents(code, limit=limit)
    if not cons:
        return {"available": False}

    from app.adapters import DataType
    from app.api.data import get_failover

    mgr = get_failover()
    sem = asyncio.Semaphore(6)
    end = datetime.now().strftime("%Y%m%d")
    start = (datetime.now() - timedelta(days=160)).strftime("%Y%m%d")

    async def _one(item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        async with sem:
            try:
                kl = await mgr.fetch_with_failover(
                    DataType.KLINE, item["code"], frequency="D",
                    start_date=start, end_date=end,
                )
            except Exception:
                return None
        if not kl or len(kl) < 25:
            return None
        closes = np.array([float(k.close) for k in kl], dtype=float)
        ma20 = float(np.mean(closes[-20:]))
        ma60 = float(np.mean(closes[-60:])) if len(closes) >= 60 else None
        r20 = (closes[-1] / closes[-21] - 1) * 100 if len(closes) > 21 else None
        chg = (closes[-1] / closes[-2] - 1) * 100 if len(closes) > 2 else None
        return {
            "code": item["code"], "name": item["name"], "weight": item["weight"],
            "price": round(float(closes[-1]), 2),
            "change_pct": round(chg, 2) if chg is not None else None,
            "r20": round(r20, 2) if r20 is not None else None,
            "above_ma20": bool(closes[-1] > ma20),
            "above_ma60": bool(closes[-1] > ma60) if ma60 else None,
        }

    results = [r for r in await asyncio.gather(*[_one(c) for c in cons]) if r]
    if not results:
        return {"available": False}

    above20 = sum(1 for r in results if r["above_ma20"])
    above60 = [r for r in results if r["above_ma60"] is not None]
    r20s = [r["r20"] for r in results if r["r20"] is not None]
    up = sum(1 for r in results if (r["change_pct"] or 0) > 0)
    payload = {
        "available": True,
        "sample": len(results),
        "above_ma20_pct": round(above20 / len(results) * 100, 1),
        "above_ma60_pct": round(sum(1 for r in above60 if r["above_ma60"]) /
                                len(above60) * 100, 1) if above60 else None,
        "median_r20": round(float(np.median(r20s)), 2) if r20s else None,
        "up_count": up,
        "down_count": len(results) - up,
        "constituents": sorted(results, key=lambda x: x["weight"], reverse=True),
    }
    _write_json(cache_name, payload)
    return payload


# -- 个股 → 行业映射 -----------------------------------------------------------
def _map_name(code: str) -> str:
    return f"cons_{code}.json"


def _fetch_map_rows_sync(code: str) -> List[Dict[str, Any]]:
    """读取/拉取单个行业的成分股（供映射构建复用）。"""
    rows = _read_json(_map_name(code), 86400 * 30)
    if rows is not None:
        return rows
    try:
        import akshare as ak
        df = ak.index_component_sw(symbol=code)
        rows = [{"code": str(r["证券代码"]).zfill(6),
                 "name": str(r["证券名称"]),
                 "weight": _num(r.get("最新权重")) or 0.0}
                for _, r in df.iterrows()]
        _write_json(_map_name(code), rows)
        return rows
    except Exception as exc:
        logger.debug(f"map build {code} failed: {exc}")
        return []


def _build_map_sync() -> Dict[str, str]:
    codes = [code for code, _ in SW_LEVEL1]
    mapping: Dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=4) as pool:
        for code, rows in zip(codes, pool.map(_fetch_map_rows_sync, codes)):
            for item in rows:
                item_code = item.get("code")
                if item_code:
                    mapping[item_code] = code
    if mapping:
        _write_json("stock_map.json", mapping)
    return mapping


async def _ensure_map() -> Dict[str, str]:
    """构建 股票代码 → 申万行业 映射（落盘 7 天，后台并发构建）。"""
    cached = _read_json("stock_map.json", 86400 * 7)
    if cached:
        return cached
    if _PREWARMING["map"]:
        return {}
    _PREWARMING["map"] = True

    def _build():
        try:
            # 跨进程单飞：只让一个 worker 构建，其余请求先走近似映射
            with _cache_file_lock("map", blocking=False) as acquired:
                if acquired:
                    _build_map_sync()
        except Exception as exc:
            logger.warning(f"industry stock map build failed: {exc}")
        finally:
            _PREWARMING["map"] = False

    threading.Thread(target=_build, daemon=True).start()
    return {}


async def resolve_stock_industry(symbol: str) -> Optional[Dict[str, Any]]:
    """优先使用精确的申万成分映射；未就绪时用证监会行业做近似。"""
    code = symbol.split(".")[0].zfill(6)
    mapping = await _ensure_map()
    sw_code = mapping.get(code) if mapping else None
    if sw_code:
        return {"code": sw_code, "name": SW_NAME.get(sw_code, ""), "source": "申万成分"}

    def _profile():
        import akshare as ak
        return ak.stock_profile_cninfo(symbol=code)

    try:
        df = await asyncio.wait_for(asyncio.to_thread(_profile), timeout=5.0)
        if df is None or df.empty:
            return None
        csrc = str(df.iloc[0].get("所属行业") or "")
        sw_name = CSRC_TO_SW.get(csrc)
        if not sw_name:
            for key, val in CSRC_TO_SW.items():
                if key and key in csrc:
                    sw_name = val
                    break
        if not sw_name:
            return None
        for c, n in SW_LEVEL1:
            if n == sw_name:
                return {"code": c, "name": n, "source": "证监会行业近似",
                        "csrc": csrc, "approx": True}
    except Exception as exc:
        logger.debug(f"stock_profile_cninfo {code} failed: {exc}")
    return None


async def industry_detail(code: str, bench: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
    name = SW_NAME.get(code, code)
    hist = await get_industry_history(code)
    metrics = industry_metrics(hist) if not hist.empty else {}
    rel = relative_to_bench(hist, bench) if (bench is not None and not hist.empty) else {}
    breadth = await get_industry_breadth(code)

    payload: Dict[str, Any] = {
        "code": code,
        "name": name,
        "metrics": metrics,
        "relative": rel,
        "breadth": breadth,
        "series": [
            {"date": r["date"], "close": round(float(r["close"]), 2)}
            for _, r in hist.tail(250).iterrows()
        ] if not hist.empty else [],
    }
    if metrics:
        base = _read_json("base.json", 86400 * 7) or {}
        row = next(
            (x for x in base.get("info", []) if x.get("code") == code), None
        )
        if row:
            payload["valuation"] = {
                "pe": row.get("pe"), "pb": row.get("pb"),
                "div_yield": row.get("div_yield"), "count": row.get("count"),
            }
    return payload


def rotation_view(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return _quadrants(rows)
