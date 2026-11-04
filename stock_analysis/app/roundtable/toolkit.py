# ============================================================================
# 投研圆桌 · 数据工具集（事实锚）
# ============================================================================
"""把 A 股真实数据整理成「可引用的数据快照」。

TradingAgents 给每个分析师配一组 tool，并强调「市场分析师必须把价格/指标
落在经过校验的数据快照上」。本项目已有成熟的取数链路（FailoverManager、
量化研判引擎 forecast、新闻多源抓取、东财资金流），因此这里不做 LLM
tool-calling 循环（网关只支持 chat completion），而是：

1. 并行拉取全部数据，超时/失败各自降级，不影响其它片段；
2. 生成「带数值的事实块」注入各角色提示词；
3. 任何拿不到的字段明确写「数据缺失」，从提示词层面禁止模型编造。
"""
from __future__ import annotations

import asyncio
import logging
from datetime import date, datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import pandas as pd

from app import ta
from app.adapters import DataType
from app.forecast import build_stock_forecast

logger = logging.getLogger(__name__)

_MISSING = "数据缺失（数据源未返回，禁止推测具体数值）"


def _num(v: Any, nd: int = 2) -> Optional[float]:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if f != f or f in (float("inf"), float("-inf")):
        return None
    return round(f, nd)


def to_df(klines: Any) -> pd.DataFrame:
    """KlineData 列表 / DataFrame → 标准升序 DataFrame（复用研判引擎口径）。"""
    from app.forecast import _to_df
    return _to_df(klines)


def _fmt_pct(v: Optional[float]) -> str:
    return "—" if v is None else f"{v:+.2f}%"


def _fmt_levels(items: Any, limit: int = 3) -> str:
    """关键位列表 → 「MA20 90.42 / 前低 88.78」式可读文本。"""
    if not isinstance(items, list) or not items:
        return "—"
    parts = []
    for it in items[:limit]:
        if isinstance(it, dict):
            label = it.get("label") or "位"
            price = it.get("price")
            dist = it.get("distance_pct")
            if price is None:
                continue
            parts.append(f"{label} {price}" + (f"({dist:+.2f}%)" if isinstance(dist, (int, float)) else ""))
        else:
            parts.append(str(it))
    return " / ".join(parts) if parts else "—"


class RoundtableToolkit:
    """一次圆桌运行共用的数据上下文。"""

    def __init__(self, failover, symbol: str, name: str,
                 trade_date: Optional[str] = None,
                 horizon_days: int = 10):
        self.failover = failover
        self.symbol = symbol
        self.name = name or symbol
        self.code = "".join(ch for ch in symbol if ch.isdigit())[:6] or symbol
        self.trade_date = trade_date or date.today().isoformat()
        self.horizon = horizon_days

        self.df: pd.DataFrame = pd.DataFrame()
        self.bench_df: pd.DataFrame = pd.DataFrame()
        self.fundamental: Dict[str, Any] = {}
        self.news: List[Dict[str, Any]] = []
        self.capital: Dict[str, Any] = {}
        self.forecast: Dict[str, Any] = {}
        self.errors: List[str] = []

    # -- 拉取 ---------------------------------------------------------------
    async def prepare(self) -> Dict[str, Any]:
        """并行拉取行情/基准/基本面/新闻/资金流，返回可入库的 snapshot。"""
        results = await asyncio.gather(
            self._load_klines(),
            self._load_bench(),
            self._load_fundamental(),
            self._load_news(),
            self._load_capital(),
            return_exceptions=True,
        )
        for result in results:
            if isinstance(result, Exception):
                self.errors.append(f"数据加载异常: {result}")
        self._build_forecast()
        return self.snapshot()

    async def _load_klines(self) -> None:
        # 拉 2.5 年日线：技术指标（MA250/条件概率）需要足够样本，默认 90 天不够用
        from datetime import timedelta
        try:
            end = self.trade_date.replace("-", "")
            start = (date.fromisoformat(self.trade_date) - timedelta(days=900)).strftime("%Y%m%d")
            raw = await asyncio.wait_for(
                self.failover.fetch_with_failover(
                    DataType.KLINE, self.symbol,
                    start_date=start, end_date=end,
                ),
                timeout=25.0,
            )
            df = to_df(raw)
            if self.trade_date:
                df = df[df["date"] <= self.trade_date].reset_index(drop=True)
            self.df = df
        except Exception as exc:
            logger.warning(f"[roundtable] kline failed {self.symbol}: {exc}")
            self.errors.append(f"K线数据获取失败: {exc}")

    async def _load_bench(self) -> None:
        """沪深300 作为相对强弱的基准（与个股研判同源）。"""

        def _load():
            import akshare as ak
            return ak.stock_zh_index_daily(symbol="sh000300")

        try:
            raw = await asyncio.wait_for(asyncio.to_thread(_load), timeout=10.0)
            df = to_df(raw)
            if self.trade_date:
                df = df[df["date"] <= self.trade_date].reset_index(drop=True)
            self.bench_df = df
        except Exception as exc:
            logger.warning(f"[roundtable] bench failed: {exc}")
            self.errors.append("基准指数数据缺失，相对强弱无法计算")

    async def _load_fundamental(self) -> None:
        today = datetime.now(timezone(timedelta(hours=8))).date().isoformat()
        if self.trade_date < today:
            self.errors.append("历史时点基本面不可用，已排除实时估值与财务快照")
            return
        try:
            raw = await asyncio.wait_for(
                self.failover.fetch_with_failover(DataType.FUNDAMENTAL, self.symbol),
                timeout=15.0,
            )
            if raw is None:
                return
            self.fundamental = {
                "pe_ratio": _num(getattr(raw, "pe_ratio", None)),
                "pb_ratio": _num(getattr(raw, "pb_ratio", None)),
                "market_cap": _num(getattr(raw, "market_cap", None)),
                "circulating_cap": _num(getattr(raw, "circulating_cap", None)),
                "roe": _num(getattr(raw, "roe", None)),
                "industry": getattr(raw, "industry", "") or "",
            }
        except Exception as exc:
            logger.warning(f"[roundtable] fundamental failed {self.symbol}: {exc}")
            self.errors.append(f"基本面数据获取失败: {exc}")

    async def _load_news(self) -> None:
        try:
            from app.news_sources import fetch_stock_news
            cutoff = datetime.combine(date.fromisoformat(self.trade_date) + timedelta(days=1),
                                      datetime.min.time()) - timedelta(hours=8)
            items = await asyncio.wait_for(
                fetch_stock_news([self.code], per_symbol=12), timeout=15.0,
            )
            self.news = [
                {
                    "title": (getattr(n, "title", "") or "")[:200],
                    "source": getattr(n, "source", ""),
                    "published_at": (
                        (getattr(n, "published_at") + timedelta(hours=8)).strftime("%Y-%m-%d %H:%M")
                        if getattr(n, "published_at", None) else ""
                    ),
                    "summary": (getattr(n, "content", "") or "")[:180],
                }
                for n in (items or [])
                # 时点一致：回看历史日期时不引入"未来新闻"
                if getattr(n, "published_at", None) is not None
                and getattr(n, "published_at") < cutoff
            ][:12]
        except Exception as exc:
            logger.warning(f"[roundtable] news failed {self.symbol}: {exc}")
            self.errors.append(f"个股新闻获取失败: {exc}")

    async def _load_capital(self) -> None:
        """东财个股资金流（best-effort，接口不稳定时直接标记缺失）。"""

        def _load():
            import akshare as ak
            market = "sh" if self.code.startswith(("6", "9")) else (
                "bj" if self.code.startswith(("4", "8")) else "sz")
            return ak.stock_individual_fund_flow(stock=self.code, market=market)

        try:
            df = await asyncio.wait_for(asyncio.to_thread(_load), timeout=10.0)
            if df is None or df.empty:
                return
            df = df[pd.to_datetime(df["日期"]).dt.strftime("%Y-%m-%d") <= self.trade_date].tail(5)
            if df.empty:
                self.errors.append("分析基准日前无可用资金流数据")
                return
            rows: List[Dict[str, Any]] = []
            for _, r in df.iterrows():
                rec = r.to_dict()
                rows.append({
                    "date": str(rec.get("日期", "")),
                    "change_pct": _num(rec.get("涨跌幅")),
                    "main_net": _num(rec.get("主力净流入-净额")),
                    "main_net_pct": _num(rec.get("主力净流入-净占比")),
                })
            total = sum(x["main_net"] or 0 for x in rows)
            inflow_days = sum(1 for x in rows if (x["main_net"] or 0) > 0)
            self.capital = {
                "rows": rows,
                "main_net_total_5d": round(total, 2),
                "inflow_days": inflow_days,
            }
        except Exception as exc:
            logger.info(f"[roundtable] capital flow unavailable {self.symbol}: {exc}")
            self.errors.append(f"资金流数据获取失败: {exc}")

    def _build_forecast(self) -> None:
        if self.df.empty or len(self.df) < 30:
            self.errors.append("K线不足 30 根，量化研判不可用")
            return
        try:
            self.forecast = build_stock_forecast(
                self.df, self.bench_df if not self.bench_df.empty else None,
                horizon=self.horizon,
            )
        except Exception as exc:
            logger.warning(f"[roundtable] forecast failed: {exc}")
            self.errors.append(f"量化研判失败: {exc}")

    # -- 快照 ---------------------------------------------------------------
    def snapshot(self) -> Dict[str, Any]:
        fc = self.forecast or {}
        snap = self.snapshot_core()
        snap.update({
            "errors": list(self.errors),
            "fundamental": self.fundamental,
            "news_count": len(self.news),
            "capital": self.capital,
            "forecast": {
                "available": bool(fc.get("available")),
                "technical_score": (fc.get("technical") or {}).get("score"),
                "direction": (fc.get("technical") or {}).get("direction"),
                "up_probability": (fc.get("probability") or {}).get("probability"),
                "levels": fc.get("levels"),
                "signals": fc.get("signals"),
                "signal_summary": fc.get("signal_summary"),
                "multi_timeframe": fc.get("multi_timeframe"),
                "market": fc.get("market"),
                "relative": fc.get("relative"),
            },
        })
        return snap

    def snapshot_core(self) -> Dict[str, Any]:
        """仅行情/指标的轻量快照（记忆结算也复用）。"""
        if self.df.empty:
            return {"bars": 0, "price": None, "trade_date": self.trade_date}
        s = ta.snapshot(self.df)
        return {
            "bars": int(len(self.df)),
            "price": _num(s.get("price"), 3),
            "change_pct": _num(s.get("change_pct")),
            "returns": s.get("returns"),
            "volatility": s.get("volatility"),
            "atr_pct": s.get("atr_pct"),
            "ma": s.get("ma"),
            "high_250": _num(s.get("high_250"), 3),
            "low_250": _num(s.get("low_250"), 3),
            "pos_250": s.get("pos_250"),
            "trade_date": self.df["date"].iloc[-1],
        }

    # -- 提示词片段 ---------------------------------------------------------
    def market_brief(self) -> str:
        fc = self.forecast or {}
        if not fc.get("available"):
            return f"### 行情\n{_MISSING}\n"
        s = fc["technical"]["snapshot"]
        rets = s.get("returns") or {}
        prob = fc.get("probability") or {}
        cond = fc.get("conditional") or {}
        levels = fc.get("levels") or {}
        market = fc.get("market") or {}
        rel = fc.get("relative") or {}
        mtf = fc.get("multi_timeframe") or {}
        return (
            f"### 行情快照（截至 {self.trade_date}，共 {s.get('bars')} 根日线）\n"
            f"- 最新收盘：{s.get('price')} 元，当日 {_fmt_pct(s.get('change_pct'))}\n"
            f"- 区间收益：5日 {_fmt_pct(rets.get('r5'))} / 20日 {_fmt_pct(rets.get('r20'))} "
            f"/ 60日 {_fmt_pct(rets.get('r60'))} / 250日 {_fmt_pct(rets.get('r250'))}\n"
            f"- 年化波动率 {s.get('volatility')}%，ATR占价比 {s.get('atr_pct')}%，"
            f"近120日最大回撤 {s.get('max_drawdown_120')}%\n"
            f"- 250日区间 {s.get('low_250')} ~ {s.get('high_250')}，当前分位 {s.get('pos_250')}%\n"
            f"- 量化技术评分 {fc['technical']['score']}（{fc['technical']['direction']}），"
            f"多空信号 {fc.get('signal_summary', {}).get('bull')}多 / {fc.get('signal_summary', {}).get('bear')}空\n"
            f"- 历史条件概率：未来 {fc.get('horizon')} 日上涨概率 "
            f"{prob.get('probability') if prob.get('probability') is not None else '—'}%"
            f"（条件样本 {cond.get('samples', '—')} 次，无条件基准 {cond.get('base_rate', '—')}%，"
            f"置信 {prob.get('confidence', '—')}）\n"
            f"- 关键位：支撑 {_fmt_levels(levels.get('supports'))} / "
            f"阻力 {_fmt_levels(levels.get('resistances'))}；"
            f"建议止损 {levels.get('suggested_stop', '—')}（{_fmt_pct(levels.get('stop_pct'))}）\n"
            f"- 多周期：{mtf.get('resonance', '—')}\n"
            f"- 市场环境：沪深300 {market.get('state', '—')}"
            f"（20日 {_fmt_pct(market.get('ret20'))}，风险偏好 {market.get('bias', '—')}）\n"
            f"- 相对强弱：20日超额 {_fmt_pct(rel.get('excess20'))}，"
            f"60日超额 {_fmt_pct(rel.get('excess60'))}\n"
        )

    def technical_brief(self) -> str:
        fc = self.forecast or {}
        if not fc.get("available"):
            return f"### 技术指标\n{_MISSING}\n"
        s = fc["technical"]["snapshot"]
        ma = s.get("ma") or {}
        macd = s.get("macd") or {}
        kdj = s.get("kdj") or {}
        boll = s.get("boll") or {}
        adx = s.get("adx") or {}
        comps = fc["technical"].get("components") or {}
        comp_lines = "\n".join(
            f"  - {k}: {v.get('score')}（{v.get('note', '')}）" for k, v in comps.items()
        )
        signals = fc.get("signals") or []
        sig_lines = "\n".join(
            f"  - [{x.get('direction')}] {x.get('label')}：{x.get('detail')}" for x in signals[:10]
        )
        return (
            "### 技术指标（同源计算，直接引用）\n"
            f"- 均线：MA5 {ma.get('ma5')} / MA10 {ma.get('ma10')} / MA20 {ma.get('ma20')} "
            f"/ MA60 {ma.get('ma60')} / MA120 {ma.get('ma120')}\n"
            f"- 均线斜率：MA20 {s.get('ma20_slope')}，MA60 {s.get('ma60_slope')}\n"
            f"- MACD：DIF {macd.get('dif')} / DEA {macd.get('dea')} / 柱 {macd.get('hist')}\n"
            f"- RSI6 {s.get('rsi6')} / RSI14 {s.get('rsi14')}；KDJ：K {kdj.get('k')} D {kdj.get('d')} J {kdj.get('j')}\n"
            f"- BOLL：上轨 {boll.get('up')} / 中轨 {boll.get('mid')} / 下轨 {boll.get('low')}\n"
            f"- ADX {adx.get('adx')}（+DI {adx.get('pdi')} / -DI {adx.get('mdi')}）；ATR {s.get('atr')}\n"
            f"- 量能：量比 {s.get('volume_ratio')}，5日均量 {s.get('volume_ma5')}，OBV斜率 {s.get('obv_slope')}\n"
            f"- 分项打分：\n{comp_lines or '  - 无'}\n"
            f"- 触发信号：\n{sig_lines or '  - 无'}\n"
        )

    def fundamentals_brief(self) -> str:
        f = self.fundamental or {}
        if not any(f.values()):
            return f"### 基本面\n{_MISSING}\n"
        return (
            "### 基本面快照\n"
            f"- 所属行业：{f.get('industry') or '—'}\n"
            f"- PE {f.get('pe_ratio') if f.get('pe_ratio') is not None else '—'}，"
            f"PB {f.get('pb_ratio') if f.get('pb_ratio') is not None else '—'}\n"
            f"- 总市值 {f.get('market_cap') if f.get('market_cap') is not None else '—'} 亿元，"
            f"流通市值 {f.get('circulating_cap') if f.get('circulating_cap') is not None else '—'} 亿元\n"
            f"- ROE {f.get('roe') if f.get('roe') is not None else '—'}%\n"
            f"（注：仅当数据源给出上述字段时引用，未见字段视为未知）\n"
        )

    def news_brief(self) -> str:
        if not self.news:
            return f"### 新闻\n{_MISSING}\n"
        lines = [
            f"- [{n['published_at'] or '时间未知'}] {n['title']}"
            + (f"（{n['summary'][:80]}…）" if n.get("summary") else "")
            for n in self.news[:10]
        ]
        return "### 近期个股新闻（东方财富，近 72 小时）\n" + "\n".join(lines) + "\n"

    def capital_brief(self) -> str:
        c = self.capital or {}
        if not c.get("rows"):
            return f"### 资金流\n{_MISSING}\n"
        lines = [
            f"- {r['date']}：涨跌 {_fmt_pct(r.get('change_pct'))}，"
            f"主力净额 {(r.get('main_net') or 0) / 1e8:+.2f} 亿，净占比 {r.get('main_net_pct')}%"
            for r in c["rows"]
        ]
        return (
            "### 主力资金（东方财富，近5日）\n" + "\n".join(lines) +
            f"\n- 5日主力净流入合计 {(c.get('main_net_total_5d') or 0) / 1e8:+.2f} 亿元，"
            f"净流入天数 {c.get('inflow_days')}/5\n"
        )

    def grounding_text(self) -> str:
        """完整事实锚，用于最终校验/落库说明。"""
        return "\n".join([
            self.market_brief(), self.technical_brief(),
            self.fundamentals_brief(), self.news_brief(), self.capital_brief(),
        ])

    def price(self) -> Optional[float]:
        p = (self.snapshot_core() or {}).get("price")
        return float(p) if p else None
