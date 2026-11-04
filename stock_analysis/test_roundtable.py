# ============================================================================
# Tests - 投研圆桌（TradingAgents 风格多智能体）
# ============================================================================
"""纯逻辑测试：不触网、不依赖数据库。

覆盖三块最容易被改坏的地方：
1. ``should_continue_debate`` / ``should_continue_risk`` 的条件路由（与
   TradingAgents ``ConditionalLogic`` 的判定条件一致）；
2. 图执行顺序与事件流（分析师 → 多空 N 轮 → 裁决 → 交易 → 三方风控 → 终审）；
3. 模型全挂时的降级路径（必须有可读兜底结论，而不是空结果）。
"""
from __future__ import annotations

import json
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import pytest

from app.adapters import DataType, FundamentalData, KlineData
from app.model_gateway import LLMResponse
from app.roundtable.agents import PortfolioManager
from app.roundtable.graph import (
    NODE_AGGRESSIVE, NODE_ANALYSTS, NODE_BEAR, NODE_BULL, NODE_CONSERVATIVE,
    NODE_NEUTRAL, NODE_PM, NODE_RESEARCH_MANAGER, NODE_TRADER, NODE_END,
    RoundtableGraph, should_continue_debate, should_continue_risk,
)
from app.roundtable.state import RoundtableConfig, new_state
from app.roundtable.toolkit import RoundtableToolkit


# -- Fixtures -----------------------------------------------------------------
def make_klines(n: int = 260, drift: float = 0.003, seed: int = 5) -> List[KlineData]:
    rng = np.random.default_rng(seed)
    rets = rng.normal(drift, 0.013, n)
    close = 50.0 * np.cumprod(1 + rets)
    dates = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=n)
    out: List[KlineData] = []
    for i, (d, c) in enumerate(zip(dates, close)):
        out.append(KlineData(
            symbol="600519", date=d.strftime("%Y-%m-%d"),
            open=float(c * 0.995), high=float(c * 1.012),
            low=float(c * 0.988), close=float(c),
            volume=int(1_000_000 + i * 1000), amount=float(c) * 1e6,
        ))
    return out


class FakeFailover:
    def __init__(self, klines=None):
        self.klines = klines if klines is not None else make_klines()

    async def fetch_with_failover(self, data_type, symbol, **kwargs):
        if data_type == DataType.KLINE:
            return self.klines
        if data_type == DataType.FUNDAMENTAL:
            return FundamentalData(
                symbol=symbol, name="测试股", pe_ratio=22.5, pb_ratio=3.1,
                market_cap=18000.0, circulating_cap=15000.0,
                industry="白酒", roe=28.4,
            )
        raise RuntimeError(f"unsupported {data_type}")


class FakeGateway:
    """按提示词特征返回不同内容，模拟各家模型。"""

    def __init__(self, fail_all: bool = False, json_garbage: bool = False):
        self.fail_all = fail_all
        # 模拟推理型模型：把思维链写进 content，并在 max_tokens 处被截断
        self.json_garbage = json_garbage
        self.calls: List[str] = []

    async def chat(self, messages, prefer_config_id=None, temperature=None,
                   max_tokens=None, stream=False, config_id=None,
                   response_format=None):
        user = messages[-1]["content"]
        self.calls.append(user[:40])
        if self.fail_all:
            return LLMResponse(content="", model="fake", success=False, error="all down")
        if self.json_garbage:
            return LLMResponse(
                content="The user wants me to act as the Portfolio Manager and output JSON.\n"
                        "I need to synthesize:\n1. Research Manager's ruling…",
                model="nemotron-test", success=True, finish_reason="length",
            )
        if "请只输出一个 JSON 对象" in user:
            return LLMResponse(
                content=json.dumps({
                    "action": "buy", "confidence": 0.66, "position_pct": 25,
                    "entry_price": 52.3, "target_price": 61.0, "stop_loss": 47.8,
                    "horizon_days": 10, "rationale": "趋势与资金共振",
                    "key_risks": ["放量滞涨"], "memory_used": ["上次高点追入被套"],
                    "lessons": "分批建仓优于一次性",
                }, ensure_ascii=False),
                model="fake", success=True, finish_reason="stop",
            )
        return LLMResponse(content="## 结论\n看多，置信度中。", model="fake",
                           success=True, finish_reason="stop")


@pytest.fixture()
def toolkit(monkeypatch):
    """构造已加载数据、但不触网的 toolkit。"""

    async def _noop_bench(self):
        self.bench_df = pd.DataFrame()

    async def _noop_news(self):
        self.news = [{"title": "测试新闻", "source": "东财",
                      "published_at": "2026-10-27 10:00", "summary": "摘要"}]

    async def _noop_capital(self):
        self.capital = {
            "rows": [{"date": "2026-10-27", "change_pct": 1.2,
                      "main_net": 1.5e8, "main_net_pct": 3.1}],
            "main_net_total_5d": 1.5e8, "inflow_days": 1,
        }

    monkeypatch.setattr(RoundtableToolkit, "_load_bench", _noop_bench)
    monkeypatch.setattr(RoundtableToolkit, "_load_news", _noop_news)
    monkeypatch.setattr(RoundtableToolkit, "_load_capital", _noop_capital)

    tk = RoundtableToolkit(FakeFailover(), "600519", "贵州茅台",
                           trade_date=pd.Timestamp.today().strftime("%Y-%m-%d"))
    return tk


# -- 条件路由 -----------------------------------------------------------------
def _state_with_debate(count: int, current: str) -> Dict[str, Any]:
    st = new_state("600519", "贵州茅台", "2026-10-28", RoundtableConfig())
    st["investment_debate_state"].update({"count": count, "current_response": current})
    return st


def test_debate_routes_to_bull_when_empty():
    st = _state_with_debate(0, "")
    assert should_continue_debate(st, 2) == NODE_BULL


def test_debate_routes_to_bear_after_bull_speaks():
    st = _state_with_debate(1, "Bull Researcher: 看多理由")
    assert should_continue_debate(st, 2) == NODE_BEAR


def test_debate_routes_to_bear_then_back_to_bull():
    st = _state_with_debate(2, "Bear Researcher: 风险")
    assert should_continue_debate(st, 2) == NODE_BULL


def test_debate_stops_at_two_rounds():
    # 2 轮 = 多空各 2 次 = count 4
    st = _state_with_debate(4, "Bear Researcher: 风险")
    assert should_continue_debate(st, 2) == NODE_RESEARCH_MANAGER
    st1 = _state_with_debate(2, "Bear Researcher: 风险")
    assert should_continue_debate(st1, 1) == NODE_RESEARCH_MANAGER


def test_risk_rotation_order():
    st = new_state("600519", "贵州茅台", "2026-10-28", RoundtableConfig())
    assert should_continue_risk(st, 1) == NODE_AGGRESSIVE
    st["risk_debate_state"].update({"count": 1, "latest_speaker": NODE_AGGRESSIVE})
    assert should_continue_risk(st, 1) == NODE_CONSERVATIVE
    st["risk_debate_state"].update({"count": 2, "latest_speaker": NODE_CONSERVATIVE})
    assert should_continue_risk(st, 1) == NODE_NEUTRAL
    st["risk_debate_state"].update({"count": 3, "latest_speaker": NODE_NEUTRAL})
    assert should_continue_risk(st, 1) == NODE_PM


def test_risk_stops_after_two_rounds():
    st = new_state("600519", "贵州茅台", "2026-10-28", RoundtableConfig())
    st["risk_debate_state"].update({"count": 6, "latest_speaker": NODE_NEUTRAL})
    assert should_continue_risk(st, 2) == NODE_PM


# -- 数据事实锚 ---------------------------------------------------------------
@pytest.mark.asyncio
async def test_snapshot_has_grounded_numbers(toolkit):
    snap = await toolkit.prepare()
    assert snap["bars"] >= 200
    assert snap["price"] and snap["price"] > 0
    assert snap["forecast"]["available"] is True
    assert isinstance(snap["forecast"]["technical_score"], (int, float))
    assert snap["news_count"] == 1
    json.dumps(snap, ensure_ascii=False)  # 必须可序列化


@pytest.mark.asyncio
async def test_prompt_blocks_quote_snapshot(toolkit):
    await toolkit.prepare()
    price = toolkit.price()
    brief = toolkit.market_brief()
    assert f"{price}" in brief or "收盘" in brief
    assert "None" not in brief.replace("None 亿元", "")
    assert "数据缺失" in toolkit.fundamentals_brief() or "PE" in toolkit.fundamentals_brief()


@pytest.mark.asyncio
async def test_empty_klines_degrades_without_crash(monkeypatch):
    async def _noop(self):
        return None

    monkeypatch.setattr(RoundtableToolkit, "_load_bench", _noop)
    monkeypatch.setattr(RoundtableToolkit, "_load_news", _noop)
    monkeypatch.setattr(RoundtableToolkit, "_load_capital", _noop)

    tk = RoundtableToolkit(FakeFailover(klines=[]), "600519", "贵州茅台")
    snap = await tk.prepare()
    assert snap["bars"] == 0
    assert "数据缺失" in tk.market_brief()
    assert tk.price() is None


# -- 图执行 -------------------------------------------------------------------
async def _collect(graph, **kwargs):
    events = []
    async for kind, payload in graph.run_stream(symbol="600519", name="贵州茅台",
                                                trade_date="2026-10-28", **kwargs):
        events.append((kind, payload))
    return events


def _node_order(events) -> List[str]:
    order = []
    for kind, payload in events:
        if kind == "phase" and payload.get("status") == "running":
            node = payload.get("node")
            if not order or order[-1] != node:
                order.append(node)
    return order


@pytest.mark.asyncio
async def test_graph_runs_full_roundtable(toolkit, monkeypatch):
    original = toolkit.prepare

    async def _prepared(self):
        return await original()

    monkeypatch.setattr(RoundtableToolkit, "prepare", _prepared)

    gateway = FakeGateway()
    settled = {"called": 0}

    async def _settle():
        settled["called"] += 1
        return {"checked": 2, "settled": 1, "skipped": 1, "failed": 0}

    graph = RoundtableGraph(gateway, FakeFailover(),
                            RoundtableConfig(max_debate_rounds=2, max_risk_rounds=1))
    events = await _collect(graph, settle=_settle)

    order = _node_order(events)
    assert order == [
        NODE_ANALYSTS, NODE_BULL, NODE_BEAR, NODE_BULL, NODE_BEAR,
        NODE_RESEARCH_MANAGER, NODE_TRADER,
        NODE_AGGRESSIVE, NODE_CONSERVATIVE, NODE_NEUTRAL, NODE_PM,
    ]
    assert settled["called"] == 1

    result = [p for k, p in events if k == "result"][0]
    assert result["decision"]["action"] == "buy"
    assert result["decision"]["confidence"] == 0.66
    assert result["decision"]["entry_price"] == 52.3
    assert len(result["transcript"]) == 4 + 4 + 1 + 1 + 3 + 1
    assert result["errors"] == []
    json.dumps(result, ensure_ascii=False, default=str)

    # 发言顺序：4 位分析师 → 多空交替 → 裁决 → 交易 → 三方风控 → 终审
    roles = [t["role"] for t in result["transcript"]]
    assert roles[:4] == ["market_analyst", "fundamentals_analyst",
                         "news_analyst", "sentiment_analyst"]
    assert roles[4:8] == ["bull_researcher", "bear_researcher"] * 2
    assert roles[8] == "research_manager"
    assert roles[9] == "trader"
    assert roles[10:13] == ["aggressive_analyst", "conservative_analyst", "neutral_analyst"]
    assert roles[13] == "portfolio_manager"


@pytest.mark.asyncio
async def test_graph_single_round_config(toolkit, monkeypatch):
    original = toolkit.prepare

    async def _prepared(self):
        return await original()

    monkeypatch.setattr(RoundtableToolkit, "prepare", _prepared)

    graph = RoundtableGraph(FakeGateway(), FakeFailover(),
                            RoundtableConfig(max_debate_rounds=1, max_risk_rounds=1))
    events = await _collect(graph)
    result = [p for k, p in events if k == "result"][0]
    roles = [t["role"] for t in result["transcript"]]
    assert roles.count("bull_researcher") == 1
    assert roles.count("bear_researcher") == 1


@pytest.mark.asyncio
async def test_graph_degrades_when_models_down(toolkit, monkeypatch):
    original = toolkit.prepare

    async def _prepared(self):
        return await original()

    monkeypatch.setattr(RoundtableToolkit, "prepare", _prepared)

    graph = RoundtableGraph(FakeGateway(fail_all=True), FakeFailover(),
                            RoundtableConfig(max_debate_rounds=1, max_risk_rounds=1))
    events = await _collect(graph)
    result = [p for k, p in events if k == "result"][0]

    assert result["errors"], "模型全挂必须记录降级原因"
    assert result["decision"]["action"] in ("buy", "hold", "sell")
    assert result["decision"].get("degraded") is True
    assert result["status"] == "completed_with_errors"
    # 每个角色都要有可读占位，不能是空发言
    assert all(t["content"].strip() for t in result["transcript"])


@pytest.mark.asyncio
async def test_graph_reports_format_error_separately_from_outage(toolkit, monkeypatch):
    """模型有响应但输出不是 JSON：报「格式不合规」，不能报「模型不可用」。"""
    original = toolkit.prepare

    async def _prepared(self):
        return await original()

    monkeypatch.setattr(RoundtableToolkit, "prepare", _prepared)

    graph = RoundtableGraph(FakeGateway(json_garbage=True), FakeFailover(),
                            RoundtableConfig(max_debate_rounds=1, max_risk_rounds=1))
    events = await _collect(graph)
    result = [p for k, p in events if k == "result"][0]

    joined = " ".join(result["errors"])
    assert "格式不合规" in joined
    assert "模型不可用" not in joined
    assert result["decision"]["degraded"] is True
    assert result["status"] == "completed_with_errors"


# -- 决策归一化 ---------------------------------------------------------------
def test_portfolio_manager_normalizes_bad_fields(toolkit):
    pm = PortfolioManager(FakeGateway(), RoundtableConfig())
    out = pm._normalize({
        "action": "BUY", "confidence": 3, "position_pct": -5,
        "entry_price": "0", "target_price": "abc", "stop_loss": None,
        "key_risks": "单条风险", "memory_used": None,
    }, toolkit)
    assert out["action"] == "buy"
    assert out["confidence"] == 1.0          # clamp 到 [0,1]
    assert out["position_pct"] == 0.0        # clamp 到 [0,100]
    assert out["target_price"] is None
    assert out["key_risks"] == ["单条风险"]
    assert out["memory_used"] == []


def test_portfolio_manager_unknown_action_becomes_hold(toolkit):
    pm = PortfolioManager(FakeGateway(), RoundtableConfig())
    out = pm._normalize({"action": "加仓"}, toolkit)
    assert out["action"] == "hold"


def test_fallback_decision_reflects_technical_score(toolkit):
    """看多兜底：目标取上方阻力，止损取下方支撑，且必须方向自洽。"""
    pm = PortfolioManager(FakeGateway(), RoundtableConfig())
    toolkit.snapshot_core = lambda: {"price": 100.0}   # type: ignore[method-assign]
    toolkit.forecast = {
        "available": True,
        "technical": {"score": 45.0, "direction": "看多", "snapshot": {}, "components": {}},
        "probability": {"probability": 62.0},
        "levels": {"supports": [94.0], "resistances": [108.0], "atr": 2.0},
    }
    out = pm.fallback_decision(toolkit, "timeout")
    assert out["action"] == "buy"
    assert out["target_price"] == 108.0
    assert out["stop_loss"] == 94.0
    assert out["target_price"] > out["entry_price"] > out["stop_loss"]
    assert out["degraded"] is True
    assert "62" in out["rationale"]


def test_fallback_decision_accepts_level_dicts(toolkit):
    """研判引擎的关键位是 {label, price, distance_pct}，不能把 dict 当价格。"""
    pm = PortfolioManager(FakeGateway(), RoundtableConfig())
    toolkit.snapshot_core = lambda: {"price": 100.0}   # type: ignore[method-assign]
    toolkit.forecast = {
        "available": True,
        "technical": {"score": -30.0, "direction": "看空", "snapshot": {}, "components": {}},
        "probability": {"probability": None},
        "levels": {
            "supports": [{"label": "前低", "price": 94.0, "distance_pct": -6.0}],
            "resistances": [{"label": "MA20", "price": 106.0, "distance_pct": 6.0}],
            "atr": 2.0,
        },
    }
    out = pm.fallback_decision(toolkit, "no model")
    assert out["action"] == "sell"
    # 卖出方向：目标是下方支撑（跌到哪），止损是上方阻力（反弹到哪算看错）
    assert out["target_price"] == 94.0
    assert out["stop_loss"] == 106.0
    assert out["target_price"] < out["entry_price"] < out["stop_loss"]
    assert out["target_note"] == "前低" and out["stop_note"] == "MA20"


# -- 回归：000524 那次「目标 8.39 贴脸、盈亏比 0.24:1」的兜底缺陷 ---------------
def _case_000524() -> Dict[str, Any]:
    """复刻真实数据：现价 8.45、ATR 0.268、支撑 8.39/8.24/8.11、阻力 8.7。"""
    return {
        "available": True,
        "technical": {"score": -20.7, "direction": "震荡", "snapshot": {}, "components": {}},
        "probability": {"probability": 44.7},
        "levels": {
            "price": 8.45, "atr": 0.268, "suggested_stop": 8.18, "stop_pct": -3.25,
            "supports": [
                {"label": "布林下轨", "price": 8.39, "distance_pct": -0.71},
                {"label": "前低", "price": 8.24, "distance_pct": -2.49},
                {"label": "前低", "price": 8.11, "distance_pct": -4.02},
            ],
            "resistances": [
                {"label": "MA20", "price": 8.7, "distance_pct": 2.96},
                {"label": "布林上轨", "price": 9.01, "distance_pct": 6.63},
            ],
        },
    }


def test_fallback_ignores_stop_level_too_close_to_price(toolkit):
    """止损贴着现价会把风险算小、盈亏比算高（比目标贴脸更危险），必须改用它处。"""
    pm = PortfolioManager(FakeGateway(), RoundtableConfig())
    toolkit.snapshot_core = lambda: {"price": 100.0}   # type: ignore[method-assign]
    toolkit.forecast = {
        "available": True,
        "technical": {"score": 30.0, "direction": "看多", "snapshot": {}, "components": {}},
        "probability": {"probability": 60.0},
        "levels": {
            "supports": [{"label": "布林下轨", "price": 100.05, "distance_pct": -0.05}],
            "resistances": [{"label": "MA20", "price": 112.0, "distance_pct": 12.0}],
            "atr": 2.0, "suggested_stop": 96.0,
        },
    }
    out = pm.fallback_decision(toolkit, "降级")
    assert out["stop_loss"] == 96.0, "0.05% 的止损等于贴着现价，应改用 suggested_stop"
    assert out["stop_note"] == "建议止损"
    assert out["risk_reward"] == 3.0                # 而不是被贴脸止损吹成 240:1


def test_fallback_sell_skips_support_too_close_to_price(toolkit):
    """SELL 目标不能取「贴脸」的布林下轨（-0.71%），要退到距离 ≥ 1×ATR 的支撑。"""
    pm = PortfolioManager(FakeGateway(), RoundtableConfig())
    toolkit.forecast = _case_000524()
    toolkit.snapshot_core = lambda: {"price": 8.45}   # type: ignore[method-assign]

    out = pm.fallback_decision(toolkit, "模型输出不是合法 JSON")
    assert out["action"] == "sell"
    assert out["entry_price"] == 8.45
    assert out["target_price"] == 8.11, "应跳过 8.39 / 8.24，目标取 -4.02% 的支撑"
    assert out["stop_loss"] == 8.7
    assert out["risk_reward"] >= PortfolioManager.MIN_RR
    assert out["target_price"] < out["entry_price"] < out["stop_loss"]


def test_fallback_sell_flags_poor_risk_reward(toolkit):
    """只有很近的支撑时仍给结论，但必须把「盈亏比不划算」写进风险提示。"""
    pm = PortfolioManager(FakeGateway(), RoundtableConfig())
    case = _case_000524()
    case["levels"]["supports"] = case["levels"]["supports"][:1]   # 只剩 8.39
    toolkit.forecast = case
    toolkit.snapshot_core = lambda: {"price": 8.45}   # type: ignore[method-assign]

    out = pm.fallback_decision(toolkit, "模型输出不是合法 JSON")
    assert out["target_price"] == 8.39
    assert out["risk_reward"] < PortfolioManager.MIN_RR
    assert any("盈亏比" in r for r in out["key_risks"])


def test_fallback_rationale_does_not_blame_model_outage(toolkit):
    """格式问题不能写成「语言模型不可用」，否则会把人引向错误的排查方向。"""
    pm = PortfolioManager(FakeGateway(), RoundtableConfig())
    toolkit.forecast = _case_000524()
    toolkit.snapshot_core = lambda: {"price": 8.45}   # type: ignore[method-assign]

    out = pm.fallback_decision(toolkit, "模型 nemotron 未返回合法 JSON（finish_reason=length）")
    assert "决策链路降级" in out["rationale"]
    assert "语言模型不可用" not in out["rationale"]


@pytest.mark.asyncio
async def test_chat_json_raises_format_error_when_truncated():
    """思维链被截断 → 抛 LLMFormatError（不是 LLMUnavailable），并带上模型名。"""
    from app.roundtable.agents import LLMFormatError, LLMUnavailable

    pm = PortfolioManager(FakeGateway(json_garbage=True), RoundtableConfig())
    with pytest.raises(LLMFormatError) as exc:
        await pm.chat_json("你是组合经理", "请只输出一个 JSON 对象")
    msg = str(exc.value)
    assert "nemotron-test" in msg and "截断" in msg
    assert not isinstance(exc.value, LLMUnavailable)


def test_decision_summary_uses_direction_aware_labels():
    """SELL 的展示必须写「下跌目标 / 上方止损」，中性「目标」会被读成算反了。"""
    sell = RoundtableGraph._render_decision({
        "action": "sell", "confidence": 0.4, "position_pct": 0.0,
        "entry_price": 8.45, "target_price": 8.11, "stop_loss": 8.7,
        "risk_reward": 1.36, "rationale": "量化规则", "key_risks": [],
    })
    assert "下跌目标 8.11" in sell and "上方止损 8.7" in sell
    assert "盈亏比 1.36:1" in sell

    buy = RoundtableGraph._render_decision({
        "action": "buy", "confidence": 0.7, "position_pct": 20.0,
        "entry_price": 100.0, "target_price": 112.0, "stop_loss": 95.0,
        "risk_reward": 2.4, "rationale": "趋势", "key_risks": [],
    })
    assert "上涨目标 112.0" in buy and "止损 95.0" in buy


def test_normalize_drops_direction_inconsistent_prices(toolkit):
    """模型把 SELL 的目标价给到现价之上时必须丢弃，不能落库成错误数字。"""
    pm = PortfolioManager(FakeGateway(), RoundtableConfig())
    toolkit.snapshot_core = lambda: {"price": 8.45}   # type: ignore[method-assign]

    out = pm._normalize({
        "action": "sell", "confidence": 0.6, "position_pct": 0,
        "entry_price": 8.45, "target_price": 8.6, "stop_loss": 8.2,
    }, toolkit)
    assert out["action"] == "sell"
    assert out["target_price"] is None and out["stop_loss"] is None
    assert out["price_notes"], "丢弃价格时必须留下说明，便于回溯"


# -- 记忆结算 ----------------------------------------------------------------
def _series(closes: List[float], start: str = "2026-01-05") -> pd.DataFrame:
    dates = pd.bdate_range(start, periods=len(closes)).strftime("%Y-%m-%d")
    return pd.DataFrame({"date": dates, "close": closes})


def test_settlement_waits_for_full_holding_period():
    from app.roundtable.memory import compute_settlement
    df = _series([10, 11, 12, 13, 14])
    assert compute_settlement(df, pd.DataFrame(), df["date"].iloc[2], 5) is None


def test_settlement_counts_trading_bars_and_alpha():
    from app.roundtable.memory import compute_settlement
    stock = _series([10, 10, 10, 11, 12, 13, 14, 15])
    bench = _series([100, 100, 100, 100, 101, 102, 103, 104])
    td = stock["date"].iloc[2]              # 基准日收盘 10
    out = compute_settlement(stock, bench, td, 3)
    assert out is not None
    assert out["entry_price"] == 10.0
    assert out["exit_price"] == 13.0        # 第 3 个交易日
    assert out["realized_return_pct"] == 30.0
    assert out["benchmark_return_pct"] == 2.0   # 100 → 102
    assert out["alpha_pct"] == 28.0


def test_settlement_uses_row_entry_price_when_given():
    from app.roundtable.memory import compute_settlement
    stock = _series([10, 10, 10, 11, 12, 13])
    td = stock["date"].iloc[2]
    out = compute_settlement(stock, pd.DataFrame(), td, 2, entry_price=11.0)
    assert out["entry_price"] == 11.0
    assert out["realized_return_pct"] == round((12 / 11 - 1) * 100, 2)
    assert out["alpha_pct"] is None          # 没有基准时不给超额
