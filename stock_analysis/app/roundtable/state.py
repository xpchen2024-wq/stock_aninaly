# ============================================================================
# 投研圆桌 · 图状态定义
# ============================================================================
"""TradingAgents 风格的状态容器。

TradingAgents 用 LangGraph 的 ``AgentState`` 承载整条图的状态：分析师报告、
多空辩论史、交易计划、风控辩论史、终审决策、历史记忆。本项目不引入
LangGraph（与现有 FastAPI + 自研 ModelGateway 架构不兼容），因此用纯
Python 的 dict + dataclass 复刻同一套状态语义，并额外记录 transcript
（逐条发言），供前端回放圆桌全过程。
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import date, datetime, timezone
from typing import Any, Dict, List, Optional


# -- 阶段 / 角色常量 ----------------------------------------------------------
PHASE_ANALYSTS = "analysts"
PHASE_RESEARCH = "research_debate"
PHASE_RESEARCH_JUDGE = "research_judge"
PHASE_TRADER = "trader"
PHASE_RISK = "risk_debate"
PHASE_DECISION = "decision"
PHASE_MEMORY = "memory"

PHASE_LABELS = {
    PHASE_ANALYSTS: "分析师团队",
    PHASE_RESEARCH: "多空辩论",
    PHASE_RESEARCH_JUDGE: "研究经理裁决",
    PHASE_TRADER: "交易计划",
    PHASE_RISK: "三方风控辩论",
    PHASE_DECISION: "组合经理终审",
    PHASE_MEMORY: "记忆结算与反思",
}

# 角色标识 → 展示信息（前端配色/头像用）
ROLE_META: Dict[str, Dict[str, str]] = {
    "market_analyst":      {"name": "市场技术分析师", "icon": "📈", "side": "analyst"},
    "fundamentals_analyst": {"name": "基本面分析师", "icon": "📊", "side": "analyst"},
    "news_analyst":        {"name": "新闻分析师",   "icon": "📰", "side": "analyst"},
    "sentiment_analyst":   {"name": "情绪分析师",   "icon": "🌡️", "side": "analyst"},
    "bull_researcher":     {"name": "看涨研究员",   "icon": "🐂", "side": "bull"},
    "bear_researcher":     {"name": "看跌研究员",   "icon": "🐻", "side": "bear"},
    "research_manager":    {"name": "研究经理",     "icon": "⚖️", "side": "judge"},
    "trader":              {"name": "交易员",       "icon": "💼", "side": "trader"},
    "aggressive_analyst":  {"name": "激进风控",     "icon": "🔥", "side": "aggressive"},
    "conservative_analyst": {"name": "保守风控",    "icon": "🛡️", "side": "conservative"},
    "neutral_analyst":     {"name": "中性风控",     "icon": "⚖️", "side": "neutral"},
    "portfolio_manager":   {"name": "组合经理",     "icon": "🏛️", "side": "manager"},
    "reflector":           {"name": "反思复盘",     "icon": "🪞", "side": "reflector"},
}


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


@dataclass
class RoundtableConfig:
    """一次圆桌的配置，对齐 TradingAgents 的 ``max_debate_rounds`` /
    ``max_risk_discuss_rounds`` / ``deep_think_llm`` / ``quick_think_llm``。"""
    max_debate_rounds: int = 2     # 多空各自发言轮数（每轮=多方1次+空方1次）
    max_risk_rounds: int = 1       # 风控三方各自发言轮数
    horizon_days: int = 10         # 记忆结算持有期（交易日）
    deep_model_id: Optional[str] = None    # 深思考档：裁决/交易/终审
    quick_model_id: Optional[str] = None   # 快思考档：分析师/辩论
    portfolio_note: str = ""       # 可选：当前持仓上下文

    def clamp(self) -> "RoundtableConfig":
        self.max_debate_rounds = max(1, min(int(self.max_debate_rounds or 2), 3))
        self.max_risk_rounds = max(1, min(int(self.max_risk_rounds or 1), 2))
        self.horizon_days = max(3, min(int(self.horizon_days or 10), 20))
        return self

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TranscriptEntry:
    """一条发言记录（分析师报告 / 辩论发言 / 裁决 / 决策）。"""
    seq: int
    role: str
    phase: str
    kind: str            # report / argument / rebuttal / judge / plan / decision / note
    content: str
    round: int = 0
    ts: str = field(default_factory=now_iso)

    def to_dict(self) -> Dict[str, Any]:
        meta = ROLE_META.get(self.role, {})
        return {
            "seq": self.seq,
            "role": self.role,
            "agent": meta.get("name", self.role),
            "icon": meta.get("icon", "🤖"),
            "side": meta.get("side", "analyst"),
            "phase": self.phase,
            "phase_label": PHASE_LABELS.get(self.phase, self.phase),
            "kind": self.kind,
            "content": self.content,
            "round": self.round,
            "ts": self.ts,
        }


def new_state(symbol: str, name: str, trade_date: str,
              config: RoundtableConfig) -> Dict[str, Any]:
    """初始化图状态，字段命名沿用 TradingAgents 的 AgentState。"""
    return {
        "company_of_interest": symbol,
        "company_name": name,
        "trade_date": trade_date,
        "config": config.to_dict(),

        # 事实锚（确定性数据，禁止 LLM 自行编造数字）
        "snapshot": {},
        "past_memory_str": "",

        # 分析师报告
        "market_report": "",
        "fundamentals_report": "",
        "news_report": "",
        "sentiment_report": "",

        # 多空辩论
        "investment_debate_state": {
            "history": [],
            "bull_history": [],
            "bear_history": [],
            "current_response": "",
            "judge_decision": "",
            "count": 0,
        },

        # 交易计划
        "trader_investment_plan": "",

        # 三方风控辩论
        "risk_debate_state": {
            "history": [],
            "aggressive_history": [],
            "conservative_history": [],
            "neutral_history": [],
            "latest_speaker": "",
            "judge_decision": "",
            "count": 0,
        },

        # 终审
        "final_trade_decision": {},

        # 过程记录
        "transcript": [],
        "errors": [],
        "checkpoints": [],
        "node": "start",
    }


def append_transcript(state: Dict[str, Any], role: str, phase: str,
                      content: str, kind: str = "report",
                      round_no: int = 0) -> Dict[str, Any]:
    """追加一条发言并返回其可序列化字典（供 SSE 立即推送）。"""
    entry = TranscriptEntry(
        seq=len(state["transcript"]) + 1,
        role=role, phase=phase, kind=kind,
        content=content, round=round_no,
    )
    payload = entry.to_dict()
    state["transcript"].append(payload)
    return payload


def add_error(state: Dict[str, Any], message: str) -> None:
    state.setdefault("errors", []).append(message)


def summarize_reports(state: Dict[str, Any], limit: int = 2200) -> str:
    """把四份分析师报告压成给辩论/交易使用的上下文。"""
    blocks = []
    for key, label in (
        ("market_report", "市场技术"),
        ("fundamentals_report", "基本面"),
        ("news_report", "新闻"),
        ("sentiment_report", "情绪"),
    ):
        text = (state.get(key) or "").strip()
        if text:
            blocks.append(f"### {label}分析师\n{text[:limit]}")
    return "\n\n".join(blocks) if blocks else "（暂无分析师报告）"


def debate_transcript(state: Dict[str, Any], side: str = "investment",
                      limit: int = 1600) -> str:
    """辩论史文本。side=investment 取多空，side=risk 取三方。"""
    key = "investment_debate_state" if side == "investment" else "risk_debate_state"
    history = state.get(key, {}).get("history") or []
    if not history:
        return "（辩论尚未开始）"
    lines = []
    for item in history[-12:]:
        lines.append(f"[{item.get('agent', '?')}] {str(item.get('content', ''))[:limit]}")
    return "\n\n".join(lines)
