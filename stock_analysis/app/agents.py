# ============================================================================
# AI Stock Analysis Platform - Multi-Agent Collaboration System
# 8 Agents + 5-Phase Workflow
# ============================================================================
from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime
from typing import Optional, Dict, Any, List
from uuid import uuid4

import numpy as np

from app.model_gateway import ModelGateway, LLMResponse
from app.schemas import (
    AgentMessage, MessageFactory,
    FundamentalReportPayload, SentimentReportPayload,
    NewsImpactPayload, TechnicalSignalPayload,
    DebateConclusionPayload, TradeProposalPayload,
    RiskAssessmentPayload, ApprovalDecisionPayload,
)
from app.adapters import FailoverManager, DataType
from app.indicators import compute_all_indicators

logger = logging.getLogger(__name__)

# -- Agent IDs ----------------------------------------------------------------
AGENT_FUNDAMENTAL = "AGENT-ANL-01"
AGENT_SENTIMENT  = "AGENT-ANL-02"
AGENT_NEWS       = "AGENT-ANL-03"
AGENT_TECHNICAL  = "AGENT-ANL-04"
AGENT_BULL       = "AGENT-RES-01"
AGENT_BEAR       = "AGENT-RES-02"
AGENT_TRADER     = "AGENT-TRD-01"
AGENT_RISK       = "AGENT-RSK-01"
AGENT_PM         = "AGENT-PM-01"


# -- Utility ------------------------------------------------------------------
def _extract_json(text: str) -> str:
    """Extract JSON from LLM response (may be wrapped in markdown)."""
    text = text.strip()
    if "```json" in text:
        start = text.find("```json") + 7
        end = text.find("```", start)
        return text[start:end].strip() if end > start else text
    if "```" in text:
        start = text.find("```") + 3
        end = text.find("```", start)
        return text[start:end].strip() if end > start else text
    if text.startswith("{") and text.endswith("}"):
        return text
    # Try to find JSON block
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        return text[start:end + 1]
    return text


# -- Base Agent ---------------------------------------------------------------
class BaseAgent:
    def __init__(self, agent_id: str, name: str, role: str, gateway: ModelGateway):
        self.agent_id = agent_id
        self.name = name
        self.role = role
        self.gateway = gateway

    async def _call_llm(self, system_prompt: str, user_prompt: str,
                        max_tokens: int = 1500) -> LLMResponse:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        return await self.gateway.chat(messages=messages, max_tokens=max_tokens)


# -- Phase 1: Analyst Agents (4 agents, parallel) -----------------------------
class FundamentalAnalyst(BaseAgent):
    """AGENT-ANL-01: Evaluates company financials and performance."""

    def __init__(self, gateway: ModelGateway):
        super().__init__(AGENT_FUNDAMENTAL, "基本面分析师", "fundamental", gateway)

    async def analyze(self, symbol: str, name: str,
                      fundamentals: Optional[Dict] = None) -> FundamentalReportPayload:
        system = "你是专业的基本面分析师。请基于财务数据进行评估，输出严格JSON格式。"
        user = f"""请对 {name}({symbol}) 进行基本面分析。
财务数据：{json.dumps(fundamentals or {}, ensure_ascii=False)}
输出JSON：{{"scores":{{"pe":0.0,"pb":0.0,"roe":0.0,"debt":0.0,"overall":0.0}},"red_flags":[],"intrinsic_value_range":{{"low":0,"high":0}},"confidence":0.0}}"""

        resp = await self._call_llm(system, user)
        try:
            data = json.loads(_extract_json(resp.content))
            return FundamentalReportPayload(
                symbol=symbol, name=name,
                scores=data.get("scores", {}),
                red_flags=data.get("red_flags", []),
                intrinsic_value_range=data.get("intrinsic_value_range", {}),
                confidence=data.get("confidence", 0.5),
            )
        except (json.JSONDecodeError, KeyError):
            return FundamentalReportPayload(
                symbol=symbol, name=name,
                scores={"overall": 0.5}, red_flags=[],
                intrinsic_value_range={"low": 0, "high": 0}, confidence=0.3,
            )


class SentimentAnalyst(BaseAgent):
    """AGENT-ANL-02: Aggregates market sentiment."""

    def __init__(self, gateway: ModelGateway):
        super().__init__(AGENT_SENTIMENT, "情绪分析师", "sentiment", gateway)

    async def analyze(self, symbol: str, name: str,
                      news_headlines: List[str] = None) -> SentimentReportPayload:
        headlines_text = "\n".join(f"- {h}" for h in (news_headlines or ["无数据"]))
        system = "你是市场情绪分析专家。请评估市场情绪，输出JSON格式。"
        user = f"""请对 {name}({symbol}) 进行情绪分析。
相关新闻标题：
{headlines_text}
输出JSON：{{"sentiment_score":0.0,"sentiment_trend":"stable","key_drivers":[],"confidence":0.0}}
sentiment_score范围：-1(极度悲观)到1(极度乐观)"""

        resp = await self._call_llm(system, user)
        try:
            data = json.loads(_extract_json(resp.content))
            return SentimentReportPayload(
                symbol=symbol, name=name,
                sentiment_score=data.get("sentiment_score", 0),
                sentiment_trend=data.get("sentiment_trend", "stable"),
                key_drivers=data.get("key_drivers", []),
                confidence=data.get("confidence", 0.5),
            )
        except (json.JSONDecodeError, KeyError):
            return SentimentReportPayload(
                symbol=symbol, name=name,
                sentiment_score=0, sentiment_trend="stable",
                key_drivers=[], confidence=0.3,
            )


class NewsAnalyst(BaseAgent):
    """AGENT-ANL-03: Monitors global news and macro events."""

    def __init__(self, gateway: ModelGateway):
        super().__init__(AGENT_NEWS, "新闻分析师", "news", gateway)

    async def analyze(self, symbol: str, name: str,
                      news_items: List[Dict] = None) -> NewsImpactPayload:
        news_text = "\n".join(
            f"- [{n.get('source','')}] {n.get('title','')}" for n in (news_items or [])
        ) or "无最新新闻"
        system = "你是宏观新闻分析专家。请评估新闻事件对股票的影响，输出JSON格式。"
        user = f"""请分析以下新闻对 {name}({symbol}) 的影响：
{news_text}
输出JSON：{{"event_summary":"","impact_score":0.0,"macro_score":0.0,"key_events":[]}}
impact_score: -1(重大利空)到1(重大利好)"""

        resp = await self._call_llm(system, user)
        try:
            data = json.loads(_extract_json(resp.content))
            return NewsImpactPayload(
                symbol=symbol, name=name,
                event_summary=data.get("event_summary", ""),
                impact_score=data.get("impact_score", 0),
                macro_score=data.get("macro_score", 0),
                key_events=data.get("key_events", []),
            )
        except (json.JSONDecodeError, KeyError):
            return NewsImpactPayload(
                symbol=symbol, name=name,
                event_summary="", impact_score=0, macro_score=0, key_events=[],
            )


class TechnicalAnalyst(BaseAgent):
    """AGENT-ANL-04: Detects trading patterns and technical signals."""

    def __init__(self, gateway: ModelGateway, failover: FailoverManager):
        super().__init__(AGENT_TECHNICAL, "技术分析师", "technical", gateway)
        self.failover = failover

    async def analyze(self, symbol: str, name: str) -> TechnicalSignalPayload:
        # Fetch K-line data and compute indicators
        closes = np.array([])
        try:
            klines = await self.failover.fetch_with_failover(DataType.KLINE, symbol)
            closes = np.array([k.close for k in klines[-60:]])
            highs = np.array([k.high for k in klines[-60:]])
            lows = np.array([k.low for k in klines[-60:]])
            volumes = np.array([k.volume for k in klines[-60:]])
            opens = np.array([k.open for k in klines[-60:]])
            indicators = compute_all_indicators(highs, lows, closes, volumes, opens)
        except Exception:
            indicators = {}

        system = "你是专业技术分析师。请分析技术形态和信号，输出JSON格式。"
        user = f"""请对 {name}({symbol}) 进行技术分析。
技术指标：{json.dumps(indicators, ensure_ascii=False, default=str)}
最近收盘价：{float(closes[-1]) if len(closes) > 0 else 'N/A'}
输出JSON：{{"trend_direction":"bullish/bearish/sideways","support_levels":[],"resistance_levels":[],"patterns":[],"confidence":0.0}}"""

        resp = await self._call_llm(system, user)
        try:
            data = json.loads(_extract_json(resp.content))
            return TechnicalSignalPayload(
                symbol=symbol, name=name,
                trend_direction=data.get("trend_direction", "sideways"),
                support_levels=data.get("support_levels", []),
                resistance_levels=data.get("resistance_levels", []),
                patterns=data.get("patterns", []),
                indicators=indicators,
                confidence=data.get("confidence", 0.5),
            )
        except (json.JSONDecodeError, KeyError):
            return TechnicalSignalPayload(
                symbol=symbol, name=name,
                trend_direction="sideways", support_levels=[],
                resistance_levels=[], patterns=[], confidence=0.3,
            )


# -- Phase 2: Research Debate Agents -----------------------------------------
class BullishResearcher(BaseAgent):
    """AGENT-RES-01: Evaluates from positive perspective."""

    def __init__(self, gateway: ModelGateway):
        super().__init__(AGENT_BULL, "看涨研究员", "bullish", gateway)

    async def debate(self, symbol: str, name: str,
                     reports: Dict[str, Any]) -> List[str]:
        system = "你是乐观派投资研究员。请从积极面评估投资机会，寻找上涨催化剂。"
        user = f"""请从看涨角度评估 {name}({symbol})：
分析师报告：{json.dumps(reports, ensure_ascii=False, default=str)[:3000]}
请列出3-5个看涨论点。"""

        resp = await self._call_llm(system, user, max_tokens=800)
        lines = [l.strip("- ").strip() for l in resp.content.split("\n")
                 if l.strip().startswith("-") or l.strip().startswith("*")]
        return lines[:5] if lines else [resp.content[:200]]


class BearishResearcher(BaseAgent):
    """AGENT-RES-02: Evaluates from risk perspective."""

    def __init__(self, gateway: ModelGateway):
        super().__init__(AGENT_BEAR, "看跌研究员", "bearish", gateway)

    async def debate(self, symbol: str, name: str,
                     reports: Dict[str, Any]) -> List[str]:
        system = "你是谨慎派投资研究员。请从风险面寻找逻辑漏洞，提出质疑。"
        user = f"""请从看跌/风险角度审查 {name}({symbol})：
分析师报告：{json.dumps(reports, ensure_ascii=False, default=str)[:3000]}
请列出3-5个风险点和看跌理由。"""

        resp = await self._call_llm(system, user, max_tokens=800)
        lines = [l.strip("- ").strip() for l in resp.content.split("\n")
                 if l.strip().startswith("-") or l.strip().startswith("*")]
        return lines[:5] if lines else [resp.content[:200]]


# -- Phase 3: Trading Agent --------------------------------------------------
class TradingAgent(BaseAgent):
    """AGENT-TRD-01: Generates trade proposals."""

    def __init__(self, gateway: ModelGateway):
        super().__init__(AGENT_TRADER, "交易代理", "trader", gateway)

    async def generate_proposal(self, symbol: str, name: str,
                                 debate: Dict) -> TradeProposalPayload:
        system = "你是专业交易员。请综合研究结论生成交易提案，输出JSON格式。"
        user = f"""请为 {name}({symbol}) 生成交易提案：
辩论结论：{json.dumps(debate, ensure_ascii=False, default=str)[:2000]}
输出JSON：{{"side":"buy/sell/hold","quantity":0,"stop_loss":0.0,"take_profit":0.0,"rationale":"","confidence":0.0}}"""

        resp = await self._call_llm(system, user)
        try:
            data = json.loads(_extract_json(resp.content))
            return TradeProposalPayload(
                symbol=symbol, name=name,
                side=data.get("side", "hold"),
                quantity=data.get("quantity", 0),
                stop_loss=data.get("stop_loss", 0),
                take_profit=data.get("take_profit", 0),
                rationale=data.get("rationale", ""),
                confidence=data.get("confidence", 0.5),
            )
        except (json.JSONDecodeError, KeyError):
            return TradeProposalPayload(
                symbol=symbol, name=name, side="hold",
                quantity=0, stop_loss=0, take_profit=0,
                rationale="无法生成交易提案", confidence=0.1,
            )


# -- Phase 4: Risk & Approval Agents -----------------------------------------
class RiskManager(BaseAgent):
    """AGENT-RSK-01: Assesses portfolio risk."""

    def __init__(self, gateway: ModelGateway):
        super().__init__(AGENT_RISK, "风险管理团队", "risk", gateway)

    async def assess(self, symbol: str, name: str,
                     proposal: TradeProposalPayload) -> RiskAssessmentPayload:
        system = "你是风险管理专家。请评估交易提案的风险，输出JSON格式。"
        user = f"""请评估以下交易提案的风险：
{name}({symbol}) - {proposal.side} {proposal.quantity}股
止损：{proposal.stop_loss} 止盈：{proposal.take_profit}
输出JSON：{{"var_value":0.0,"cvar_value":0.0,"concentration_risk":0.0,"stress_test_results":{{}},"risk_level":"low/medium/high/critical","recommendations":[]}}"""

        resp = await self._call_llm(system, user)
        try:
            data = json.loads(_extract_json(resp.content))
            return RiskAssessmentPayload(
                symbol=symbol,
                var_value=data.get("var_value", 0),
                cvar_value=data.get("cvar_value", 0),
                concentration_risk=data.get("concentration_risk", 0),
                stress_test_results=data.get("stress_test_results", {}),
                risk_level=data.get("risk_level", "medium"),
                recommendations=data.get("recommendations", []),
            )
        except (json.JSONDecodeError, KeyError):
            return RiskAssessmentPayload(
                symbol=symbol, var_value=0, cvar_value=0,
                concentration_risk=0, risk_level="medium", recommendations=[],
            )


class PortfolioManager(BaseAgent):
    """AGENT-PM-01: Final approval/rejection authority."""

    def __init__(self, gateway: ModelGateway):
        super().__init__(AGENT_PM, "投资组合经理", "portfolio_manager", gateway)

    async def approve(self, symbol: str, name: str,
                      proposal: TradeProposalPayload,
                      risk: RiskAssessmentPayload) -> ApprovalDecisionPayload:
        system = "你是投资组合经理，拥有最终审批权。请审核交易提案并做出决策，输出JSON格式。"
        user = f"""请审核以下交易：
{name}({symbol}) - {proposal.side} {proposal.quantity}股
风险等级：{risk.risk_level}
VaR：{risk.var_value} CVaR：{risk.cvar_value}
输出JSON：{{"status":"approved/rejected","allocated_amount":0.0,"rationale":"","conditions":[]}}"""

        resp = await self._call_llm(system, user)
        try:
            data = json.loads(_extract_json(resp.content))
            return ApprovalDecisionPayload(
                symbol=symbol,
                status=data.get("status", "rejected"),
                allocated_amount=data.get("allocated_amount", 0),
                rationale=data.get("rationale", ""),
                conditions=data.get("conditions", []),
            )
        except (json.JSONDecodeError, KeyError):
            return ApprovalDecisionPayload(
                symbol=symbol, status="rejected",
                allocated_amount=0, rationale="审批失败", conditions=[],
            )


# -- 5-Phase Agent Workflow Orchestrator -------------------------------------
class AgentWorkflow:
    """Orchestrates the complete 5-phase multi-agent decision pipeline."""

    def __init__(self, gateway: ModelGateway, failover: FailoverManager):
        self.gateway = gateway
        self.failover = failover

        # Initialize all agents
        self.fundamental_analyst = FundamentalAnalyst(gateway)
        self.sentiment_analyst = SentimentAnalyst(gateway)
        self.news_analyst = NewsAnalyst(gateway)
        self.technical_analyst = TechnicalAnalyst(gateway, failover)
        self.bull_researcher = BullishResearcher(gateway)
        self.bear_researcher = BearishResearcher(gateway)
        self.trading_agent = TradingAgent(gateway)
        self.risk_manager = RiskManager(gateway)
        self.portfolio_manager = PortfolioManager(gateway)

    async def run(self, symbol: str, name: str = "",
                  fundamentals: Optional[Dict] = None,
                  news_items: Optional[List[Dict]] = None) -> Dict[str, Any]:
        """
        Execute the complete 5-phase agent workflow.
        Returns full decision record.
        """
        session_id = str(uuid4())
        correlation_id = str(uuid4())
        errors = []

        logger.info(f"Starting agent workflow: session={session_id}, symbol={symbol}")

        # ===== Phase 1: Parallel Analyst Analysis =====
        logger.info(f"Phase 1: Analyst analysis for {symbol}")
        phase1_results = await asyncio.gather(
            self.fundamental_analyst.analyze(symbol, name, fundamentals),
            self.sentiment_analyst.analyze(symbol, name),
            self.news_analyst.analyze(symbol, name, news_items),
            self.technical_analyst.analyze(symbol, name),
            return_exceptions=True,
        )

        fundamental_rpt = phase1_results[0] if not isinstance(phase1_results[0], Exception) else None
        sentiment_rpt = phase1_results[1] if not isinstance(phase1_results[1], Exception) else None
        news_rpt = phase1_results[2] if not isinstance(phase1_results[2], Exception) else None
        technical_rpt = phase1_results[3] if not isinstance(phase1_results[3], Exception) else None

        for i, r in enumerate(phase1_results):
            if isinstance(r, Exception):
                errors.append(f"Agent {i+1} failed: {str(r)}")

        # ===== Phase 2: Research Debate =====
        logger.info(f"Phase 2: Research debate for {symbol}")
        reports_dict = {
            "fundamental": fundamental_rpt.model_dump() if fundamental_rpt else {},
            "sentiment": sentiment_rpt.model_dump() if sentiment_rpt else {},
            "news": news_rpt.model_dump() if news_rpt else {},
            "technical": technical_rpt.model_dump() if technical_rpt else {},
        }

        debate_tasks = await asyncio.gather(
            self.bull_researcher.debate(symbol, name, reports_dict),
            self.bear_researcher.debate(symbol, name, reports_dict),
            return_exceptions=True,
        )

        bull_args = debate_tasks[0] if not isinstance(debate_tasks[0], Exception) else []
        bear_args = debate_tasks[1] if not isinstance(debate_tasks[1], Exception) else []

        # Determine final stance
        bull_strength = len(bull_args)
        bear_strength = len(bear_args)
        if bull_strength > bear_strength:
            final_stance = "bullish"
        elif bear_strength > bull_strength:
            final_stance = "bearish"
        else:
            final_stance = "neutral"

        debate_conclusion = DebateConclusionPayload(
            symbol=symbol, name=name,
            bull_arguments=bull_args,
            bear_arguments=bear_args,
            final_stance=final_stance,
            conviction_level=abs(bull_strength - bear_strength) / max(bull_strength + bear_strength, 1),
            key_risks=bear_args[:3],
        )

        # ===== Phase 3: Trade Proposal =====
        logger.info(f"Phase 3: Trade proposal for {symbol}")
        trade_proposal = await self.trading_agent.generate_proposal(
            symbol, name, debate_conclusion.model_dump()
        )

        # ===== Phase 4: Risk Assessment & Approval =====
        logger.info(f"Phase 4: Risk assessment for {symbol}")
        risk_assessment = await self.risk_manager.assess(symbol, name, trade_proposal)
        approval = await self.portfolio_manager.approve(
            symbol, name, trade_proposal, risk_assessment
        )

        # ===== Compile Result =====
        result = {
            "session_id": session_id,
            "correlation_id": correlation_id,
            "symbol": symbol,
            "name": name,
            "phase": 5,
            "status": "completed" if not errors else "completed_with_errors",

            # Phase 1
            "fundamental_report": fundamental_rpt.model_dump() if fundamental_rpt else None,
            "sentiment_report": sentiment_rpt.model_dump() if sentiment_rpt else None,
            "news_report": news_rpt.model_dump() if news_rpt else None,
            "technical_report": technical_rpt.model_dump() if technical_rpt else None,

            # Phase 2
            "debate_conclusion": debate_conclusion.model_dump(),

            # Phase 3
            "trade_proposal": trade_proposal.model_dump(),

            # Phase 4
            "risk_assessment": risk_assessment.model_dump(),
            "approval_decision": approval.model_dump(),

            # Meta
            "errors": errors,
            "completed_at": datetime.utcnow().isoformat(),
        }

        logger.info(f"Agent workflow completed: session={session_id}, status={result['status']}")
        return result
