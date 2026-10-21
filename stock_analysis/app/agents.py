# ============================================================================
# AI Stock Analysis Platform - Multi-Agent Collaboration System
# 8 Agents + 5-Phase Workflow
# ============================================================================
from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
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
from app.risk_metrics import historical_var_cvar, position_size
from app.utils.json_extract import extract_json, parse_llm_json, build_retry_prompt

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
    """Backward-compatible alias; canonical implementation in app.utils.json_extract."""
    return extract_json(text)


# -- Base Agent ---------------------------------------------------------------
class BaseAgent:
    def __init__(self, agent_id: str, name: str, role: str, gateway: ModelGateway):
        self.agent_id = agent_id
        self.name = name
        self.role = role
        self.gateway = gateway
        self.last_parse_ok: bool = True

    async def _call_llm(self, system_prompt: str, user_prompt: str,
                        max_tokens: int = 1500) -> LLMResponse:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        return await self.gateway.chat(messages=messages, max_tokens=max_tokens)

    async def _call_llm_json(self, system_prompt: str, user_prompt: str,
                             max_tokens: int = 1500) -> tuple[Optional[dict], int]:
        """LLM call with strict-JSON intent; retries once on unparseable output.

        Returns (parsed_dict_or_None, attempts_used). Infrastructure failures
        (success=False) are NOT retried — only bad/partial content is.
        Sets self.last_parse_ok for the orchestrator's degradation notes.
        """
        resp = await self._call_llm(system_prompt, user_prompt, max_tokens)
        parsed = parse_llm_json(resp.content) if (resp.success and resp.content) else None
        attempts = 1
        if parsed is None and resp.success and resp.content:
            retry_user = build_retry_prompt(user_prompt, resp.content)
            retry_resp = await self._call_llm(system_prompt, retry_user, max_tokens)
            if retry_resp.success:
                parsed = parse_llm_json(retry_resp.content)
            attempts = 2
        self.last_parse_ok = parsed is not None
        return parsed, attempts


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

        data, _attempts = await self._call_llm_json(system, user)
        if data is not None:
            return FundamentalReportPayload(
                symbol=symbol, name=name,
                scores=data.get("scores", {}),
                red_flags=data.get("red_flags", []),
                intrinsic_value_range=data.get("intrinsic_value_range", {}),
                confidence=data.get("confidence", 0.5),
            )
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

        data, _attempts = await self._call_llm_json(system, user)
        if data is not None:
            return SentimentReportPayload(
                symbol=symbol, name=name,
                sentiment_score=data.get("sentiment_score", 0),
                sentiment_trend=data.get("sentiment_trend", "stable"),
                key_drivers=data.get("key_drivers", []),
                confidence=data.get("confidence", 0.5),
            )
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

        data, _attempts = await self._call_llm_json(system, user)
        if data is not None:
            return NewsImpactPayload(
                symbol=symbol, name=name,
                event_summary=data.get("event_summary", ""),
                impact_score=data.get("impact_score", 0),
                macro_score=data.get("macro_score", 0),
                key_events=data.get("key_events", []),
            )
        return NewsImpactPayload(
            symbol=symbol, name=name,
            event_summary="", impact_score=0, macro_score=0, key_events=[],
        )


class TechnicalAnalyst(BaseAgent):
    """AGENT-ANL-04: Detects trading patterns and technical signals."""

    def __init__(self, gateway: ModelGateway, failover: FailoverManager):
        super().__init__(AGENT_TECHNICAL, "技术分析师", "technical", gateway)
        self.failover = failover

    async def analyze(self, symbol: str, name: str,
                      klines: Optional[list] = None) -> TechnicalSignalPayload:
        # Fetch K-line data and compute indicators
        closes = np.array([])
        try:
            if klines is None:
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

        data, _attempts = await self._call_llm_json(system, user)
        if data is not None:
            return TechnicalSignalPayload(
                symbol=symbol, name=name,
                trend_direction=data.get("trend_direction", "sideways"),
                support_levels=data.get("support_levels", []),
                resistance_levels=data.get("resistance_levels", []),
                patterns=data.get("patterns", []),
                indicators=indicators,
                confidence=data.get("confidence", 0.5),
            )
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

    async def rebuttal(self, symbol: str, name: str,
                       own_args: List[str], opponent_args: List[str]) -> List[str]:
        """Round 2: respond to the bearish case with targeted counter-points."""
        system = "你是乐观派投资研究员。请针对对方的看跌论点逐条反驳，维护或修正你的看涨立场。"
        user = f"""{name}({symbol}) 多空辩论第二轮。
你（看涨方）第一轮论点：
{chr(10).join('- ' + a for a in own_args) or '（无）'}
对方（看跌方）论点：
{chr(10).join('- ' + a for a in opponent_args) or '（无）'}
请给出最多3条针对性反驳。"""

        resp = await self._call_llm(system, user, max_tokens=600)
        lines = [l.strip("- ").strip() for l in resp.content.split("\n")
                 if l.strip().startswith("-") or l.strip().startswith("*")]
        return lines[:3] if lines else []


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

    async def rebuttal(self, symbol: str, name: str,
                       own_args: List[str], opponent_args: List[str]) -> List[str]:
        """Round 2: press the bearish case against the bullish arguments."""
        system = "你是谨慎派投资研究员。请针对对方的看涨论点逐条质疑，挖掘被忽视的风险。"
        user = f"""{name}({symbol}) 多空辩论第二轮。
你（看跌方）第一轮论点：
{chr(10).join('- ' + a for a in own_args) or '（无）'}
对方（看涨方）论点：
{chr(10).join('- ' + a for a in opponent_args) or '（无）'}
请给出最多3条针对性反驳。"""

        resp = await self._call_llm(system, user, max_tokens=600)
        lines = [l.strip("- ").strip() for l in resp.content.split("\n")
                 if l.strip().startswith("-") or l.strip().startswith("*")]
        return lines[:3] if lines else []


# -- Phase 3: Trading Agent --------------------------------------------------
class ResearchManager(BaseAgent):
    """AGENT-RES-03: Judges the bull/bear debate and synthesizes the stance."""

    def __init__(self, gateway: ModelGateway):
        super().__init__("AGENT-RES-03", "研究经理", "research_manager", gateway)

    async def synthesize(self, symbol: str, name: str,
                         bull_args: List[str], bear_args: List[str],
                         bull_rebuttals: List[str], bear_rebuttals: List[str]
                         ) -> Optional[DebateConclusionPayload]:
        system = "你是研究经理，主持多空辩论并做出裁决。请综合双方两轮论点给出结论，输出严格JSON格式。"
        user = f"""{name}({symbol}) 多空辩论已结束，请裁决：
看涨方论点：{json.dumps(bull_args, ensure_ascii=False)}
看涨方反驳：{json.dumps(bull_rebuttals, ensure_ascii=False)}
看跌方论点：{json.dumps(bear_args, ensure_ascii=False)}
看跌方反驳：{json.dumps(bear_rebuttals, ensure_ascii=False)}
输出JSON：{{"final_stance":"bullish/bearish/neutral","conviction_level":0.0,"key_risks":["风险1","风险2"],"judge_summary":"一段话裁决理由"}}
conviction_level 取值 0.0-1.0，表示你对最终立场的信心强度。"""

        data, _attempts = await self._call_llm_json(system, user)
        if data is None or not data.get("final_stance"):
            return None
        stance = data["final_stance"]
        if stance not in ("bullish", "bearish", "neutral"):
            return None
        try:
            conviction = min(max(float(data.get("conviction_level", 0.5)), 0.0), 1.0)
        except (TypeError, ValueError):
            conviction = 0.5
        return DebateConclusionPayload(
            symbol=symbol, name=name,
            bull_arguments=bull_args,
            bear_arguments=bear_args,
            final_stance=stance,
            conviction_level=conviction,
            key_risks=data.get("key_risks", [])[:5] or bear_args[:3],
        )


# -- Phase 3b: Trade Proposal -------------------------------------------------
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

        data, _attempts = await self._call_llm_json(system, user)
        if data is not None:
            return TradeProposalPayload(
                symbol=symbol, name=name,
                side=data.get("side", "hold"),
                quantity=data.get("quantity", 0),
                stop_loss=data.get("stop_loss", 0),
                take_profit=data.get("take_profit", 0),
                rationale=data.get("rationale", ""),
                confidence=data.get("confidence", 0.5),
            )
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
                     proposal: TradeProposalPayload,
                     historical_closes: Optional[np.ndarray] = None,
                     price_hint: Optional[float] = None) -> RiskAssessmentPayload:
        """Assess proposal risk. When real K-line history is supplied, VaR/CVaR
        are computed deterministically and the LLM only provides qualitative
        judgement (risk_level + recommendations)."""
        computed = None
        if historical_closes is not None and price_hint is not None and price_hint > 0:
            position_value = float(price_hint) * max(float(proposal.quantity), 0.0)
            if position_value <= 0:
                position_value = 1_000_000.0
            computed = historical_var_cvar(
                historical_closes, position_value,
                confidence_level=0.95, horizon_days=5,
            )

        system = "你是风险管理专家。请评估交易提案的风险，输出JSON格式。"
        if computed is not None:
            user = f"""请评估以下交易提案的风险（VaR/CVaR 已由系统基于历史行情确定性计算，请勿自行估算数值）：
{name}({symbol}) - {proposal.side} {proposal.quantity}股
止损：{proposal.stop_loss} 止盈：{proposal.take_profit}
系统基于近 60 日历史行情确定性计算（95% 置信）：VaR={computed['var_value']}，CVaR={computed['cvar_value']}（样本数 {computed['sample_size']}）
输出JSON：{{"risk_level":"low/medium/high/critical","recommendations":[]}}"""
        else:
            user = f"""请评估以下交易提案的风险：
{name}({symbol}) - {proposal.side} {proposal.quantity}股
止损：{proposal.stop_loss} 止盈：{proposal.take_profit}
输出JSON：{{"var_value":0.0,"cvar_value":0.0,"concentration_risk":0.0,"stress_test_results":{{}},"risk_level":"low/medium/high/critical","recommendations":[]}}"""

        data, _attempts = await self._call_llm_json(system, user)
        if data is not None:
            var_value = computed["var_value"] if computed else data.get("var_value", 0)
            cvar_value = computed["cvar_value"] if computed else data.get("cvar_value", 0)
            return RiskAssessmentPayload(
                symbol=symbol,
                var_value=var_value,
                cvar_value=cvar_value,
                concentration_risk=data.get("concentration_risk", 0),
                stress_test_results=data.get("stress_test_results", {}),
                risk_level=data.get("risk_level", "medium"),
                recommendations=data.get("recommendations", []),
            )
        return RiskAssessmentPayload(
            symbol=symbol,
            var_value=computed["var_value"] if computed else 0,
            cvar_value=computed["cvar_value"] if computed else 0,
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

        data, _attempts = await self._call_llm_json(system, user)
        if data is not None:
            return ApprovalDecisionPayload(
                symbol=symbol,
                status=data.get("status", "rejected"),
                allocated_amount=data.get("allocated_amount", 0),
                rationale=data.get("rationale", ""),
                conditions=data.get("conditions", []),
            )
        return ApprovalDecisionPayload(
            symbol=symbol, status="rejected",
            allocated_amount=0, rationale="审批失败", conditions=[],
        )


# -- Fallback Analyzer (always produces a real-data report) ------------------
class FallbackAnalyzer:
    """非 LLM 兜底分析器 - 总是基于真实数据生成详细报告 (即使 LLM 不可用).

    用途: 在 Agent 流程末尾附带生成一份综合报告. 即便 LLM 因任何原因(网络/限流/被拒)
    返回空对象, 用户也能看到基于真实行情/基本面/新闻的分析.
    """

    def __init__(self, failover: FailoverManager):
        self.failover = failover

    async def generate(self, symbol: str, name: str,
                       approval: ApprovalDecisionPayload,
                       trade: TradeProposalPayload,
                       debate: DebateConclusionPayload,
                       fundamentals: Optional[Dict] = None,
                       news_items: Optional[List[Dict]] = None) -> Dict[str, Any]:
        """综合所有真实数据源, 输出一份结构化分析报告 (无需 LLM)."""
        report: Dict[str, Any] = {
            "symbol": symbol, "name": name,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "data_source": "real_data_fallback",
        }

        # ── 1. 实时行情 ──────────────────────────────────────────────
        try:
            quote = await self.failover.fetch_with_failover(DataType.REALTIME_QUOTE, symbol)
            report["market_quote"] = {
                "price": getattr(quote, "price", 0.0),
                "change_pct": getattr(quote, "change_pct", 0.0),
                "high": getattr(quote, "high", 0.0),
                "low": getattr(quote, "low", 0.0),
                "open": getattr(quote, "open", 0.0),
                "pre_close": getattr(quote, "pre_close", 0.0),
                "volume": getattr(quote, "volume", 0),
            }
        except Exception as e:
            logger.warning(f"FallbackAnalyzer: quote fetch failed: {e}")
            report["market_quote"] = None

        # ── 2. 走势 (用 K 线 + 真实技术指标) ─────────────────────────
        try:
            klines = await self.failover.fetch_with_failover(DataType.KLINE, symbol)
            closes = np.array([k.close for k in klines[-60:]])
            highs = np.array([k.high for k in klines[-60:]])
            lows = np.array([k.low for k in klines[-60:]])
            volumes = np.array([k.volume for k in klines[-60:]])

            indicators = compute_all_indicators(highs, lows, closes, volumes)

            # 走势方向 (基于 MA5/MA20/MA60 关系)
            ma5 = float(indicators.get("ma5", 0) or 0)
            ma20 = float(indicators.get("ma20", 0) or 0)
            ma60 = float(indicators.get("ma60", 0) or 0)
            if ma5 > ma20 > ma60:
                trend_dir = "bullish"; trend_label = "多头排列, 上升趋势"
            elif ma5 < ma20 < ma60:
                trend_dir = "bearish"; trend_label = "空头排列, 下降趋势"
            elif ma5 > ma20:
                trend_dir = "recovery"; trend_label = "短期反弹, 关注中期均线压力"
            elif ma5 < ma20:
                trend_dir = "weakening"; trend_label = "短期回落, 关注中期均线支撑"
            else:
                trend_dir = "sideways"; trend_label = "横盘整理, 等待方向选择"

            # 支撑/压力 (用最近 60 根 K 线的 min/max)
            support_levels = [float(lows[-20:].min())] if len(lows) >= 20 else []
            resistance_levels = [float(highs[-20:].max())] if len(highs) >= 20 else []

            # 涨跌幅 (近 20 日)
            if len(closes) >= 20 and closes[-20] > 0:
                change_20d = (closes[-1] - closes[-20]) / closes[-20] * 100
            else:
                change_20d = 0.0

            report["trend"] = {
                "direction": trend_dir,
                "label": trend_label,
                "ma5": round(ma5, 2),
                "ma20": round(ma20, 2),
                "ma60": round(ma60, 2),
                "change_20d_pct": round(change_20d, 2),
                "support_levels": [round(s, 2) for s in support_levels],
                "resistance_levels": [round(r, 2) for r in resistance_levels],
                "indicators": {k: round(float(v), 4) if isinstance(v, (int, float)) else v
                               for k, v in indicators.items() if k in
                               ("ma5", "ma20", "ma60", "rsi_14",
                                "macd_dif", "macd_dea", "macd_hist",
                                "bollinger_upper", "bollinger_middle", "bollinger_lower",
                                "atr_14", "kdj_k", "kdj_d")},
            }
        except Exception as e:
            logger.warning(f"FallbackAnalyzer: trend fetch failed: {e}")
            report["trend"] = None

        # ── 3. 财报基本面 ────────────────────────────────────────────
        fund = None
        if fundamentals:
            fund = fundamentals
        else:
            try:
                fd = await self.failover.fetch_with_failover(DataType.FUNDAMENTAL, symbol)
                fund = {
                    "pe_ratio": getattr(fd, "pe_ratio", 0.0),
                    "pb_ratio": getattr(fd, "pb_ratio", 0.0),
                    "market_cap": getattr(fd, "market_cap", 0.0),
                    "circulating_cap": getattr(fd, "circulating_cap", 0.0),
                    "industry": getattr(fd, "industry", ""),
                    "roe": getattr(fd, "roe", 0.0),
                }
            except Exception as e:
                logger.warning(f"FallbackAnalyzer: fundamental fetch failed: {e}")

        if fund:
            pe = float(fund.get("pe_ratio", 0) or 0)
            pb = float(fund.get("pb_ratio", 0) or 0)
            roe = float(fund.get("roe", 0) or 0)
            mc = float(fund.get("market_cap", 0) or 0)
            industry = fund.get("industry", "")
            # 估值/盈利能力评级
            if 0 < pe < 15 and 0 < pb < 1.5 and roe > 10:
                val_label = "低估值 + 高 ROE, 价值洼地"
                val_color = "good"
            elif pe > 50 or pb > 8:
                val_label = "高估值, 业绩支撑是核心"
                val_color = "warning"
            elif pe < 0:
                val_label = "亏损或微利, PE 失效"
                val_color = "bad"
            else:
                val_label = "估值中性, 关注成长性"
                val_color = "neutral"
            report["fundamentals"] = {
                **fund,
                "pe_ratio": pe, "pb_ratio": pb, "roe": roe, "market_cap": mc,
                "valuation_label": val_label, "valuation_color": val_color,
                "industry": industry,
            }
        else:
            report["fundamentals"] = None

        # ── 4. 最新资讯 ─────────────────────────────────────────────
        news_list: List[Dict[str, Any]] = []
        if news_items:
            news_list = list(news_items)
        else:
            try:
                raw = await self.failover.fetch_with_failover(DataType.NEWS, symbol)
                news_list = [
                    {
                        "title": getattr(n, "title", ""),
                        "source": getattr(n, "source", ""),
                        "url": getattr(n, "url", ""),
                        "published_at": (getattr(n, "published_at", None).isoformat()
                                        if getattr(n, "published_at", None) else ""),
                        "content": (getattr(n, "content", "") or "")[:200],
                    }
                    for n in (raw or [])[:10]
                ]
            except Exception as e:
                logger.warning(f"FallbackAnalyzer: news fetch failed: {e}")

        if news_list:
            # 关键事件: 取最近 3 条
            key_events = [n.get("title", "") for n in news_list[:3] if n.get("title")]
            # 资讯摘要 (基于前 3 条拼接)
            summary_parts = [n.get("title", "") for n in news_list[:5] if n.get("title")]
            news_summary = " · ".join(summary_parts)[:300] if summary_parts else "近 7 日无重大新闻"
            report["news"] = {
                "count": len(news_list),
                "summary": news_summary,
                "key_events": key_events,
                "items": news_list[:5],
            }
        else:
            report["news"] = {"count": 0, "summary": "暂无最近新闻", "key_events": [], "items": []}

        # ── 5. 市场关注度 (基于多空辩论 + 新闻 + 行情综合) ─────────────
        bull_n = len(debate.bull_arguments or [])
        bear_n = len(debate.bear_arguments or [])
        news_n = report["news"]["count"]
        # 关注度评级
        if news_n >= 5 and abs(bull_n - bear_n) <= 1:
            attention_level = "high"
            attention_label = "高关注: 资讯密集 + 多空分歧大"
        elif news_n >= 3:
            attention_level = "medium"
            attention_label = "中等关注: 有持续报道"
        else:
            attention_level = "low"
            attention_label = "低关注: 资讯稀少, 缺乏催化"

        report["market_attention"] = {
            "level": attention_level,
            "label": attention_label,
            "bull_arguments_count": bull_n,
            "bear_arguments_count": bear_n,
            "news_count_7d": news_n,
            "consensus_stance": debate.final_stance,
        }

        # ── 6. 投资方向 ──────────────────────────────────────────────
        trend_dir = (report.get("trend") or {}).get("direction", "sideways")
        fund_color = (report.get("fundamentals") or {}).get("valuation_color", "neutral")
        attn_level = attention_level

        if trend_dir == "bullish" and fund_color in ("good", "neutral"):
            short_dir = "顺势: 关注回踩支撑位低吸机会"
        elif trend_dir == "bullish" and fund_color == "warning":
            short_dir = "谨慎: 趋势向好但估值偏高, 严控仓位"
        elif trend_dir == "bearish" and fund_color == "bad":
            short_dir = "回避: 趋势向下 + 业绩疲软"
        elif trend_dir == "bearish" and fund_color in ("good", "neutral"):
            short_dir = "观望: 趋势偏弱但估值合理, 等待企稳信号"
        else:
            short_dir = "震荡: 区间操作为主, 不追涨杀跌"

        if attn_level == "high":
            watch_points = ["重大政策/事件催化", "分析师评级密集调整", "市场情绪拐点"]
        elif attn_level == "medium":
            watch_points = ["季报披露窗口", "行业景气度变化", "同板块联动"]
        else:
            watch_points = ["突发新闻/政策", "技术面突破/破位", "成交量异动"]

        report["investment_direction"] = {
            "short_term": short_dir,
            "mid_term": ("关注" + (industry or "该行业") + "景气度 + 估值修复空间") if fund else "关注行业景气度",
            "long_term": "需结合公司护城河 + 长期成长性 (本次为数据快照, 不替代深度研究)",
            "watch_points": watch_points,
            "approval_status": approval.status,
            "approval_rationale": approval.rationale or "",
        }

        # ── 7. 综合摘要 (一段话) ─────────────────────────────────────
        price_txt = ""
        mq = report.get("market_quote")
        if mq and mq.get("price"):
            ch = mq.get("change_pct", 0)
            price_txt = f"现价 {mq['price']:.2f} ({'+' if ch>=0 else ''}{ch:.2f}%)"
        trend_txt = (report.get("trend") or {}).get("label", "走势数据缺失")
        fund_txt = (report.get("fundamentals") or {}).get("valuation_label", "基本面数据缺失")
        news_txt = report["news"]["summary"][:60]
        report["summary"] = (
            f"{name or symbol} ({symbol}) {price_txt}. "
            f"走势: {trend_txt}. "
            f"估值: {fund_txt}. "
            f"市场关注度: {attention_label}. "
            f"最新资讯: {news_txt}."
        )

        return report



class AgentWorkflow:
    """Orchestrates the complete multi-agent decision pipeline."""

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
        self.research_manager = ResearchManager(gateway)
        self.trading_agent = TradingAgent(gateway)
        self.risk_manager = RiskManager(gateway)
        self.portfolio_manager = PortfolioManager(gateway)
        self.fallback_analyzer = FallbackAnalyzer(failover)

    # ── Shared data loading ────────────────────────────────────────────────
    async def _load_shared_data(self, symbol: str,
                                fundamentals: Optional[Dict],
                                news_items: Optional[List[Dict]]):
        """Fetch K-line and news once so every consumer reuses them."""
        klines: Optional[list] = None
        try:
            klines = await self.failover.fetch_with_failover(DataType.KLINE, symbol)
        except Exception as e:
            logger.warning(f"Workflow shared kline fetch failed: {e}")

        resolved_news = news_items
        if resolved_news is None:
            try:
                fetched = await self.failover.fetch_with_failover(DataType.NEWS, symbol)
                # Only well-formed dict items are usable downstream
                resolved_news = [n for n in (fetched or []) if isinstance(n, dict)]
            except Exception as e:
                logger.warning(f"Workflow shared news fetch failed: {e}")
                resolved_news = []

        headlines = [
            str(n.get("title", "")).strip()
            for n in (resolved_news or [])
            if isinstance(n, dict) and n.get("title")
        ]

        return klines, resolved_news or [], headlines

    async def _phase1_analysts(self, symbol, name, fundamentals, news_items, headlines):
        errors = []
        phase1_results = await asyncio.gather(
            self.fundamental_analyst.analyze(symbol, name, fundamentals),
            self.sentiment_analyst.analyze(symbol, name, headlines),
            self.news_analyst.analyze(symbol, name, news_items),
            self.technical_analyst.analyze(symbol, name, klines=self._shared_klines),
            return_exceptions=True,
        )
        fundamental_rpt = phase1_results[0] if not isinstance(phase1_results[0], Exception) else None
        sentiment_rpt = phase1_results[1] if not isinstance(phase1_results[1], Exception) else None
        news_rpt = phase1_results[2] if not isinstance(phase1_results[2], Exception) else None
        technical_rpt = phase1_results[3] if not isinstance(phase1_results[3], Exception) else None

        agent_names = ["基本面分析师", "情绪分析师", "新闻分析师", "技术分析师"]
        agents = [self.fundamental_analyst, self.sentiment_analyst,
                  self.news_analyst, self.technical_analyst]
        for i, r in enumerate(phase1_results):
            if isinstance(r, Exception):
                errors.append(f"{agent_names[i]}失败: {str(r)}")
            elif not getattr(agents[i], "last_parse_ok", True):
                errors.append(f"{agent_names[i]}: LLM输出无法解析(已重试1次)，使用默认值")

        reports_dict = {
            "fundamental": fundamental_rpt.model_dump() if fundamental_rpt else {},
            "sentiment": sentiment_rpt.model_dump() if sentiment_rpt else {},
            "news": news_rpt.model_dump() if news_rpt else {},
            "technical": technical_rpt.model_dump() if technical_rpt else {},
        }
        return fundamental_rpt, sentiment_rpt, news_rpt, technical_rpt, reports_dict, errors

    async def _phase2_debate(self, symbol, name, reports_dict):
        round2: Dict[str, Any] = {}
        debate_tasks = await asyncio.gather(
            self.bull_researcher.debate(symbol, name, reports_dict),
            self.bear_researcher.debate(symbol, name, reports_dict),
            return_exceptions=True,
        )
        bull_args = debate_tasks[0] if not isinstance(debate_tasks[0], Exception) else []
        bear_args = debate_tasks[1] if not isinstance(debate_tasks[1], Exception) else []

        rebuttal_tasks = await asyncio.gather(
            self.bull_researcher.rebuttal(symbol, name, bull_args or [], bear_args or []),
            self.bear_researcher.rebuttal(symbol, name, bear_args or [], bull_args or []),
            return_exceptions=True,
        )
        bull_rebuttals = rebuttal_tasks[0] if not isinstance(rebuttal_tasks[0], Exception) else []
        bear_rebuttals = rebuttal_tasks[1] if not isinstance(rebuttal_tasks[1], Exception) else []
        round2 = {"bull_rebuttals": bull_rebuttals, "bear_rebuttals": bear_rebuttals}

        judgement = await self.research_manager.synthesize(
            symbol, name, bull_args, bear_args, bull_rebuttals, bear_rebuttals
        )
        if judgement is not None:
            return judgement, round2, []

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
        note = "研究经理: 裁决输出无法解析，回退为论点计数启发式"
        return debate_conclusion, round2, [note]

    def _deterministic_quantity(self, proposal: TradeProposalPayload,
                                last_close: Optional[float]) -> tuple[int, str]:
        if last_close is None or last_close <= 0:
            return proposal.quantity, "llm_default"
        qty = position_size(
            confidence=proposal.confidence,
            price=float(last_close),
            stop_loss=float(proposal.stop_loss or 0),
        )
        return qty, "risk_parity_deterministic"

    async def _finalize(self, symbol, name, session_id, correlation_id, errors,
                        fundamental_rpt, sentiment_rpt, news_rpt, technical_rpt,
                        debate_conclusion, round2, trade_proposal,
                        risk_assessment, approval, fundamentals, news_items,
                        quantity_method: str = "llm_default") -> Dict[str, Any]:
        trade_proposal_dict = trade_proposal.model_dump()
        trade_proposal_dict["quantity_method"] = quantity_method

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
            "research_judgement": {"rounds": 2, **round2},

            # Phase 3
            "trade_proposal": trade_proposal_dict,

            # Phase 4
            "risk_assessment": risk_assessment.model_dump(),
            "approval_decision": approval.model_dump(),

            # Phase 5 (real-data fallback, 总是有内容, 不依赖 LLM)
            "fallback_detailed_report": await self.fallback_analyzer.generate(
                symbol, name, approval, trade_proposal, debate_conclusion,
                fundamentals=fundamentals, news_items=news_items,
            ),

            # Meta
            "errors": errors,
            "completed_at": datetime.now(timezone.utc).isoformat(),
        }
        logger.info(f"Agent workflow completed: session={session_id}, status={result['status']}")
        return result

    async def run(self, symbol: str, name: str = "",
                  fundamentals: Optional[Dict] = None,
                  news_items: Optional[List[Dict]] = None) -> Dict[str, Any]:
        """
        Execute the complete multi-agent decision workflow.
        Returns full decision record.
        """
        session_id = str(uuid4())
        correlation_id = str(uuid4())
        errors: List[str] = []

        logger.info(f"Starting agent workflow: session={session_id}, symbol={symbol}")

        async for kind, payload in self.run_stream(
            symbol=symbol, name=name, fundamentals=fundamentals,
            news_items=news_items, session_id=session_id,
            correlation_id=correlation_id,
        ):
            if kind == "result":
                return payload
        raise RuntimeError("run_stream finished without a result event")

    async def run_stream(self, symbol: str, name: str = "",
                         fundamentals: Optional[Dict] = None,
                         news_items: Optional[List[Dict]] = None,
                         session_id: Optional[str] = None,
                         correlation_id: Optional[str] = None):
        """Async generator variant of run(): yields ("phase", info) progress
        events followed by a final ("result", full_record) event."""
        session_id = session_id or str(uuid4())
        correlation_id = correlation_id or str(uuid4())
        errors: List[str] = []
        self._shared_klines: Optional[list] = None

        logger.info(f"Starting agent workflow: session={session_id}, symbol={symbol}")

        yield "phase", {"phase": 0, "label": "数据加载",
                        "message": f"正在获取 {symbol} 行情与资讯..."}

        klines, resolved_news, headlines = await self._load_shared_data(
            symbol, fundamentals, news_items
        )
        self._shared_klines = klines

        yield "phase", {"phase": 1, "status": "running", "label": "分析师团队",
                        "message": "基本面/情绪/新闻/技术 四位分析师并行分析中..."}
        (fundamental_rpt, sentiment_rpt, news_rpt, technical_rpt,
         reports_dict, p1_errors) = await self._phase1_analysts(
            symbol, name, fundamentals, resolved_news, headlines
        )
        errors.extend(p1_errors)
        yield "phase", {"phase": 1, "status": "done", "label": "分析师团队",
                        "message": "四位分析师分析完成 ✓"}

        yield "phase", {"phase": 2, "status": "running", "label": "多空辩论",
                        "message": "两轮多空对抗辩论进行中..."}
        debate_conclusion, round2, p2_errors = await self._phase2_debate(
            symbol, name, reports_dict
        )
        errors.extend(p2_errors)
        yield "phase", {"phase": 2, "status": "done", "label": "多空辩论",
                        "message": f"研究经理裁决: {debate_conclusion.final_stance} ✓"}

        yield "phase", {"phase": 3, "status": "running", "label": "交易提案",
                        "message": "交易代理生成交易提案中..."}
        trade_proposal = await self.trading_agent.generate_proposal(
            symbol, name, debate_conclusion.model_dump()
        )

        last_close: Optional[float] = None
        if klines:
            try:
                closes = [k.close for k in klines[-60:] if getattr(k, "close", None)]
                last_close = float(closes[-1]) if closes else None
            except (TypeError, ValueError):
                last_close = None
        qty, qty_method = self._deterministic_quantity(trade_proposal, last_close)
        trade_proposal.quantity = qty

        yield "phase", {"phase": 3, "status": "done", "label": "交易提案",
                        "message": f"交易提案就绪（仓位由风险模型确定: {qty_method}）✓"}

        yield "phase", {"phase": 4, "status": "running", "label": "风险评估与审批",
                        "message": "风险管理团队评估中，投资组合经理审批中..."}
        historical_closes: Optional[np.ndarray] = None
        if klines:
            try:
                historical_closes = np.array([k.close for k in klines[-60:]])
            except Exception:
                historical_closes = None

        risk_assessment = await self.risk_manager.assess(
            symbol, name, trade_proposal,
            historical_closes=historical_closes, price_hint=last_close,
        )
        approval = await self.portfolio_manager.approve(
            symbol, name, trade_proposal, risk_assessment
        )
        yield "phase", {"phase": 4, "status": "done", "label": "风险评估与审批",
                        "message": f"审批结果: {approval.status} ✓"}

        yield "phase", {"phase": 5, "status": "running", "label": "综合报告",
                        "message": "基于真实数据生成综合分析报告中..."}
        result = await self._finalize(
            symbol, name, session_id, correlation_id, errors,
            fundamental_rpt, sentiment_rpt, news_rpt, technical_rpt,
            debate_conclusion, round2, trade_proposal,
            risk_assessment, approval, fundamentals, resolved_news,
            quantity_method=qty_method,
        )
        yield "result", result
