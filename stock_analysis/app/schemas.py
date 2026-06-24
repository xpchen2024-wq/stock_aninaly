# ============================================================================
# AI Stock Analysis Platform - Agent Message Schemas
# ============================================================================
from __future__ import annotations

from datetime import datetime
from typing import Optional, Dict, Any, List
from uuid import uuid4

from pydantic import BaseModel, Field


# -- Base Message -------------------------------------------------------------
class AgentMessage(BaseModel):
    """Base message for inter-agent communication (NF-013, NF-014)."""
    message_id: str = Field(default_factory=lambda: str(uuid4()))
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")
    source_agent: str = ""
    target_agent: str = ""
    message_type: str = ""
    payload: Dict[str, Any] = Field(default_factory=dict)
    schema_version: str = "1.0.0"
    correlation_id: str = Field(default_factory=lambda: str(uuid4()))


# -- Fundamental Analysis -----------------------------------------------------
class FundamentalReportPayload(BaseModel):
    symbol: str
    name: str
    scores: Dict[str, float] = Field(default_factory=dict)
    red_flags: List[str] = Field(default_factory=list)
    intrinsic_value_range: Dict[str, float] = Field(default_factory=dict)
    confidence: float = 0.0


# -- Sentiment Analysis -------------------------------------------------------
class SentimentReportPayload(BaseModel):
    symbol: str
    name: str
    sentiment_score: float = 0.0  # -1 to 1
    sentiment_trend: str = ""     # improving / deteriorating / stable
    key_drivers: List[str] = Field(default_factory=list)
    confidence: float = 0.0


# -- News Impact Analysis -----------------------------------------------------
class NewsImpactPayload(BaseModel):
    symbol: str
    name: str
    event_summary: str = ""
    impact_score: float = 0.0    # -1 to 1
    macro_score: float = 0.0
    key_events: List[Dict[str, str]] = Field(default_factory=list)


# -- Technical Analysis -------------------------------------------------------
class TechnicalSignalPayload(BaseModel):
    symbol: str
    name: str
    trend_direction: str = ""    # bullish / bearish / sideways
    support_levels: List[float] = Field(default_factory=list)
    resistance_levels: List[float] = Field(default_factory=list)
    patterns: List[str] = Field(default_factory=list)
    indicators: Dict[str, Any] = Field(default_factory=dict)
    confidence: float = 0.0


# -- Debate Conclusion --------------------------------------------------------
class DebateConclusionPayload(BaseModel):
    symbol: str
    name: str
    bull_arguments: List[str] = Field(default_factory=list)
    bear_arguments: List[str] = Field(default_factory=list)
    final_stance: str = ""       # bullish / bearish / neutral
    conviction_level: float = 0.0
    key_risks: List[str] = Field(default_factory=list)


# -- Trade Proposal -----------------------------------------------------------
class TradeProposalPayload(BaseModel):
    symbol: str
    name: str
    side: str = ""               # buy / sell / hold
    quantity: int = 0
    stop_loss: float = 0.0
    take_profit: float = 0.0
    rationale: str = ""
    confidence: float = 0.0


# -- Risk Assessment ----------------------------------------------------------
class RiskAssessmentPayload(BaseModel):
    symbol: str
    var_value: float = 0.0
    cvar_value: float = 0.0
    concentration_risk: float = 0.0
    stress_test_results: Dict[str, float] = Field(default_factory=dict)
    risk_level: str = ""         # low / medium / high / critical
    recommendations: List[str] = Field(default_factory=list)


# -- Approval Decision --------------------------------------------------------
class ApprovalDecisionPayload(BaseModel):
    symbol: str
    proposal_id: str = ""
    status: str = ""             # approved / rejected
    allocated_amount: float = 0.0
    rationale: str = ""
    conditions: List[str] = Field(default_factory=list)


# -- Message Factory ----------------------------------------------------------
class MessageFactory:
    """Helper to create properly structured agent messages."""

    @staticmethod
    def fundamental_report(source: str, target: str, payload: FundamentalReportPayload,
                           correlation_id: str = "") -> AgentMessage:
        return AgentMessage(
            source_agent=source,
            target_agent=target,
            message_type="FUNDAMENTAL_ANALYSIS_REPORT",
            payload=payload.model_dump(),
            correlation_id=correlation_id or str(uuid4()),
        )

    @staticmethod
    def sentiment_report(source: str, target: str, payload: SentimentReportPayload,
                         correlation_id: str = "") -> AgentMessage:
        return AgentMessage(
            source_agent=source,
            target_agent=target,
            message_type="SENTIMENT_REPORT",
            payload=payload.model_dump(),
            correlation_id=correlation_id or str(uuid4()),
        )

    @staticmethod
    def news_impact(source: str, target: str, payload: NewsImpactPayload,
                    correlation_id: str = "") -> AgentMessage:
        return AgentMessage(
            source_agent=source,
            target_agent=target,
            message_type="NEWS_IMPACT_REPORT",
            payload=payload.model_dump(),
            correlation_id=correlation_id or str(uuid4()),
        )

    @staticmethod
    def technical_signal(source: str, target: str, payload: TechnicalSignalPayload,
                         correlation_id: str = "") -> AgentMessage:
        return AgentMessage(
            source_agent=source,
            target_agent=target,
            message_type="TECHNICAL_SIGNAL",
            payload=payload.model_dump(),
            correlation_id=correlation_id or str(uuid4()),
        )

    @staticmethod
    def debate_conclusion(source: str, target: str, payload: DebateConclusionPayload,
                          correlation_id: str = "") -> AgentMessage:
        return AgentMessage(
            source_agent=source,
            target_agent=target,
            message_type="DEBATE_CONCLUSION",
            payload=payload.model_dump(),
            correlation_id=correlation_id or str(uuid4()),
        )

    @staticmethod
    def trade_proposal(source: str, target: str, payload: TradeProposalPayload,
                       correlation_id: str = "") -> AgentMessage:
        return AgentMessage(
            source_agent=source,
            target_agent=target,
            message_type="TRADE_PROPOSAL",
            payload=payload.model_dump(),
            correlation_id=correlation_id or str(uuid4()),
        )

    @staticmethod
    def risk_assessment(source: str, target: str, payload: RiskAssessmentPayload,
                        correlation_id: str = "") -> AgentMessage:
        return AgentMessage(
            source_agent=source,
            target_agent=target,
            message_type="RISK_ASSESSMENT",
            payload=payload.model_dump(),
            correlation_id=correlation_id or str(uuid4()),
        )

    @staticmethod
    def approval_decision(source: str, target: str, payload: ApprovalDecisionPayload,
                          correlation_id: str = "") -> AgentMessage:
        return AgentMessage(
            source_agent=source,
            target_agent=target,
            message_type="APPROVAL_DECISION",
            payload=payload.model_dump(),
            correlation_id=correlation_id or str(uuid4()),
        )
