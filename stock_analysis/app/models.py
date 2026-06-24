# ============================================================================
# AI Stock Analysis Platform - Database Models
# ============================================================================
from __future__ import annotations

import uuid
from datetime import datetime, date
from typing import Optional, List

from sqlalchemy import (
    Column, String, Text, Integer, Float, Boolean, DateTime, Date,
    Enum, ForeignKey, JSON, BigInteger, Index, UniqueConstraint
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship

from app.database import Base


# -- Helper -------------------------------------------------------------------
def new_uuid() -> str:
    return str(uuid.uuid4())


def now_utc() -> datetime:
    return datetime.utcnow()


# -- User ---------------------------------------------------------------------
class User(Base):
    __tablename__ = "users"

    id = Column(UUID(as_uuid=False), primary_key=True, default=new_uuid)
    username = Column(String(64), unique=True, nullable=False, index=True)
    email = Column(String(255), unique=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)
    is_superuser = Column(Boolean, default=False, nullable=False)
    created_at = Column(DateTime, default=now_utc, nullable=False)
    updated_at = Column(DateTime, default=now_utc, onupdate=now_utc, nullable=False)

    # Relationships
    watchlist_groups = relationship("WatchlistGroup", back_populates="user", cascade="all, delete-orphan")
    watchlist_items = relationship("WatchlistItem", back_populates="user", cascade="all, delete-orphan")


# -- Watchlist ----------------------------------------------------------------
class WatchlistGroup(Base):
    __tablename__ = "watchlist_groups"

    id = Column(UUID(as_uuid=False), primary_key=True, default=new_uuid)
    user_id = Column(UUID(as_uuid=False), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    name = Column(String(64), nullable=False)
    description = Column(String(255), nullable=True)
    sort_order = Column(Integer, default=0, nullable=False)
    created_at = Column(DateTime, default=now_utc, nullable=False)

    user = relationship("User", back_populates="watchlist_groups")
    items = relationship("WatchlistItem", back_populates="group", cascade="all, delete-orphan")


class WatchlistItem(Base):
    __tablename__ = "watchlist_items"

    id = Column(UUID(as_uuid=False), primary_key=True, default=new_uuid)
    user_id = Column(UUID(as_uuid=False), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    group_id = Column(UUID(as_uuid=False), ForeignKey("watchlist_groups.id", ondelete="SET NULL"), nullable=True)
    symbol = Column(String(32), nullable=False, index=True)
    name = Column(String(128), nullable=False)
    market = Column(String(16), nullable=True)  # SH / SZ / HK / US
    notes = Column(Text, nullable=True)
    added_at = Column(DateTime, default=now_utc, nullable=False)

    user = relationship("User", back_populates="watchlist_items")
    group = relationship("WatchlistGroup", back_populates="items")

    __table_args__ = (
        UniqueConstraint("user_id", "symbol", name="uq_user_symbol"),
    )


# -- Stock K-Line Cache (watchlist only, 90 days) ----------------------------
class KlineCache(Base):
    __tablename__ = "kline_cache"

    id = Column(UUID(as_uuid=False), primary_key=True, default=new_uuid)
    symbol = Column(String(32), nullable=False, index=True)
    frequency = Column(String(8), nullable=False)  # D / W / M / 5min / 15min / 30min / 60min
    trade_date = Column(Date, nullable=False)
    open = Column(Float, nullable=False)
    high = Column(Float, nullable=False)
    low = Column(Float, nullable=False)
    close = Column(Float, nullable=False)
    volume = Column(BigInteger, nullable=False)
    amount = Column(Float, nullable=True)
    turnover_rate = Column(Float, nullable=True)
    cached_at = Column(DateTime, default=now_utc, nullable=False)

    __table_args__ = (
        Index("ix_kline_symbol_freq_date", "symbol", "frequency", "trade_date"),
    )


# -- Stock Fundamental --------------------------------------------------------
class StockFundamental(Base):
    __tablename__ = "stock_fundamentals"

    id = Column(UUID(as_uuid=False), primary_key=True, default=new_uuid)
    symbol = Column(String(32), nullable=False, index=True)
    name = Column(String(128), nullable=False)
    market = Column(String(16), nullable=True)
    industry = Column(String(64), nullable=True)
    sector = Column(String(64), nullable=True)
    pe_ratio = Column(Float, nullable=True)
    pb_ratio = Column(Float, nullable=True)
    market_cap = Column(Float, nullable=True)          # 总市值 (亿元)
    circulating_cap = Column(Float, nullable=True)     # 流通市值 (亿元)
    total_shares = Column(Float, nullable=True)        # 总股本 (亿股)
    revenue_yoy = Column(Float, nullable=True)         # 营收同比 %
    profit_yoy = Column(Float, nullable=True)          # 利润同比 %
    roe = Column(Float, nullable=True)
    debt_ratio = Column(Float, nullable=True)
    report_date = Column(Date, nullable=True)
    updated_at = Column(DateTime, default=now_utc, onupdate=now_utc, nullable=False)

    __table_args__ = (
        UniqueConstraint("symbol", "report_date", name="uq_symbol_report_date"),
    )


# -- AI Model Config ----------------------------------------------------------
class ModelConfig(Base):
    __tablename__ = "model_configs"

    id = Column(UUID(as_uuid=False), primary_key=True, default=new_uuid)
    name = Column(String(128), nullable=False)
    platform = Column(String(32), nullable=False)       # openai / deepseek / openrouter / opencodezen / custom
    base_url = Column(String(512), nullable=True)
    api_key_encrypted = Column(Text, nullable=False)     # AES-256 encrypted
    model_name = Column(String(128), nullable=False)
    max_tokens = Column(Integer, default=4096, nullable=False)
    temperature = Column(Float, default=0.7, nullable=False)
    is_default = Column(Boolean, default=False, nullable=False)
    is_enabled = Column(Boolean, default=True, nullable=False)
    status = Column(String(16), default="active", nullable=False, index=True)  # active / deleted
    extra_headers = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=now_utc, nullable=False)
    updated_at = Column(DateTime, default=now_utc, onupdate=now_utc, nullable=False)

    bindings = relationship("AgentModelBinding", back_populates="model_config", cascade="all, delete-orphan")


class AgentModelBinding(Base):
    __tablename__ = "agent_model_bindings"

    id = Column(UUID(as_uuid=False), primary_key=True, default=new_uuid)
    model_config_id = Column(UUID(as_uuid=False), ForeignKey("model_configs.id", ondelete="CASCADE"), nullable=False)
    agent_id = Column(String(64), nullable=True)         # AGENT-ANL-01, AGENT-ANL-02, etc.
    scene = Column(String(64), nullable=True)             # hot_track / trend_analysis / report / serenity / decision
    is_active = Column(Boolean, default=True, nullable=False)
    created_at = Column(DateTime, default=now_utc, nullable=False)

    model_config = relationship("ModelConfig", back_populates="bindings")


# -- News / Hot Topics --------------------------------------------------------
class NewsArticle(Base):
    __tablename__ = "news_articles"

    id = Column(UUID(as_uuid=False), primary_key=True, default=new_uuid)
    source = Column(String(64), nullable=False)           # 财联社 / 华尔街见闻 / 新浪财经 etc.
    title = Column(String(512), nullable=False)
    content = Column(Text, nullable=True)
    summary = Column(Text, nullable=True)                 # AI summary
    url = Column(String(1024), nullable=True)
    published_at = Column(DateTime, nullable=False, index=True)
    collected_at = Column(DateTime, default=now_utc, nullable=False)
    related_stocks = Column(JSON, nullable=True)          # [{symbol, name, relevance}]
    topic_tags = Column(JSON, nullable=True)              # ["AI芯片", "新能源"]
    sentiment = Column(String(16), nullable=True)         # positive / neutral / negative
    heat_score = Column(Float, nullable=True)
    is_archived = Column(Boolean, default=False, nullable=False)

    __table_args__ = (
        Index("ix_news_published", "published_at"),
        Index("ix_news_heat", "heat_score"),
    )


class HotTopic(Base):
    __tablename__ = "hot_topics"

    id = Column(UUID(as_uuid=False), primary_key=True, default=new_uuid)
    topic_name = Column(String(256), nullable=False)
    description = Column(Text, nullable=True)
    heat_index = Column(Float, nullable=False, index=True)
    news_count = Column(Integer, default=0, nullable=False)
    related_stocks = Column(JSON, nullable=True)
    ai_conclusion = Column(Text, nullable=True)           # AC-001 AI生成结论
    generated_at = Column(DateTime, default=now_utc, nullable=False)
    expires_at = Column(DateTime, nullable=False)


# -- Research Reports ---------------------------------------------------------
class ResearchReport(Base):
    __tablename__ = "research_reports"

    id = Column(UUID(as_uuid=False), primary_key=True, default=new_uuid)
    broker = Column(String(128), nullable=False)          # 中金 / 中信 / 华泰 etc.
    title = Column(String(512), nullable=False)
    content = Column(Text, nullable=True)
    stock_symbol = Column(String(32), nullable=True, index=True)
    stock_name = Column(String(128), nullable=True)
    rating = Column(String(32), nullable=True)            # 买入 / 增持 / 中性 / 减持 / 卖出
    target_price = Column(Float, nullable=True)
    core_opinion = Column(Text, nullable=True)            # NLP extracted
    verification_result = Column(JSON, nullable=True)     # cross-check vs real data
    bottleneck_score = Column(Float, nullable=True)       # Serenity bottleneck filter
    published_at = Column(DateTime, nullable=False, index=True)
    collected_at = Column(DateTime, default=now_utc, nullable=False)

    __table_args__ = (
        Index("ix_report_broker_date", "broker", "published_at"),
    )


# -- Serenity Analysis Results ------------------------------------------------
class SerenityAnalysis(Base):
    __tablename__ = "serenity_analyses"

    id = Column(UUID(as_uuid=False), primary_key=True, default=new_uuid)
    symbol = Column(String(32), nullable=False, index=True)
    name = Column(String(128), nullable=False)
    step1_bom = Column(Text, nullable=True)               # Step 1: 供应链BOM
    step2_bottleneck = Column(Text, nullable=True)         # Step 2: 卡脖子审计
    step3_adversarial = Column(Text, nullable=True)        # Step 3: 逆向证伪
    step4_float = Column(Text, nullable=True)              # Step 4: 筹码结构
    step5_matrix = Column(Text, nullable=True)             # Step 5: 执行矩阵
    conditions_met = Column(JSON, nullable=True)           # 3 bottleneck conditions
    ai_conclusion = Column(Text, nullable=True)            # AC-004 AI结论
    generated_at = Column(DateTime, default=now_utc, nullable=False)


# -- AI Trend Analysis --------------------------------------------------------
class TrendAnalysis(Base):
    __tablename__ = "trend_analyses"

    id = Column(UUID(as_uuid=False), primary_key=True, default=new_uuid)
    symbol = Column(String(32), nullable=False, index=True)
    name = Column(String(128), nullable=False)
    frequency = Column(String(8), nullable=False)          # 5min / 15min / 30min / 60min / D / W / M
    trend_direction = Column(String(16), nullable=True)    # bullish / bearish / sideways
    support_level = Column(Float, nullable=True)
    resistance_level = Column(Float, nullable=True)
    pattern_detected = Column(String(128), nullable=True)
    confidence = Column(String(16), nullable=True)         # high / medium / low
    ai_conclusion = Column(Text, nullable=True)            # AC-002
    raw_indicators = Column(JSON, nullable=True)           # MACD/RSI/KDJ values
    generated_at = Column(DateTime, default=now_utc, nullable=False, index=True)


# -- Agent Decision Records ---------------------------------------------------
class AgentDecision(Base):
    __tablename__ = "agent_decisions"

    id = Column(UUID(as_uuid=False), primary_key=True, default=new_uuid)
    session_id = Column(String(64), nullable=False, index=True)
    symbol = Column(String(32), nullable=False, index=True)
    name = Column(String(128), nullable=False)

    # Phase 1: Analyst reports (JSON)
    analyst_reports = Column(JSON, nullable=True)

    # Phase 2: Debate conclusion
    debate_conclusion = Column(JSON, nullable=True)

    # Phase 3: Trade proposal
    trade_side = Column(String(16), nullable=True)         # buy / sell / hold
    trade_quantity = Column(Integer, nullable=True)
    stop_loss = Column(Float, nullable=True)
    take_profit = Column(Float, nullable=True)

    # Phase 4: Risk & Approval
    risk_assessment = Column(JSON, nullable=True)
    var_value = Column(Float, nullable=True)
    cvar_value = Column(Float, nullable=True)
    approval_status = Column(String(16), nullable=True)    # pending / approved / rejected
    approved_by = Column(String(64), nullable=True)
    approval_reason = Column(Text, nullable=True)          # AI-generated

    # Phase 5: Execution
    execution_status = Column(String(32), nullable=True)   # pending / executed / failed

    created_at = Column(DateTime, default=now_utc, nullable=False)
    updated_at = Column(DateTime, default=now_utc, onupdate=now_utc, nullable=False)


# -- KOL (Key Opinion Leader) -------------------------------------------------
class KOL(Base):
    __tablename__ = "kols"

    id = Column(UUID(as_uuid=False), primary_key=True, default=new_uuid)
    platform = Column(String(16), nullable=False, index=True)    # douyin / weibo
    nickname = Column(String(128), nullable=False)
    avatar_url = Column(String(1024), nullable=True)
    certification = Column(String(256), nullable=True)
    followers_count = Column(BigInteger, default=0, nullable=False)
    rank_position = Column(Integer, nullable=True)
    is_active = Column(Boolean, default=True, nullable=False)
    last_updated = Column(DateTime, default=now_utc, nullable=False)

    opinions = relationship("KOLOpinion", back_populates="kol", cascade="all, delete-orphan")


class KOLOpinion(Base):
    __tablename__ = "kol_opinions"

    id = Column(UUID(as_uuid=False), primary_key=True, default=new_uuid)
    kol_id = Column(UUID(as_uuid=False), ForeignKey("kols.id", ondelete="CASCADE"), nullable=False, index=True)
    platform = Column(String(16), nullable=False)
    content_url = Column(String(1024), nullable=True)
    summary = Column(Text, nullable=True)                     # AI summary ≤ 150 chars
    direction = Column(String(16), nullable=True)             # bullish / bearish / neutral
    raw_text = Column(Text, nullable=True)
    mentioned_stocks = Column(JSON, nullable=True)            # [{code, name, direction}]
    topic_tags = Column(JSON, nullable=True)
    heat_score = Column(Float, nullable=True)
    likes_count = Column(Integer, default=0, nullable=False)
    comments_count = Column(Integer, default=0, nullable=False)
    shares_count = Column(Integer, default=0, nullable=False)
    published_at = Column(DateTime, nullable=False, index=True)
    collected_at = Column(DateTime, default=now_utc, nullable=False)

    kol = relationship("KOL", back_populates="opinions")

    __table_args__ = (
        Index("ix_kol_opinion_published", "published_at"),
        Index("ix_kol_opinion_heat", "heat_score"),
    )


class KOLConsensus(Base):
    __tablename__ = "kol_consensus"

    id = Column(UUID(as_uuid=False), primary_key=True, default=new_uuid)
    summary_date = Column(Date, nullable=False, index=True)
    topic = Column(String(256), nullable=True)
    bullish_stocks = Column(JSON, nullable=True)
    bearish_stocks = Column(JSON, nullable=True)
    mention_count = Column(Integer, default=0, nullable=False)
    consensus_score = Column(Float, nullable=True)
    ai_summary = Column(Text, nullable=True)                  # AC-008
    generated_at = Column(DateTime, default=now_utc, nullable=False)


# -- AI Analysis Conclusions (unified) ----------------------------------------
class AnalysisConclusion(Base):
    """Unified storage for all AI-generated conclusions."""
    __tablename__ = "analysis_conclusions"

    id = Column(UUID(as_uuid=False), primary_key=True, default=new_uuid)
    scene = Column(String(32), nullable=False, index=True)    # hot_topic / trend / report / serenity / decision / kol
    reference_id = Column(String(64), nullable=True, index=True)
    symbol = Column(String(32), nullable=True, index=True)
    conclusion_md = Column(Text, nullable=False)              # Structured Markdown
    model_used = Column(String(128), nullable=True)
    tokens_used = Column(Integer, nullable=True)
    generated_at = Column(DateTime, default=now_utc, nullable=False)

    __table_args__ = (
        Index("ix_conclusion_scene_ref", "scene", "reference_id"),
    )


# -- System Audit Log ---------------------------------------------------------
class AuditLog(Base):
    __tablename__ = "audit_logs"

    id = Column(UUID(as_uuid=False), primary_key=True, default=new_uuid)
    user_id = Column(UUID(as_uuid=False), nullable=True, index=True)
    action = Column(String(128), nullable=False)
    resource = Column(String(128), nullable=True)
    detail = Column(JSON, nullable=True)
    ip_address = Column(String(64), nullable=True)
    correlation_id = Column(String(64), nullable=True, index=True)
    created_at = Column(DateTime, default=now_utc, nullable=False)
