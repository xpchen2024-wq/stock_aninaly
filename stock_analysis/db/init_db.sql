-- ============================================================================
-- AI Stock Analysis Platform - Database Initialization Script
-- PostgreSQL 15+
-- ============================================================================

-- Create database and user (run as superuser)
-- CREATE USER stock_user WITH PASSWORD 'stock_pass';
-- CREATE DATABASE stock_analysis OWNER stock_user;
-- GRANT ALL PRIVILEGES ON DATABASE stock_analysis TO stock_user;

-- ============================================================================
-- Extensions
-- ============================================================================
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- ============================================================================
-- 1. Users & Authentication
-- ============================================================================
CREATE TABLE IF NOT EXISTS users (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    username        VARCHAR(64) UNIQUE NOT NULL,
    email           VARCHAR(255) UNIQUE NOT NULL,
    hashed_password VARCHAR(255) NOT NULL,
    is_active       BOOLEAN DEFAULT TRUE NOT NULL,
    is_superuser    BOOLEAN DEFAULT FALSE NOT NULL,
    created_at      TIMESTAMPTZ DEFAULT NOW() NOT NULL,
    updated_at      TIMESTAMPTZ DEFAULT NOW() NOT NULL
);

-- ============================================================================
-- 2. Watchlist
-- ============================================================================
CREATE TABLE IF NOT EXISTS watchlist_groups (
    id          UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id     UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    name        VARCHAR(64) NOT NULL,
    description VARCHAR(255),
    sort_order  INTEGER DEFAULT 0 NOT NULL,
    created_at  TIMESTAMPTZ DEFAULT NOW() NOT NULL
);

CREATE TABLE IF NOT EXISTS watchlist_items (
    id          UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id     UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    group_id    UUID REFERENCES watchlist_groups(id) ON DELETE SET NULL,
    symbol      VARCHAR(32) NOT NULL,
    name        VARCHAR(128) NOT NULL,
    market      VARCHAR(16),          -- SH / SZ / HK / US
    notes       TEXT,
    added_at    TIMESTAMPTZ DEFAULT NOW() NOT NULL,
    UNIQUE (user_id, symbol)
);
CREATE INDEX IF NOT EXISTS ix_watchlist_items_symbol ON watchlist_items(symbol);

-- ============================================================================
-- 3. K-Line Cache (watchlist only, 90 days)
-- ============================================================================
CREATE TABLE IF NOT EXISTS kline_cache (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    symbol          VARCHAR(32) NOT NULL,
    frequency       VARCHAR(8) NOT NULL,       -- D / W / M / 5min / 15min / 30min / 60min
    trade_date      DATE NOT NULL,
    open            DOUBLE PRECISION NOT NULL,
    high            DOUBLE PRECISION NOT NULL,
    low             DOUBLE PRECISION NOT NULL,
    close           DOUBLE PRECISION NOT NULL,
    volume          BIGINT NOT NULL,
    amount          DOUBLE PRECISION,
    turnover_rate   DOUBLE PRECISION,
    cached_at       TIMESTAMPTZ DEFAULT NOW() NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_kline_symbol_freq_date ON kline_cache(symbol, frequency, trade_date);

-- ============================================================================
-- 4. Stock Fundamentals
-- ============================================================================
CREATE TABLE IF NOT EXISTS stock_fundamentals (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    symbol          VARCHAR(32) NOT NULL,
    name            VARCHAR(128) NOT NULL,
    market          VARCHAR(16),
    industry        VARCHAR(64),
    sector          VARCHAR(64),
    pe_ratio        DOUBLE PRECISION,
    pb_ratio        DOUBLE PRECISION,
    market_cap      DOUBLE PRECISION,       -- 总市值（亿元）
    circulating_cap DOUBLE PRECISION,       -- 流通市值（亿元）
    total_shares    DOUBLE PRECISION,       -- 总股本（亿股）
    revenue_yoy     DOUBLE PRECISION,       -- 营收同比 %
    profit_yoy      DOUBLE PRECISION,       -- 利润同比 %
    roe             DOUBLE PRECISION,
    debt_ratio      DOUBLE PRECISION,
    report_date     DATE,
    updated_at      TIMESTAMPTZ DEFAULT NOW() NOT NULL,
    UNIQUE (symbol, report_date)
);
CREATE INDEX IF NOT EXISTS ix_fundamental_symbol ON stock_fundamentals(symbol);

-- ============================================================================
-- 5. AI Model Configuration
-- ============================================================================
CREATE TABLE IF NOT EXISTS model_configs (
    id                  UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name                VARCHAR(128) NOT NULL,
    platform            VARCHAR(32) NOT NULL,   -- openai / deepseek / openrouter / custom
    base_url            VARCHAR(512),
    api_key_encrypted   TEXT NOT NULL,           -- AES-256 encrypted
    model_name          VARCHAR(128) NOT NULL,
    max_tokens          INTEGER DEFAULT 4096 NOT NULL,
    temperature         DOUBLE PRECISION DEFAULT 0.7 NOT NULL,
    is_default          BOOLEAN DEFAULT FALSE NOT NULL,
    is_enabled          BOOLEAN DEFAULT TRUE NOT NULL,
    extra_headers       JSONB,
    created_at          TIMESTAMPTZ DEFAULT NOW() NOT NULL,
    updated_at          TIMESTAMPTZ DEFAULT NOW() NOT NULL
);

CREATE TABLE IF NOT EXISTS agent_model_bindings (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    model_config_id UUID NOT NULL REFERENCES model_configs(id) ON DELETE CASCADE,
    agent_id        VARCHAR(64),              -- AGENT-ANL-01, AGENT-ANL-02, etc.
    scene           VARCHAR(64),              -- hot_track / trend_analysis / report / serenity / decision
    is_active       BOOLEAN DEFAULT TRUE NOT NULL,
    created_at      TIMESTAMPTZ DEFAULT NOW() NOT NULL
);

-- ============================================================================
-- 6. News & Hot Topics
-- ============================================================================
CREATE TABLE IF NOT EXISTS news_articles (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    source          VARCHAR(64) NOT NULL,      -- 财联社 / 华尔街见闻 etc.
    title           VARCHAR(512) NOT NULL,
    content         TEXT,
    summary         TEXT,                      -- AI summary
    url             VARCHAR(1024),
    published_at    TIMESTAMPTZ NOT NULL,
    collected_at    TIMESTAMPTZ DEFAULT NOW() NOT NULL,
    related_stocks  JSONB,                     -- [{symbol, name, relevance}]
    topic_tags      JSONB,                     -- ["AI芯片", "新能源"]
    sentiment       VARCHAR(16),               -- positive / neutral / negative
    heat_score      DOUBLE PRECISION,
    is_archived     BOOLEAN DEFAULT FALSE NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_news_published ON news_articles(published_at);
CREATE INDEX IF NOT EXISTS ix_news_heat ON news_articles(heat_score);

CREATE TABLE IF NOT EXISTS hot_topics (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    topic_name      VARCHAR(256) NOT NULL,
    description     TEXT,
    heat_index      DOUBLE PRECISION NOT NULL,
    news_count      INTEGER DEFAULT 0 NOT NULL,
    related_stocks  JSONB,
    ai_conclusion   TEXT,                      -- AC-001
    generated_at    TIMESTAMPTZ DEFAULT NOW() NOT NULL,
    expires_at      TIMESTAMPTZ NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_hot_topic_heat ON hot_topics(heat_index);

-- ============================================================================
-- 7. Research Reports
-- ============================================================================
CREATE TABLE IF NOT EXISTS research_reports (
    id                  UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    broker              VARCHAR(128) NOT NULL,  -- 中金 / 中信 / 华泰
    title               VARCHAR(512) NOT NULL,
    content             TEXT,
    stock_symbol        VARCHAR(32),
    stock_name          VARCHAR(128),
    rating              VARCHAR(32),            -- 买入 / 增持 / 中性 / 减持 / 卖出
    target_price        DOUBLE PRECISION,
    core_opinion        TEXT,                   -- NLP extracted
    verification_result JSONB,                  -- cross-check vs real data
    bottleneck_score    DOUBLE PRECISION,       -- Serenity bottleneck filter
    published_at        TIMESTAMPTZ NOT NULL,
    collected_at        TIMESTAMPTZ DEFAULT NOW() NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_report_broker_date ON research_reports(broker, published_at);
CREATE INDEX IF NOT EXISTS ix_report_symbol ON research_reports(stock_symbol);

-- ============================================================================
-- 8. Serenity Analysis Results
-- ============================================================================
CREATE TABLE IF NOT EXISTS serenity_analyses (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    symbol          VARCHAR(32) NOT NULL,
    name            VARCHAR(128) NOT NULL,
    step1_bom       TEXT,                      -- Step 1: 供应链BOM
    step2_bottleneck TEXT,                     -- Step 2: 卡脖子审计
    step3_adversarial TEXT,                    -- Step 3: 逆向证伪
    step4_float     TEXT,                      -- Step 4: 筹码结构
    step5_matrix    TEXT,                      -- Step 5: 执行矩阵
    conditions_met  JSONB,                     -- 3 bottleneck conditions
    ai_conclusion   TEXT,                      -- AC-004
    generated_at    TIMESTAMPTZ DEFAULT NOW() NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_serenity_symbol ON serenity_analyses(symbol);

-- ============================================================================
-- 9. AI Trend Analysis
-- ============================================================================
CREATE TABLE IF NOT EXISTS trend_analyses (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    symbol          VARCHAR(32) NOT NULL,
    name            VARCHAR(128) NOT NULL,
    frequency       VARCHAR(8) NOT NULL,        -- 5min / 15min / 30min / 60min / D / W / M
    trend_direction VARCHAR(16),                -- bullish / bearish / sideways
    support_level   DOUBLE PRECISION,
    resistance_level DOUBLE PRECISION,
    pattern_detected VARCHAR(128),
    confidence      VARCHAR(16),                -- high / medium / low
    ai_conclusion   TEXT,                       -- AC-002
    raw_indicators  JSONB,                      -- MACD/RSI/KDJ values
    generated_at    TIMESTAMPTZ DEFAULT NOW() NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_trend_symbol_freq ON trend_analyses(symbol, frequency, generated_at);

-- ============================================================================
-- 10. Agent Decision Records
-- ============================================================================
CREATE TABLE IF NOT EXISTS agent_decisions (
    id                  UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    session_id          VARCHAR(64) NOT NULL,
    symbol              VARCHAR(32) NOT NULL,
    name                VARCHAR(128) NOT NULL,

    -- Phase 1: Analyst reports
    analyst_reports     JSONB,

    -- Phase 2: Debate conclusion
    debate_conclusion   JSONB,

    -- Phase 3: Trade proposal
    trade_side          VARCHAR(16),            -- buy / sell / hold
    trade_quantity      INTEGER,
    stop_loss           DOUBLE PRECISION,
    take_profit         DOUBLE PRECISION,

    -- Phase 4: Risk & Approval
    risk_assessment     JSONB,
    var_value           DOUBLE PRECISION,
    cvar_value          DOUBLE PRECISION,
    approval_status     VARCHAR(16),            -- pending / approved / rejected
    approved_by         VARCHAR(64),
    approval_reason     TEXT,                   -- AI-generated

    -- Phase 5: Execution
    execution_status    VARCHAR(32),            -- pending / executed / failed

    created_at          TIMESTAMPTZ DEFAULT NOW() NOT NULL,
    updated_at          TIMESTAMPTZ DEFAULT NOW() NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_decision_session ON agent_decisions(session_id);
CREATE INDEX IF NOT EXISTS ix_decision_symbol ON agent_decisions(symbol);

-- ============================================================================
-- 11. KOL (Key Opinion Leaders)
-- ============================================================================
CREATE TABLE IF NOT EXISTS kols (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    platform        VARCHAR(16) NOT NULL,       -- douyin / weibo
    nickname        VARCHAR(128) NOT NULL,
    avatar_url      VARCHAR(1024),
    certification   VARCHAR(256),
    followers_count BIGINT DEFAULT 0 NOT NULL,
    rank_position   INTEGER,
    is_active       BOOLEAN DEFAULT TRUE NOT NULL,
    last_updated    TIMESTAMPTZ DEFAULT NOW() NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_kol_platform ON kols(platform);

CREATE TABLE IF NOT EXISTS kol_opinions (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    kol_id          UUID NOT NULL REFERENCES kols(id) ON DELETE CASCADE,
    platform        VARCHAR(16) NOT NULL,
    content_url     VARCHAR(1024),
    summary         TEXT,                       -- AI summary ≤ 150 chars
    direction       VARCHAR(16),                -- bullish / bearish / neutral
    raw_text        TEXT,
    mentioned_stocks JSONB,                     -- [{code, name, direction}]
    topic_tags      JSONB,
    heat_score      DOUBLE PRECISION,
    likes_count     INTEGER DEFAULT 0 NOT NULL,
    comments_count  INTEGER DEFAULT 0 NOT NULL,
    shares_count    INTEGER DEFAULT 0 NOT NULL,
    published_at    TIMESTAMPTZ NOT NULL,
    collected_at    TIMESTAMPTZ DEFAULT NOW() NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_kol_opinion_published ON kol_opinions(published_at);
CREATE INDEX IF NOT EXISTS ix_kol_opinion_heat ON kol_opinions(heat_score);
CREATE INDEX IF NOT EXISTS ix_kol_opinion_kol_id ON kol_opinions(kol_id);

CREATE TABLE IF NOT EXISTS kol_consensus (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    summary_date    DATE NOT NULL,
    topic           VARCHAR(256),
    bullish_stocks  JSONB,
    bearish_stocks  JSONB,
    mention_count   INTEGER DEFAULT 0 NOT NULL,
    consensus_score DOUBLE PRECISION,
    ai_summary      TEXT,                       -- AC-008
    generated_at    TIMESTAMPTZ DEFAULT NOW() NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_kol_consensus_date ON kol_consensus(summary_date);

-- ============================================================================
-- 12. AI Analysis Conclusions (unified)
-- ============================================================================
CREATE TABLE IF NOT EXISTS analysis_conclusions (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    scene           VARCHAR(32) NOT NULL,        -- hot_topic / trend / report / serenity / decision / kol
    reference_id    VARCHAR(64),
    symbol          VARCHAR(32),
    conclusion_md   TEXT NOT NULL,               -- Structured Markdown
    model_used      VARCHAR(128),
    tokens_used     INTEGER,
    generated_at    TIMESTAMPTZ DEFAULT NOW() NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_conclusion_scene_ref ON analysis_conclusions(scene, reference_id);
CREATE INDEX IF NOT EXISTS ix_conclusion_symbol ON analysis_conclusions(symbol);

-- ============================================================================
-- 13. Audit Logs
-- ============================================================================
CREATE TABLE IF NOT EXISTS audit_logs (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id         UUID,
    action          VARCHAR(128) NOT NULL,
    resource        VARCHAR(128),
    detail          JSONB,
    ip_address      VARCHAR(64),
    correlation_id  VARCHAR(64),
    created_at      TIMESTAMPTZ DEFAULT NOW() NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_audit_correlation ON audit_logs(correlation_id);
CREATE INDEX IF NOT EXISTS ix_audit_user ON audit_logs(user_id);
CREATE INDEX IF NOT EXISTS ix_audit_created ON audit_logs(created_at);

-- ============================================================================
-- Auto-update updated_at trigger function
-- ============================================================================
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Apply triggers
CREATE TRIGGER update_users_updated_at BEFORE UPDATE ON users
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_model_configs_updated_at BEFORE UPDATE ON model_configs
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_agent_decisions_updated_at BEFORE UPDATE ON agent_decisions
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_stock_fundamentals_updated_at BEFORE UPDATE ON stock_fundamentals
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- ============================================================================
-- Cleanup: Remove expired data
-- ============================================================================
CREATE OR REPLACE FUNCTION cleanup_expired_data()
RETURNS void AS $$
BEGIN
    -- Archive news older than 72 hours
    UPDATE news_articles SET is_archived = TRUE
    WHERE is_archived = FALSE
      AND published_at < NOW() - INTERVAL '72 hours';

    -- Delete KOL opinions older than 30 days
    DELETE FROM kol_opinions
    WHERE published_at < NOW() - INTERVAL '30 days';

    -- Delete K-Line cache older than 90 days
    DELETE FROM kline_cache
    WHERE trade_date < CURRENT_DATE - INTERVAL '90 days';

    -- Delete old hot topics
    DELETE FROM hot_topics WHERE expires_at < NOW();

    -- Delete old audit logs (keep 90 days)
    DELETE FROM audit_logs WHERE created_at < NOW() - INTERVAL '90 days';
END;
$$ LANGUAGE plpgsql;

-- ============================================================================
-- Completion
-- ============================================================================
-- Run: psql -U stock_user -d stock_analysis -f db/init_db.sql
