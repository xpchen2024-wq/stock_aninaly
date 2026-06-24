-- ============================================================================
-- AI Stock Analysis Platform - Seed Data
-- Run: psql -U stock_user -d stock_analysis -f db/seed_data.sql
-- ============================================================================

-- ============================================================================
-- 1. Default Admin User (password: admin123)
-- ============================================================================
INSERT INTO users (id, username, email, hashed_password, is_active, is_superuser)
VALUES (
    'a0000000-0000-0000-0000-000000000001',
    'admin',
    'admin@aistock.local',
    -- bcrypt hash of 'admin123'
    '$2b$12$LJ3m4ys3Lk0TSwHlvX0LaObFJiYiXpPLFSVYQvMLkRMCvVNjGBqSi',
    TRUE,
    TRUE
) ON CONFLICT (username) DO NOTHING;

-- ============================================================================
-- 2. Default Watchlist Group
-- ============================================================================
INSERT INTO watchlist_groups (id, user_id, name, description, sort_order)
VALUES (
    'b0000000-0000-0000-0000-000000000001',
    'a0000000-0000-0000-0000-000000000001',
    '默认自选',
    '默认关注列表',
    0
) ON CONFLICT DO NOTHING;

-- ============================================================================
-- 3. Sample Model Configs
-- ============================================================================
INSERT INTO model_configs (id, name, platform, base_url, api_key_encrypted, model_name, max_tokens, temperature, is_default, is_enabled, status)
VALUES
(
    'c0000000-0000-0000-0000-000000000001',
    'OpenCode Zen (Default)',
    'opencodezen',
    'https://opencode.ai/zen/v1',
    'ENCRYPTED_PLACEHOLDER',
    'opencode/gpt-5.5',
    128000,
    0.7,
    TRUE,
    TRUE,
    'active'
),
(
    'c0000000-0000-0000-0000-000000000002',
    'DeepSeek Chat',
    'deepseek',
    'https://api.deepseek.com/v1',
    'ENCRYPTED_PLACEHOLDER',
    'deepseek-chat',
    4096,
    0.7,
    FALSE,
    FALSE,
    'active'
),
(
    'c0000000-0000-0000-0000-000000000003',
    'GPT-4o (Deep Analysis)',
    'openai',
    'https://api.openai.com/v1',
    'ENCRYPTED_PLACEHOLDER',
    'gpt-4o',
    8192,
    0.5,
    FALSE,
    FALSE,
    'active'
)
ON CONFLICT DO NOTHING;

-- ============================================================================
-- 4. Sample KOL Data (Top 10 Douyin + Top 10 Weibo)
-- ============================================================================
-- Douyin KOLs
INSERT INTO kols (id, platform, nickname, certification, followers_count, rank_position)
VALUES
('d0000000-0000-0000-0000-000000000001', 'douyin', '财经林哥', '认证财经博主', 3280000, 1),
('d0000000-0000-0000-0000-000000000002', 'douyin', '量化解盘', 'CFA持证人', 1860000, 2),
('d0000000-0000-0000-0000-000000000003', 'douyin', '老王聊财经', '财经博主', 5120000, 3),
('d0000000-0000-0000-0000-000000000004', 'douyin', '趋势猎手', '认证财经博主', 2450000, 4),
('d0000000-0000-0000-0000-000000000005', 'douyin', '投资有道', '财经博主', 1680000, 5),
('d0000000-0000-0000-0000-000000000006', 'douyin', '股市老炮', '认证财经博主', 980000, 6),
('d0000000-0000-0000-0000-000000000007', 'douyin', '技术流涛哥', '财经博主', 1250000, 7),
('d0000000-0000-0000-0000-000000000008', 'douyin', '财经小白说', '认证财经博主', 890000, 8),
('d0000000-0000-0000-0000-000000000009', 'douyin', '深度观察室', '财经博主', 1560000, 9),
('d0000000-0000-0000-0000-000000000010', 'douyin', '投资笔记', '认证财经博主', 2100000, 10)
ON CONFLICT DO NOTHING;

-- Weibo KOLs
INSERT INTO kols (id, platform, nickname, certification, followers_count, rank_position)
VALUES
('d0000000-0000-0000-0000-000000000011', 'weibo', '但斌', 'V认证 · 深圳东方港湾董事长', 12800000, 1),
('d0000000-0000-0000-0000-000000000012', 'weibo', '洪灏', 'V认证 · 交银国际董事总经理', 6800000, 2),
('d0000000-0000-0000-0000-000000000013', 'weibo', '李大霄', 'V认证 · 英大证券首席经济学家', 10200000, 3),
('d0000000-0000-0000-0000-000000000014', 'weibo', '任泽平', 'V认证 · 经济学家', 5200000, 4),
('d0000000-0000-0000-0000-000000000015', 'weibo', '林园', 'V认证 · 林园投资董事长', 3500000, 5),
('d0000000-0000-0000-0000-000000000016', 'weibo', '吴晓波', 'V认证 · 财经作家', 4500000, 6),
('d0000000-0000-0000-0000-000000000017', 'weibo', '叶檀', 'V认证 · 财经评论员', 2800000, 7),
('d0000000-0000-0000-0000-000000000018', 'weibo', '管清友', 'V认证 · 如是金融研究院院长', 1900000, 8),
('d0000000-0000-0000-0000-000000000019', 'weibo', '马光远', 'V认证 · 经济学家', 2300000, 9),
('d0000000-0000-0000-0000-000000000020', 'weibo', '水皮', 'V认证 · 华夏时报总编辑', 1600000, 10)
ON CONFLICT DO NOTHING;

-- ============================================================================
-- 5. Sample News Articles
-- ============================================================================
INSERT INTO news_articles (id, source, title, content, summary, url, published_at, related_stocks, topic_tags, sentiment, heat_score)
VALUES
(
    'e0000000-0000-0000-0000-000000000001',
    '财联社',
    'AI芯片国产替代加速：寒武纪思元590即将量产',
    '据产业链消息，寒武纪新一代AI训练芯片思元590已进入量产准备阶段...',
    '寒武纪思元590即将量产，性能对标A100差距缩小至30%以内，国产AI芯片替代进程加速。',
    'https://www.cls.cn/detail/example1',
    NOW() - INTERVAL '2 hours',
    '[{"symbol":"688256","name":"寒武纪","relevance":0.95},{"symbol":"688041","name":"海光信息","relevance":0.78}]',
    '["AI芯片","国产替代","半导体"]',
    'positive',
    92.5
),
(
    'e0000000-0000-0000-0000-000000000002',
    '华尔街见闻',
    '央行降准0.25个百分点 释放长期资金约5000亿元',
    '中国人民银行决定于6月17日下调金融机构存款准备金率0.25个百分点...',
    '央行宣布降准0.25个百分点，释放约5000亿长期资金，利好银行地产板块。',
    'https://wallstreetcn.com/articles/example2',
    NOW() - INTERVAL '5 hours',
    '[{"symbol":"600036","name":"招商银行","relevance":0.82},{"symbol":"000002","name":"万科A","relevance":0.75}]',
    '["降准","货币政策","银行","地产"]',
    'positive',
    88.3
),
(
    'e0000000-0000-0000-0000-000000000003',
    '新浪财经',
    '光伏产业链价格企稳 组件开工率回升',
    '本周光伏产业链价格出现企稳迹象，硅料价格止跌...',
    '光伏产业链价格企稳，硅料止跌组件开工率回升，龙头估值处于历史底部。',
    'https://finance.sina.com.cn/example3',
    NOW() - INTERVAL '8 hours',
    '[{"symbol":"601012","name":"隆基绿能","relevance":0.91},{"symbol":"600438","name":"通威股份","relevance":0.85}]',
    '["光伏","新能源","硅料"]',
    'positive',
    76.8
)
ON CONFLICT DO NOTHING;

-- ============================================================================
-- 6. Sample Hot Topics
-- ============================================================================
INSERT INTO hot_topics (id, topic_name, description, heat_index, news_count, related_stocks, expires_at)
VALUES
(
    'f0000000-0000-0000-0000-000000000001',
    'AI芯片国产替代',
    '寒武纪思元590量产在即，国产AI芯片加速追赶，产业链受益标的受关注',
    94.2,
    15,
    '[{"symbol":"688256","name":"寒武纪"},{"symbol":"688041","name":"海光信息"},{"symbol":"002049","name":"紫光国微"}]',
    NOW() + INTERVAL '24 hours'
),
(
    'f0000000-0000-0000-0000-000000000002',
    '央行降准',
    '央行宣布降准0.25个百分点，释放流动性约5000亿元，市场解读为稳增长信号',
    88.3,
    22,
    '[{"symbol":"600036","name":"招商银行"},{"symbol":"601166","name":"兴业银行"},{"symbol":"000002","name":"万科A"}]',
    NOW() + INTERVAL '24 hours'
),
(
    'f0000000-0000-0000-0000-000000000003',
    '光伏底部信号',
    '硅料价格企稳，组件开工率回升，行业龙头估值处于历史低位',
    76.8,
    12,
    '[{"symbol":"601012","name":"隆基绿能"},{"symbol":"600438","name":"通威股份"},{"symbol":"002459","name":"晶澳科技"}]',
    NOW() + INTERVAL '24 hours'
)
ON CONFLICT DO NOTHING;
