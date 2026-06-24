# AI 股票分析平台 (AI Stock Analysis Platform)

> **版本**：v1.4.0 | **更新日期**：2026-06-18
> 基于 LangGraph 多智能体架构的 AI 股票分析系统，为个人投资者提供机构级别的深度分析能力。

## 核心功能

| 功能模块 | 需求编号 | 说明 |
|----------|----------|------|
| 多源数据接入与故障转移 | DS-001~005 | 6 大数据源自动级联切换，故障 < 2s |
| 实时热点追踪播报 | HT-001~004 | NLP 提取热点，15 分钟更新，72h 时效 |
| AI 趋势分析与预测 | TA-001~005 | AI 大模型分析技术指标，多周期交叉验证 |
| Serenity 深度分析 | SA-001~003 | 5 步瓶颈猎头投研框架 |
| 多智能体协作决策 | AGENT-* | 8 个 Agent 角色，5 阶段工作流 |
| AI 模型配置管理 | AI-001~007 | OpenAI/DeepSeek/OpenRouter 多平台 UI 配置 |
| AI 分析结论生成 | AC-001~008 | 所有分析场景自动生成 AI 结论 |
| 用户关注列表 | WL-001~005 | 缓存加速，分组管理 |
| 财经大V观点聚合 | KV-001~005 | 抖音+微博 Top 100 大V 48h 观点 |

## 技术栈

- **语言**：Python 3.11+
- **Web 框架**：FastAPI + Uvicorn
- **Agent 框架**：LangGraph + LangChain
- **AI 网关**：LiteLLM（统一 100+ LLM 路由）
- **数据库**：PostgreSQL 15
- **缓存**：Redis 7
- **任务队列**：Celery + Redis
- **技术指标**：TA-Lib
- **数据源**：AkShare / Tushare / Yahoo Finance

## 项目结构

```
stock_analysis/
├── app/
│   ├── main.py              # FastAPI 应用入口
│   ├── config.py            # 配置管理
│   ├── database.py          # 数据库连接
│   ├── models.py            # ORM 数据模型（13 张表）
│   ├── schemas.py           # Agent 消息协议
│   ├── cache.py             # Redis 缓存
│   ├── adapters.py          # 多源数据适配器 + 故障转移
│   ├── model_gateway.py     # AI 模型网关 (LiteLLM 封装)
│   ├── indicators.py        # TA-Lib 技术指标计算
│   ├── engines.py           # 分析引擎（趋势/Serenity/热点/大V）
│   ├── agents.py            # 8 Agent + 5 阶段工作流
│   ├── tasks.py             # Celery 定时任务
│   ├── celery_app.py        # Celery 配置
│   └── api/                 # API 路由
│       ├── auth.py          # 认证 API
│       ├── data.py          # 数据 API (K线/行情/基本面)
│       ├── models.py        # AI 模型配置 API
│       ├── analysis.py      # 分析 API (趋势/Serenity/研报)
│       ├── agent.py         # Agent 决策 API
│       ├── watchlist.py     # 关注列表 API
│       ├── news.py          # 新闻热点 API
│       └── kol.py           # 大V观点 API
├── db/
│   ├── init_db.sql          # 数据库初始化脚本
│   └── seed_data.sql        # 种子数据
├── monitoring/
│   └── prometheus.yml       # Prometheus 配置
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── .env.example
```

## 快速开始

### 1. 环境准备

```bash
# 复制环境配置
cp .env.example .env

# 编辑 .env，填入你的 API Key
vim .env
```

### 2. Docker 一键启动

```bash
# 构建并启动所有服务
docker compose up -d

# 查看服务状态
docker compose ps

# 查看日志
docker compose logs -f api
```

### 3. 本地开发（不用 Docker）

```bash
# 安装依赖
pip install -r requirements.txt

# 启动 PostgreSQL 和 Redis（可用 Docker）
docker run -d --name stock_pg -p 5432:5432 \
  -e POSTGRES_USER=stock_user -e POSTGRES_PASSWORD=stock_pass \
  -e POSTGRES_DB=stock_analysis postgres:15-alpine

docker run -d --name stock_redis -p 6379:6379 redis:7-alpine

# 初始化数据库
psql -U stock_user -d stock_analysis -f db/init_db.sql
psql -U stock_user -d stock_analysis -f db/seed_data.sql

# 启动 API
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# 启动 Celery Worker（新终端）
celery -A app.celery_app worker --loglevel=info

# 启动 Celery Beat 定时任务（新终端）
celery -A app.celery_app beat --loglevel=info
```

### 4. 访问服务

| 服务 | 地址 | 说明 |
|------|------|------|
| API 文档 | http://localhost:8000/docs | Swagger UI |
| ReDoc | http://localhost:8000/redoc | ReDoc 文档 |
| 健康检查 | http://localhost:8000/health | 服务状态 |
| Grafana | http://localhost:3000 | 监控面板（需启用 monitoring profile） |

## 默认账号

- **用户名**：admin
- **密码**：admin123

## API 概览

| 模块 | 端点 | 说明 |
|------|------|------|
| 认证 | `POST /api/v1/auth/register` | 注册 |
| | `POST /api/v1/auth/login` | 登录 |
| 数据 | `GET /api/v1/data/kline` | K线数据 |
| | `GET /api/v1/data/quote` | 实时行情 |
| 模型 | `GET /api/v1/models` | 模型列表 |
| | `POST /api/v1/models` | 新增模型 |
| | `POST /api/v1/models/{id}/test` | 测试连接 |
| 分析 | `POST /api/v1/analysis/trend` | AI 趋势分析 |
| | `POST /api/v1/analysis/serenity` | Serenity 5步分析 |
| Agent | `POST /api/v1/agent/run` | 运行 5 阶段工作流 |
| 关注 | `GET /api/v1/watchlist/items` | 关注列表 |
| 新闻 | `GET /api/v1/news` | 新闻列表 |
| | `GET /api/v1/news/hot-topics` | 热点排行 |
| 大V | `GET /api/v1/kol/opinions` | 大V观点 |
| | `GET /api/v1/kol/consensus` | AI 共识摘要 |

## 定时任务

| 任务 | 频率 | 说明 |
|------|------|------|
| `crawl_news` | 每 15 分钟 | 新闻采集 |
| `compute_hot_topics` | 每 15 分钟 | 热点计算 |
| `crawl_kol_opinions` | 每 30 分钟 | 大V观点采集 |
| `generate_kol_consensus` | 每日 18:00 | 大V共识生成 |
| `cleanup_expired_data` | 每日 02:00 | 过期数据清理 |
| `update_watchlist_cache` | 每日 18:00 | 关注列表缓存更新 |

## 文档

- [需求文档](../需求文档.md)
- [系统架构文档](../系统架构文档.md)
- [部署文档](部署文档.md)

## License

MIT
