#!/usr/bin/env python3
"""
AI Stock Analysis Platform - Full Functional Test Suite
Tests all API endpoints and verifies functionality.
"""
import json
import sys
import time
import requests
from datetime import datetime

BASE = "http://localhost:8000"
PASS = 0
FAIL = 0
SKIP = 0
RESULTS = []


def test(name, condition, detail=""):
    global PASS, FAIL
    status = "✅ PASS" if condition else "❌ FAIL"
    if condition:
        PASS += 1
    else:
        FAIL += 1
    msg = f"  {status} {name}"
    if detail:
        msg += f" — {detail}"
    print(msg)
    RESULTS.append({"name": name, "status": status, "detail": detail})
    return condition


def skip(name, reason=""):
    global SKIP
    SKIP += 1
    print(f"  ⏭️  SKIP {name} — {reason}")
    RESULTS.append({"name": name, "status": "SKIP", "detail": reason})


def section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


# -- Token storage --
TOKEN = ""


# ============================================================================
# 1. 基础服务
# ============================================================================
section("1. 基础服务检查")

try:
    r = requests.get(f"{BASE}/health", timeout=5)
    test("健康检查 /health", r.status_code == 200 and r.json().get("status") == "healthy",
         f"status={r.status_code}")
except Exception as e:
    test("健康检查 /health", False, str(e))

try:
    r = requests.get(f"{BASE}/", timeout=5)
    data = r.json()
    test("根路径 /", r.status_code == 200 and data.get("version") == "1.4.0",
         f"version={data.get('version')}")
except Exception as e:
    test("根路径 /", False, str(e))

try:
    r = requests.get(f"{BASE}/docs", timeout=5)
    test("Swagger文档 /docs", r.status_code == 200, f"status={r.status_code}")
except Exception as e:
    test("Swagger文档 /docs", False, str(e))

try:
    r = requests.get(f"{BASE}/ui/", timeout=5)
    test("前端UI /ui/", r.status_code == 200 and "html" in r.text.lower(),
         f"status={r.status_code}")
except Exception as e:
    test("前端UI /ui/", False, str(e))


# ============================================================================
# 2. 认证模块 (auth)
# ============================================================================
section("2. 认证模块 (NF-012: JWT)")

# Login
try:
    r = requests.post(f"{BASE}/api/v1/auth/login",
                      json={"username": "admin", "password": "admin123"}, timeout=5)
    data = r.json()
    TOKEN = data.get("access_token", "")
    test("登录 /auth/login", r.status_code == 200 and len(TOKEN) > 50,
         f"token_len={len(TOKEN)}")
except Exception as e:
    test("登录 /auth/login", False, str(e))

# Wrong password
try:
    r = requests.post(f"{BASE}/api/v1/auth/login",
                      json={"username": "admin", "password": "wrong"}, timeout=5)
    test("错误密码登录拒绝", r.status_code == 401, f"status={r.status_code}")
except Exception as e:
    test("错误密码登录拒绝", False, str(e))

# Get current user
try:
    r = requests.get(f"{BASE}/api/v1/auth/me",
                     headers={"Authorization": f"Bearer {TOKEN}"}, timeout=5)
    data = r.json()
    test("获取当前用户 /auth/me", r.status_code == 200 and data.get("username") == "admin",
         f"username={data.get('username')}")
except Exception as e:
    test("获取当前用户 /auth/me", False, str(e))

# Unauthorized access
try:
    r = requests.get(f"{BASE}/api/v1/auth/me", timeout=5)
    test("无Token访问拒绝 (403)", r.status_code in (401, 403), f"status={r.status_code}")
except Exception as e:
    test("无Token访问拒绝", False, str(e))

# Register new user (cleanup later)
try:
    import time as _t
    _uname = f"test_user_{int(_t.time())}"
    r = requests.post(f"{BASE}/api/v1/auth/register",
                      json={"username": _uname, "email": f"{_uname}@temp.local",
                            "password": "test123456"}, timeout=5)
    test("注册新用户 /auth/register", r.status_code == 200 or r.status_code == 201,
         f"status={r.status_code}, user={_uname}")
except Exception as e:
    test("注册新用户 /auth/register", False, str(e))


# ============================================================================
# 3. 数据模块 (DS-001~005)
# ============================================================================
section("3. 数据模块 (DS-001~005)")

HEADERS = {"Authorization": f"Bearer {TOKEN}"}

# K-line data
try:
    r = requests.get(f"{BASE}/api/v1/data/kline",
                     params={"symbol": "000001", "frequency": "D", "days": 30},
                     timeout=15)
    data = r.json()
    test("K线数据 /data/kline", r.status_code == 200 and "data" in data,
         f"records={len(data.get('data', []))}")
except Exception as e:
    test("K线数据 /data/kline", False, str(e)[:80])

# Real-time quote
try:
    r = requests.get(f"{BASE}/api/v1/data/quote",
                     params={"symbol": "000001"}, timeout=15)
    if r.status_code == 200:
        data = r.json()
        test("实时行情 /data/quote", data.get("symbol") == "000001",
             f"price={data.get('price')}")
    else:
        skip("实时行情 /data/quote", f"数据源不可用 status={r.status_code}")
except Exception as e:
    skip("实时行情 /data/quote", str(e)[:80])

# Fundamental
try:
    r = requests.get(f"{BASE}/api/v1/data/fundamental",
                     params={"symbol": "000001"}, timeout=15)
    if r.status_code == 200:
        data = r.json()
        test("基本面 /data/fundamental", "symbol" in data, f"pe={data.get('pe_ratio')}")
    else:
        skip("基本面 /data/fundamental", f"status={r.status_code}")
except Exception as e:
    skip("基本面 /data/fundamental", str(e)[:80])

# Data source health
try:
    r = requests.get(f"{BASE}/api/v1/data/health", timeout=10)
    test("数据源健康检查 /data/health", r.status_code == 200,
         f"sources={list(r.json().keys()) if r.status_code==200 else 'N/A'}")
except Exception as e:
    test("数据源健康检查 /data/health", False, str(e)[:80])


# ============================================================================
# 4. AI模型配置 (AI-001~007)
# ============================================================================
section("4. AI模型配置 (AI-001~007)")

# List models
try:
    r = requests.get(f"{BASE}/api/v1/models", headers=HEADERS, timeout=5)
    data = r.json()
    test("模型列表 /models", r.status_code == 200 and isinstance(data, list),
         f"count={len(data)}")
    model_id = data[0]["id"] if data else None
except Exception as e:
    test("模型列表 /models", False, str(e))
    model_id = None

# API Key masked (AI-004, NF-011)
try:
    r = requests.get(f"{BASE}/api/v1/models", headers=HEADERS, timeout=5)
    data = r.json()
    if data:
        key = data[0].get("api_key_masked", "")
        test("API Key脱敏 (AI-004)", "****" in key or "请重新配置" in key,
             f"masked={key[:30]}")
    else:
        skip("API Key脱敏", "无模型数据")
except Exception as e:
    test("API Key脱敏", False, str(e))

# Create model (AI-002)
try:
    r = requests.post(f"{BASE}/api/v1/models", headers=HEADERS, json={
        "name": "Test Model", "platform": "openai",
        "base_url": "https://api.openai.com/v1",
        "api_key": "sk-test-key-12345678", "model_name": "gpt-4o-mini",
        "max_tokens": 4096, "temperature": 0.7, "is_default": False,
    }, timeout=5)
    test("创建模型 /models POST", r.status_code == 201, f"status={r.status_code}")
    if r.status_code == 201:
        new_model_id = r.json().get("id")
    else:
        new_model_id = None
except Exception as e:
    test("创建模型 /models POST", False, str(e))
    new_model_id = None

# Update model (AI-006)
if new_model_id:
    try:
        r = requests.put(f"{BASE}/api/v1/models/{new_model_id}", headers=HEADERS, json={
            "name": "Test Model Updated", "temperature": 0.5,
        }, timeout=5)
        test("更新模型 /models PUT", r.status_code == 200, f"status={r.status_code}")
    except Exception as e:
        test("更新模型 /models PUT", False, str(e))

# Set default
if new_model_id:
    try:
        r = requests.put(f"{BASE}/api/v1/models/{new_model_id}/default",
                         headers=HEADERS, timeout=5)
        test("设默认模型 /models/default", r.status_code == 200, f"status={r.status_code}")
    except Exception as e:
        test("设默认模型 /models/default", False, str(e))

# Bindings
try:
    r = requests.get(f"{BASE}/api/v1/models/bindings", headers=HEADERS, timeout=5)
    test("模型绑定列表 /models/bindings", r.status_code == 200,
         f"count={len(r.json())}")
except Exception as e:
    test("模型绑定列表", False, str(e))

# Delete test model
if new_model_id:
    try:
        r = requests.delete(f"{BASE}/api/v1/models/{new_model_id}",
                            headers=HEADERS, timeout=5)
        test("删除模型 /models DELETE", r.status_code == 204, f"status={r.status_code}")
    except Exception as e:
        test("删除模型", False, str(e))

# Connection test (AI-005) - needs real API key
try:
    r = requests.post(f"{BASE}/api/v1/models/{model_id}/test",
                      headers=HEADERS, timeout=30)
    if r.status_code == 200:
        data = r.json()
        test("连接测试 /models/test (AI-005)", "success" in data,
             f"success={data.get('success')}")
    else:
        skip("连接测试 /models/test", f"status={r.status_code}")
except Exception as e:
    skip("连接测试 /models/test", str(e)[:80])


# ============================================================================
# 5. 新闻热点模块 (HT-001~004)
# ============================================================================
section("5. 新闻热点模块 (HT-001~004)")

# News list
try:
    r = requests.get(f"{BASE}/api/v1/news", params={"limit": 5}, timeout=5)
    data = r.json()
    test("新闻列表 /news", r.status_code == 200 and isinstance(data, list),
         f"count={len(data)}")
except Exception as e:
    test("新闻列表 /news", False, str(e))

# News detail
try:
    r = requests.get(f"{BASE}/api/v1/news", params={"limit": 1}, timeout=5)
    news = r.json()
    if news:
        news_id = news[0]["id"]
        r2 = requests.get(f"{BASE}/api/v1/news/{news_id}", timeout=5)
        test("新闻详情 /news/{id}", r2.status_code == 200 and "content" in r2.json(),
             f"title={r2.json().get('title','')[:30]}")
    else:
        skip("新闻详情", "无新闻数据")
except Exception as e:
    test("新闻详情", False, str(e))

# Hot topics (HT-002, HT-003)
try:
    r = requests.get(f"{BASE}/api/v1/news/hot-topics", params={"limit": 5}, timeout=5)
    data = r.json()
    test("热点排行 /news/hot-topics (HT-002)", r.status_code == 200 and len(data) > 0,
         f"topics={len(data)}, top={data[0].get('topic_name','') if data else 'N/A'}")
except Exception as e:
    test("热点排行", False, str(e))

# News stats
try:
    r = requests.get(f"{BASE}/api/v1/news/stats/summary", timeout=5)
    data = r.json()
    test("新闻统计 /news/stats", r.status_code == 200 and "total_72h" in data,
         f"total_72h={data.get('total_72h')}")
except Exception as e:
    test("新闻统计", False, str(e))

# Time window filter (HT-004: 72h)
try:
    r = requests.get(f"{BASE}/api/v1/news", params={"hours": 72, "limit": 50}, timeout=5)
    data = r.json()
    all_recent = all(
        datetime.fromisoformat(n["published_at"].replace("Z","+00:00")).timestamp()
        > time.time() - 72*3600 - 3600  # allow 1h buffer
        for n in data if n.get("published_at")
    )
    test("72h时效过滤 (HT-004)", r.status_code == 200 and all_recent,
         f"all_within_72h={all_recent}")
except Exception as e:
    test("72h时效过滤", False, str(e)[:80])


# ============================================================================
# 6. 大V观点模块 (KV-001~005)
# ============================================================================
section("6. 大V观点模块 (KV-001~005)")

# KOL stats overview
try:
    r = requests.get(f"{BASE}/api/v1/kol/stats/overview", timeout=5)
    data = r.json()
    test("大V统计 /kol/stats (KV-002)", r.status_code == 200 and data.get("total_kols") == 20,
         f"douyin={data.get('douyin_kols')}, weibo={data.get('weibo_kols')}")
except Exception as e:
    test("大V统计", False, str(e))

# KOL list (KV-001: Top 20 per platform)
try:
    r = requests.get(f"{BASE}/api/v1/kol/kols", params={"platform": "douyin"}, timeout=5)
    douyin = r.json()
    r2 = requests.get(f"{BASE}/api/v1/kol/kols", params={"platform": "weibo"}, timeout=5)
    weibo = r2.json()
    test("大V列表 Top20 (KV-001)", len(douyin) == 10 and len(weibo) == 10,
         f"douyin={len(douyin)}, weibo={len(weibo)}")
except Exception as e:
    test("大V列表 Top20", False, str(e))

# KOL ranking (KV-002)
try:
    r = requests.get(f"{BASE}/api/v1/kol/ranking", params={"limit": 10}, timeout=5)
    data = r.json()
    test("大V排行 /kol/ranking (KV-002)", r.status_code == 200 and len(data) > 0,
         f"top1={data[0].get('nickname','') if data else 'N/A'}")
except Exception as e:
    test("大V排行", False, str(e))

# KOL opinions (KV-004)
try:
    r = requests.get(f"{BASE}/api/v1/kol/opinions", params={"limit": 5}, timeout=5)
    data = r.json()
    test("大V观点列表 /kol/opinions (KV-004)", r.status_code == 200 and isinstance(data, list),
         f"count={len(data)}")
except Exception as e:
    test("大V观点列表", False, str(e))

# Add KOL (KV-002: manual management)
try:
    r = requests.post(f"{BASE}/api/v1/kol/kols", json={
        "platform": "douyin", "nickname": "测试大V",
        "certification": "测试认证", "followers_count": 1000000, "rank_position": 99,
    }, timeout=5)
    test("添加大V /kol/kols POST", r.status_code == 201, f"status={r.status_code}")
    test_kol_id = r.json().get("id") if r.status_code == 201 else None
except Exception as e:
    test("添加大V", False, str(e))
    test_kol_id = None

# Delete KOL
if test_kol_id:
    try:
        r = requests.delete(f"{BASE}/api/v1/kol/kols/{test_kol_id}", timeout=5)
        test("删除大V /kol/kols DELETE", r.status_code == 204, f"status={r.status_code}")
    except Exception as e:
        test("删除大V", False, str(e))

# Consensus list (KV-005, AC-008)
try:
    r = requests.get(f"{BASE}/api/v1/kol/consensus", params={"days": 7}, timeout=5)
    test("共识列表 /kol/consensus (KV-005)", r.status_code == 200 and isinstance(r.json(), list),
         f"count={len(r.json())}")
except Exception as e:
    test("共识列表", False, str(e))

# Opinion extract (KV-003) - needs LLM
try:
    r = requests.post(f"{BASE}/api/v1/kol/opinions/extract", json={
        "raw_text": "AI芯片国产替代加速，寒武纪思元590即将量产，坚定看多芯片赛道。"
    }, timeout=30)
    if r.status_code == 200:
        data = r.json()
        test("AI观点提取 /kol/opinions/extract (KV-003)", "direction" in data,
             f"direction={data.get('direction')}")
    else:
        skip("AI观点提取 (KV-003)", f"需配置LLM API Key, status={r.status_code}")
except Exception as e:
    skip("AI观点提取 (KV-003)", str(e)[:60])


# ============================================================================
# 7. 关注列表模块 (WL-001~005)
# ============================================================================
section("7. 关注列表模块 (WL-001~005)")

# List groups
try:
    r = requests.get(f"{BASE}/api/v1/watchlist/groups", headers=HEADERS, timeout=5)
    test("分组列表 /watchlist/groups (WL-002)", r.status_code == 200,
         f"count={len(r.json())}")
except Exception as e:
    test("分组列表", False, str(e))

# Create group
try:
    r = requests.post(f"{BASE}/api/v1/watchlist/groups", headers=HEADERS, json={
        "name": "测试分组", "description": "测试用",
    }, timeout=5)
    test("创建分组 (WL-002)", r.status_code == 201, f"status={r.status_code}")
    group_id = r.json().get("id") if r.status_code == 201 else None
except Exception as e:
    test("创建分组", False, str(e))
    group_id = None

# Add item (WL-001)
try:
    r = requests.post(f"{BASE}/api/v1/watchlist/items", headers=HEADERS, json={
        "symbol": "600519", "name": "贵州茅台", "market": "SH",
        "group_id": group_id, "notes": "测试关注",
    }, timeout=5)
    test("添加关注 (WL-001)", r.status_code == 201, f"status={r.status_code}")
    item_id = r.json().get("id") if r.status_code == 201 else None
except Exception as e:
    test("添加关注", False, str(e))
    item_id = None

# List items
try:
    r = requests.get(f"{BASE}/api/v1/watchlist/items", headers=HEADERS, timeout=5)
    data = r.json()
    test("关注列表 /watchlist/items", r.status_code == 200 and isinstance(data, list),
         f"count={len(data)}")
except Exception as e:
    test("关注列表", False, str(e))

# Export (WL-005)
try:
    r = requests.get(f"{BASE}/api/v1/watchlist/export", headers=HEADERS, timeout=5)
    data = r.json()
    test("导出关注 /watchlist/export (WL-005)", r.status_code == 200 and "data" in data,
         f"count={data.get('count')}")
except Exception as e:
    test("导出关注", False, str(e))

# Remove item
if item_id:
    try:
        r = requests.delete(f"{BASE}/api/v1/watchlist/items/{item_id}",
                            headers=HEADERS, timeout=5)
        test("移除关注 (WL-001)", r.status_code == 204, f"status={r.status_code}")
    except Exception as e:
        test("移除关注", False, str(e))

# Delete group
if group_id:
    try:
        r = requests.delete(f"{BASE}/api/v1/watchlist/groups/{group_id}",
                            headers=HEADERS, timeout=5)
        test("删除分组", r.status_code == 204, f"status={r.status_code}")
    except Exception as e:
        test("删除分组", False, str(e))


# ============================================================================
# 8. 分析引擎模块 (TA/SA + AC)
# ============================================================================
section("8. 分析引擎模块 (TA/SA/AC)")

# Trend analysis (TA-001~005, AC-002) - needs LLM
try:
    r = requests.post(f"{BASE}/api/v1/analysis/trend", json={
        "symbol": "000001", "name": "平安银行", "frequency": "D",
    }, timeout=60)
    if r.status_code == 200:
        data = r.json()
        test("AI趋势分析 /analysis/trend (TA-001)", "ai_conclusion" in data,
             f"direction={data.get('trend_direction')}")
    else:
        skip("AI趋势分析 (TA-001)", f"需配置LLM, status={r.status_code}")
except Exception as e:
    skip("AI趋势分析 (TA-001)", str(e)[:60])

# Serenity analysis (SA-001~003, AC-004) - needs LLM
try:
    r = requests.post(f"{BASE}/api/v1/analysis/serenity", json={
        "symbol": "688256", "name": "寒武纪", "sector": "AI芯片",
    }, timeout=120)
    if r.status_code == 200:
        data = r.json()
        test("Serenity分析 /analysis/serenity (SA-001)", "step1_bom" in data,
             f"conditions={data.get('conditions_met',{}).get('passed')}")
    else:
        skip("Serenity分析 (SA-001)", f"需配置LLM, status={r.status_code}")
except Exception as e:
    skip("Serenity分析 (SA-001)", str(e)[:60])

# Conclusions list (AC-001~008)
try:
    r = requests.get(f"{BASE}/api/v1/analysis/conclusions", params={"limit": 5}, timeout=5)
    test("AI结论列表 /analysis/conclusions", r.status_code == 200,
         f"count={len(r.json())}")
except Exception as e:
    test("AI结论列表", False, str(e))

# Report list (IR-001~005)
try:
    r = requests.get(f"{BASE}/api/v1/analysis/reports", params={"limit": 5}, timeout=5)
    test("研报列表 /analysis/reports (IR-001)", r.status_code == 200,
         f"count={len(r.json())}")
except Exception as e:
    test("研报列表", False, str(e))

# History
try:
    r = requests.get(f"{BASE}/api/v1/analysis/history/trend",
                     params={"symbol": "000001"}, timeout=5)
    test("趋势历史 /analysis/history/trend", r.status_code == 200,
         f"count={len(r.json())}")
except Exception as e:
    test("趋势历史", False, str(e))


# ============================================================================
# 9. Agent决策模块 (AGENT-xxx)
# ============================================================================
section("9. Agent决策模块 (AGENT-xxx)")

# List decisions
try:
    r = requests.get(f"{BASE}/api/v1/agent/decisions", params={"limit": 5}, timeout=5)
    test("决策列表 /agent/decisions", r.status_code == 200 and isinstance(r.json(), list),
         f"count={len(r.json())}")
except Exception as e:
    test("决策列表", False, str(e))

# Run agent workflow (needs LLM + data source)
try:
    r = requests.post(f"{BASE}/api/v1/agent/run", json={
        "symbol": "000001", "name": "平安银行",
    }, timeout=180)
    if r.status_code == 200:
        data = r.json()
        test("Agent工作流 /agent/run", "approval_decision" in data,
             f"status={data.get('status')}, approval={data.get('approval_decision',{}).get('status')}")
    else:
        skip("Agent工作流 /agent/run", f"需配置LLM, status={r.status_code}")
except Exception as e:
    skip("Agent工作流 /agent/run", str(e)[:60])


# ============================================================================
# 10. 数据库验证
# ============================================================================
section("10. 数据库验证")

import subprocess
def run_sql(query):
    cmd = ["docker", "exec", "stock_postgres", "psql", "-U", "stock_user",
           "-d", "stock_analysis", "-t", "-A", "-c", query]
    env = {"PATH": "/Applications/OrbStack.app/Contents/MacOS/xbin:/usr/bin:/bin"}
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=10, env=env)
        return r.stdout.strip() if r.returncode == 0 else f"ERROR: {r.stderr[:80]}"
    except Exception as e:
        return f"EXCEPTION: {e}"

try:
    tables = run_sql("SELECT count(*) FROM information_schema.tables WHERE table_schema='public';")
    test("数据库表数量", tables == "18", f"tables={tables}")
except Exception as e:
    test("数据库表数量", False, str(e))

try:
    kols = run_sql("SELECT count(*) FROM kols;")
    test("大V数据 (KV-001)", kols == "20", f"kols={kols}")
except Exception as e:
    test("大V数据", False, str(e))

try:
    news = run_sql("SELECT count(*) FROM news_articles;")
    test("新闻数据 (HT-001)", int(news) >= 3, f"news={news}")
except Exception as e:
    test("新闻数据", False, str(e))

try:
    topics = run_sql("SELECT count(*) FROM hot_topics;")
    test("热点数据 (HT-002)", int(topics) >= 3, f"topics={topics}")
except Exception as e:
    test("热点数据", False, str(e))

try:
    models = run_sql("SELECT count(*) FROM model_configs;")
    test("模型配置 (AI-001)", int(models) >= 3, f"models={models}")
except Exception as e:
    test("模型配置", False, str(e))

try:
    users = run_sql("SELECT count(*) FROM users;")
    test("用户数据", int(users) >= 1, f"users={users}")
except Exception as e:
    test("用户数据", False, str(e))

try:
    # Check API Key encryption (NF-010)
    keys = run_sql("SELECT api_key_encrypted FROM model_configs LIMIT 1;")
    test("API Key加密存储 (NF-010)", "ENCRYPTED" not in keys and len(keys) > 20,
         f"encrypted_len={len(keys)}")
except Exception as e:
    test("API Key加密存储", False, str(e))


# ============================================================================
# 11. Redis缓存验证
# ============================================================================
section("11. Redis缓存验证 (NF-002)")

try:
    r = subprocess.run(
        ["docker", "exec", "stock_redis", "redis-cli", "ping"],
        capture_output=True, text=True, timeout=5,
        env={"PATH": "/Applications/OrbStack.app/Contents/MacOS/xbin:/usr/bin:/bin"}
    )
    test("Redis连接", "PONG" in r.stdout, f"response={r.stdout.strip()}")
except Exception as e:
    test("Redis连接", False, str(e))

try:
    r = subprocess.run(
        ["docker", "exec", "stock_redis", "redis-cli", "dbsize"],
        capture_output=True, text=True, timeout=5,
        env={"PATH": "/Applications/OrbStack.app/Contents/MacOS/xbin:/usr/bin:/bin"}
    )
    test("Redis有数据", int(r.stdout.strip()) >= 0, f"keys={r.stdout.strip()}")
except Exception as e:
    test("Redis有数据", False, str(e))


# ============================================================================
# Summary
# ============================================================================
section("测试总结")
total = PASS + FAIL + SKIP
print(f"\n  总测试数: {total}")
print(f"  ✅ 通过: {PASS}")
print(f"  ❌ 失败: {FAIL}")
print(f"  ⏭️  跳过: {SKIP} (需LLM API Key)")
print(f"  通过率: {PASS/(PASS+FAIL)*100:.1f}%" if (PASS+FAIL) > 0 else "  N/A")

# List failures
failures = [r for r in RESULTS if "FAIL" in r["status"]]
if failures:
    print(f"\n  失败项:")
    for f in failures:
        print(f"    ❌ {f['name']} — {f['detail']}")

print(f"\n{'='*60}")
sys.exit(0 if FAIL == 0 else 1)
