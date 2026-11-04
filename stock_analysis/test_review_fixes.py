"""Regression coverage for the twelve findings in the working-tree review."""
from datetime import date, datetime, timedelta
from html.parser import HTMLParser
import json
from pathlib import Path
import re
import shutil
import subprocess
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pandas as pd
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from app.api.roundtable import _kol_context, _persist_run
from app.api import roundtable as roundtable_api
from app.api.watchlist import watchlist_news
from app.database import get_db
from app.global_markets import _fetch_one, _parse_meta
from app.kol_feed import _build_summary, generate_kol_opinions
from app.models import (
    KOL, KOLOpinion, NewsArticle, RoundtableMemory, RoundtableRun,
    WatchlistItem, now_utc,
)
from app.news_sources import RawNews, ingest_news
from app.roundtable.agents import PortfolioManager, Trader
from app.roundtable.graph import RoundtableGraph
from app.roundtable.memory import _load_series, compute_settlement, load_memory_context
from app.roundtable.state import RoundtableConfig
from app.roundtable.toolkit import RoundtableToolkit
from test_roundtable import FakeGateway


@pytest.fixture
def db():
    engine = create_engine("sqlite://", poolclass=StaticPool,
                           connect_args={"check_same_thread": False})
    for model in (KOL, KOLOpinion, NewsArticle, RoundtableMemory, RoundtableRun, WatchlistItem):
        model.__table__.create(engine)
    with Session(engine, expire_on_commit=False) as session:
        yield SimpleNamespace(
            execute=AsyncMock(side_effect=session.execute),
            add=session.add, commit=AsyncMock(side_effect=session.commit),
            flush=AsyncMock(side_effect=session.flush), session=session,
        )
    engine.dispose()


def test_global_quote_uses_previous_trading_close():
    quote = _parse_meta({"regularMarketPrice": 110, "previousClose": 112,
                         "chartPreviousClose": 100})
    assert quote["change_pct"] == -1.79
    assert quote["change"] == -2


@pytest.mark.asyncio
async def test_global_chart_range_is_one_day():
    def respond(request):
        assert request.url.params["range"] == "1d"
        return httpx.Response(200, json={"chart": {"result": [{"meta": {
            "regularMarketPrice": 110, "chartPreviousClose": 100,
        }}]}})
    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        quote = await _fetch_one(client, {"symbol": "AAPL", "name": "Apple", "group": "us"})
    assert quote["change_pct"] == 10


@pytest.mark.asyncio
async def test_historical_capital_excludes_future_rows(monkeypatch):
    raw = pd.DataFrame({
        "日期": ["2026-01-02", "2026-01-05", "2026-01-06"],
        "主力净流入-净额": [10, 20, 9999],
    })
    monkeypatch.setattr("asyncio.to_thread", AsyncMock(return_value=raw))
    tk = RoundtableToolkit(None, "600519", "test", trade_date="2026-01-05")
    await tk._load_capital()
    assert [r["date"] for r in tk.capital["rows"]] == ["2026-01-02", "2026-01-05"]
    assert tk.capital["main_net_total_5d"] == 30


@pytest.mark.asyncio
async def test_historical_fundamental_does_not_fetch_current_snapshot():
    failover = SimpleNamespace(fetch_with_failover=AsyncMock())
    tk = RoundtableToolkit(failover, "600519", "test", trade_date="2020-01-01")
    await tk._load_fundamental()
    failover.fetch_with_failover.assert_not_called()
    assert not tk.fundamental
    assert tk.errors


@pytest.mark.asyncio
async def test_historical_news_uses_china_day_boundary(monkeypatch):
    def article(hour):
        return RawNews("source", str(hour), published_at=datetime(2026, 1, 5, hour))
    monkeypatch.setattr("app.news_sources.fetch_stock_news",
                        AsyncMock(return_value=[article(15), article(16)]))
    tk = RoundtableToolkit(None, "600519", "test", trade_date="2026-01-05")
    await tk._load_news()
    assert [n["title"] for n in tk.news] == ["15"]
    assert tk.news[0]["published_at"] == "2026-01-05 23:00"


@pytest.mark.asyncio
async def test_historical_kol_excludes_later_publications_and_collections(db):
    cutoff = datetime(2026, 1, 6)
    for title, published, collected in (
        ("known", cutoff - timedelta(days=1), cutoff - timedelta(days=1)),
        ("future publication", cutoff + timedelta(days=1), cutoff - timedelta(days=1)),
        ("backfilled", cutoff - timedelta(days=1), cutoff + timedelta(days=1)),
    ):
        db.add(KOLOpinion(kol_id="a0000000-0000-0000-0000-000000000001",
                         platform="weibo", summary=title, published_at=published,
                         collected_at=collected, mentioned_stocks=[{"code": "600519"}]))
    await db.commit()
    text = await _kol_context(db, "600519", as_of=cutoff)
    assert "known" in text
    assert "future publication" not in text and "backfilled" not in text


@pytest.mark.asyncio
async def test_historical_memory_hides_future_decisions_and_reflections(db):
    cutoff = datetime(2026, 1, 6)
    for symbol, created, reflection in (
        ("600519", cutoff - timedelta(days=3), "future same-symbol lesson"),
        ("600519", cutoff + timedelta(days=1), "future decision"),
        ("000001", cutoff - timedelta(days=3), "future other-symbol lesson"),
    ):
        db.add(RoundtableMemory(session_id=reflection, symbol=symbol, name="test",
                               trade_date=date(2026, 1, 2), signal="buy", confidence=0.6,
                               created_at=created, settled=True, settled_at=cutoff + timedelta(days=1),
                               realized_return_pct=99, reflection=reflection))
    await db.commit()
    text = await load_memory_context(db, "600519", as_of=cutoff)
    assert "尚未结算" in text
    assert "99" not in text and "future" not in text


@pytest.mark.asyncio
async def test_old_settlement_fetch_starts_before_decision():
    failover = SimpleNamespace(fetch_with_failover=AsyncMock(return_value=[]))
    await _load_series(failover, "600519", date(2020, 1, 6))
    kwargs = failover.fetch_with_failover.call_args.kwargs
    assert kwargs["start_date"] == "20191207"
    assert kwargs["end_date"] >= "20200106"


def test_settlement_rejects_truncated_history_even_with_entry_price():
    recent = pd.DataFrame({"date": ["2026-04-01", "2026-04-02"], "close": [110, 120]})
    assert compute_settlement(recent, pd.DataFrame(), "2026-01-05", 2, 100) is None


def test_suspended_stock_and_benchmark_share_exit_date():
    stock = pd.DataFrame({"date": ["2026-01-05", "2026-01-08", "2026-01-09"],
                          "close": [100, 105, 110]})
    bench = pd.DataFrame({"date": ["2026-01-05", "2026-01-06", "2026-01-07", "2026-01-08", "2026-01-09"],
                          "close": [100, 101, 102, 103, 104]})
    outcome = compute_settlement(stock, bench, "2026-01-05", 2)
    assert outcome["benchmark_return_pct"] == 4
    assert outcome["alpha_pct"] == 6
    outcome = compute_settlement(stock, bench.iloc[:-1], "2026-01-05", 2)
    assert outcome["benchmark_return_pct"] is None


@pytest.mark.asyncio
async def test_decision_horizon_matches_persisted_memory(db):
    tk = RoundtableToolkit(None, "600519", "test", horizon_days=20)
    tk.snapshot_core = lambda: {"price": 100}
    decision = PortfolioManager(None, RoundtableConfig())._normalize(
        {"action": "buy", "horizon_days": 10}, tk)
    await _persist_run(db, {"session_id": "horizon", "symbol": "600519",
                           "trade_date": "2026-01-05", "decision": decision,
                           "snapshot": {"price": 100}, "config": {"horizon_days": 20}})
    memory = db.session.execute(select(RoundtableMemory)).scalar_one()
    assert decision["horizon_days"] == memory.horizon_days == 20


@pytest.mark.asyncio
async def test_data_failures_mark_graph_as_degraded(monkeypatch):
    async def prepare(tk):
        tk.errors.append("fundamental source failed")
        return tk.snapshot()
    monkeypatch.setattr(RoundtableToolkit, "prepare", prepare)
    events = [event async for event in RoundtableGraph(FakeGateway(), None).run_stream("600519")]
    result = events[-1][1]
    assert result["status"] == "completed_with_errors"
    assert result["degraded"] is True
    assert "fundamental source failed" in result["errors"]
    assert result["snapshot"]["errors"] == result["errors"]


def mock_news_sources(monkeypatch, general, stock):
    monkeypatch.setattr("app.news_sources.fetch_all_news", AsyncMock(return_value=general))
    monkeypatch.setattr("app.news_sources.fetch_stock_news", AsyncMock(return_value=stock))
    monkeypatch.setattr("app.news_sources._watched_symbols", AsyncMock(return_value=["600519"]))
    monkeypatch.setattr("app.news_sources._keyword_tag_items", AsyncMock())


@pytest.mark.asyncio
async def test_existing_news_does_not_consume_new_article_limit(db, monkeypatch):
    db.add(NewsArticle(source="source", title="existing", published_at=now_utc(),
                       related_stocks=[{"symbol": "600519"}]))
    await db.commit()
    mock_news_sources(monkeypatch, [RawNews("source", "new general")],
                      [RawNews("source", "existing", related_stocks=[{"symbol": "600519"}])])
    stats = await ingest_news(db, limit=1)
    assert stats["inserted"] == 1
    assert db.session.execute(select(NewsArticle.title)).scalars().all() == ["existing", "new general"]


@pytest.mark.asyncio
@pytest.mark.parametrize("already_stored", [False, True])
async def test_duplicate_news_merges_stock_associations(db, monkeypatch, already_stored):
    if already_stored:
        db.add(NewsArticle(source="source", title="shared", published_at=now_utc(),
                           related_stocks=[{"symbol": "000001", "name": "first"}]))
        await db.commit()
    mock_news_sources(monkeypatch, [], [
        RawNews("source", "shared", related_stocks=[{"symbol": "600519"}]),
        RawNews("source", "shared", related_stocks=[{"symbol": "000001", "name": "first"}]),
    ])
    await ingest_news(db)
    row = db.session.execute(select(NewsArticle)).scalar_one()
    assert {s["symbol"] for s in row.related_stocks} == {"600519", "000001"}


@pytest.mark.asyncio
async def test_news_insert_limit_is_respected(db, monkeypatch):
    mock_news_sources(monkeypatch, [RawNews("source", str(i)) for i in range(10)], [])
    stats = await ingest_news(db, limit=2)
    assert stats["inserted"] == 2
    assert len(db.session.execute(select(NewsArticle)).scalars().all()) == 2


@pytest.mark.asyncio
async def test_kol_skips_used_url_less_article_and_tries_next(db):
    kol = KOL(platform="weibo", nickname="synthetic", certification="", followers_count=1000)
    db.add(kol)
    await db.flush()
    now = now_utc()
    a = NewsArticle(source="source", title="A", published_at=now)
    b = NewsArticle(source="source", title="B", published_at=now - timedelta(minutes=1))
    db.add(a)
    db.add(b)
    db.add(KOLOpinion(kol_id=kol.id, platform="weibo", published_at=now,
                     summary=_build_summary(a, "neutral")))
    await db.commit()
    stats = await generate_kol_opinions(db)
    assert stats["inserted"] == 1
    summaries = db.session.execute(select(KOLOpinion.summary)).scalars().all()
    assert _build_summary(b, "neutral") in summaries


@pytest.mark.asyncio
async def test_watchlist_finds_related_news_beyond_latest_200(db):
    user_id = "a0000000-0000-0000-0000-000000000001"
    db.add(WatchlistItem(user_id=user_id, symbol="600519", name="test"))
    now = now_utc()
    for i in range(201):
        db.add(NewsArticle(source="source", title=str(i), published_at=now))
    db.add(NewsArticle(source="source", title="related older article",
                       published_at=now - timedelta(hours=1), related_stocks=[{"symbol": "600519"}]))
    await db.commit()
    result = await watchlist_news(user=SimpleNamespace(id=user_id), db=db)
    assert len(result) == 50
    assert result[0].title == "related older article"


@pytest.mark.asyncio
async def test_portfolio_constraints_reach_trader_and_manager():
    config = RoundtableConfig(horizon_days=20, portfolio_note="already invested 30%; cash 1000")
    tk = RoundtableToolkit(None, "600519", "test", horizon_days=20)
    trader = Trader(None, config)
    trader.chat = AsyncMock(return_value="plan")
    await trader.plan(tk, "reports", "judgement")
    manager = PortfolioManager(None, config)
    manager.chat_json = AsyncMock(return_value={"action": "hold"})
    await manager.decide(tk, "reports", "judgement", "plan", "risk", "memory")
    for prompt in (trader.chat.call_args.args[1], manager.chat_json.call_args.args[1]):
        assert config.portfolio_note in prompt
        assert "20" in prompt


def test_roundtable_stream_persists_consistent_decision_and_memory(db, monkeypatch):
    monkeypatch.setattr(roundtable_api, "_resolve_symbol", AsyncMock(return_value=("600519", "test")))
    monkeypatch.setattr(roundtable_api, "_resolve_tiers", AsyncMock(return_value=(None, None)))
    monkeypatch.setattr(roundtable_api, "get_gateway", lambda: FakeGateway())
    monkeypatch.setattr(roundtable_api, "get_failover", lambda: None)
    monkeypatch.setattr(roundtable_api, "settle_due", AsyncMock(return_value={"settled": 0}))

    async def prepare(tk):
        tk.errors.append("historical fundamentals unavailable")
        return tk.snapshot()

    monkeypatch.setattr(RoundtableToolkit, "prepare", prepare)
    app = FastAPI()
    app.include_router(roundtable_api.router, prefix="/roundtable")

    async def test_db():
        yield db

    app.dependency_overrides[get_db] = test_db
    with TestClient(app) as client:
        response = client.post("/roundtable/run/stream", json={
            "symbol": "600519", "trade_date": "2020-01-05", "horizon_days": 20,
            "portfolio_note": "cash 1000",
        })
    assert response.status_code == 200
    events = response.text.split("\n\n")
    result = next(json.loads(event.split("data: ", 1)[1]) for event in events
                  if event.startswith("event: result"))
    assert result["decision"]["horizon_days"] == 20
    assert result["degraded"] is True
    run = db.session.execute(select(RoundtableRun)).scalar_one()
    memory = db.session.execute(select(RoundtableMemory)).scalar_one()
    assert run.decision["horizon_days"] == memory.horizon_days == 20
    assert run.errors == ["historical fundamentals unavailable"]


class MarkupInspector(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []

    def handle_starttag(self, tag, attrs):
        assert tag not in {"script", "img", "svg", "iframe"}
        assert not any(key.startswith("on") for key, _ in attrs)
        if tag == "a":
            self.links.append(dict(attrs)["href"])


@pytest.mark.parametrize("url", ["javascript:alert(1)", "data:text/html,test", 'https://example.com/?q=" onclick="alert(1)'])
def test_news_markup_escapes_external_fields_and_restricts_links(url):
    node = shutil.which("node")
    if not node:
        pytest.skip("Node.js is required for frontend regression checks")
    source = (Path(__file__).resolve().parent.parent / "UI_design" / "index.html").read_text()
    markup_fn = source.split("  function watchlistNewsMarkup(n) {", 1)[1].split("  function escapeHtml", 1)[0]
    escape_fn = source.split("  function escapeHtml(s) {", 1)[1].split("\n  }", 1)[0]
    payload = '<img src=x onerror="alert(1)"><script>alert(1)</script>'
    news = {key: payload for key in ("title", "summary", "source", "sentiment", "heat_score", "published_at")}
    news.update(url=url, related_stocks=[{"symbol": payload}])
    script = ("function watchlistNewsMarkup(n) {" + markup_fn
              + "function escapeHtml(s) {" + escape_fn + "\n}\n"
              + "console.log(watchlistNewsMarkup(" + json.dumps(news) + "));" )
    completed = subprocess.run([node, "-e", script], text=True, capture_output=True, check=True)
    inspector = MarkupInspector()
    inspector.feed(completed.stdout)
    assert "&lt;img" in completed.stdout
    if url.startswith("https:"):
        assert len(inspector.links) == 1
    else:
        assert inspector.links == []


def test_frontend_inline_script_syntax():
    node = shutil.which("node")
    if not node:
        pytest.skip("Node.js is required for frontend syntax checks")
    source = (Path(__file__).resolve().parent.parent / "UI_design" / "index.html").read_text()
    scripts = re.findall(r"<script\b[^>]*>([\s\S]*?)</script>", source)
    for script in scripts:
        subprocess.run([node, "-e", "new (require('vm').Script)(" + json.dumps(script) + ");"],
                       text=True, capture_output=True, check=True)
