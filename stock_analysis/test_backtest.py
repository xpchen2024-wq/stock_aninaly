#!/usr/bin/env python3
"""
Backtest engine + API tests.

The 策略回测 page's 运行回测 button was a dead static element: no JS handler
and no backend existed. These tests lock in the new implementation:
- app.backtest: deterministic strategy engine over K-lines (ma_cross, macd_cross)
- POST /api/v1/backtest/run, GET /api/v1/backtest/list
"""
import asyncio
from datetime import date
from unittest.mock import AsyncMock

import pytest


def _make_klines(n=120, base=100.0):
    """Synthetic daily klines: sine-wave around base so MA/MACD cross over."""
    from app.adapters import KlineData
    import math
    klines = []
    for i in range(n):
        close = base + 10 * math.sin(i / 8.0) + i * 0.05
        high = close + 1.5
        low = close - 1.5
        open_ = closes_prev = close - 0.5
        klines.append(KlineData(
            symbol="300750", date=f"2025-{(i // 28) % 12 + 1:02d}-{i % 28 + 1:02d}",
            open=open_, high=high, low=low, close=close,
            volume=1_000_000 + i * 1000, amount=close * 1_000_000,
        ))
    return klines


class MockFailover:
    def __init__(self, klines):
        self.fetch_with_failover = AsyncMock(return_value=klines)


# ===================================================================
# Engine — pure functions
# ===================================================================
class TestBacktestEngine:

    def test_ma_cross_generates_trades_and_metrics(self):
        from app.backtest import run_backtest, BacktestParams

        params = BacktestParams(
            symbol="300750", strategy="ma_cross",
            initial_capital=1_000_000.0,
        )
        result = run_backtest(_make_klines(), params)

        assert result["symbol"] == "300750"
        assert result["strategy"] == "ma_cross"
        assert result["total_bars"] > 60
        # metrics present and sane
        m = result["metrics"]
        assert "total_return_pct" in m and isinstance(m["total_return_pct"], float)
        assert "max_drawdown_pct" in m and m["max_drawdown_pct"] <= 0
        assert "sharpe_ratio" in m
        assert "win_rate_pct" in m and 0 <= m["win_rate_pct"] <= 100
        assert "trade_count" in m and m["trade_count"] >= 0
        # equity curve aligned with bars
        assert len(result["equity_curve"]) == result["total_bars"]
        # trades have required fields
        for t in result["trades"]:
            assert {"date", "side", "price", "quantity"} <= set(t.keys())

    def test_monthly_and_yearly_returns_present(self):
        from app.backtest import run_backtest, BacktestParams

        params = BacktestParams(symbol="300750", strategy="ma_cross",
                                initial_capital=1_000_000.0)
        result = run_backtest(_make_klines(n=400), params)

        monthly = result["monthly_returns_pct"]
        yearly = result["yearly_returns_pct"]
        assert monthly and yearly, "月度/年度收益必须返回，供热力图与年度表渲染"
        for k, v in monthly.items():
            assert len(k) == 7 and k[4] == "-"
            assert isinstance(v, float)
        for k, v in yearly.items():
            assert len(k) == 4 and k.isdigit()
            assert isinstance(v, float)
        # 月度键的数量不应超过 K 线覆盖的月份数
        assert len(monthly) <= result["total_bars"]

    def test_macd_cross_runs(self):
        from app.backtest import run_backtest, BacktestParams
        result = run_backtest(
            _make_klines(),
            BacktestParams(symbol="300750", strategy="macd_cross"),
        )
        assert result["strategy"] == "macd_cross"
        assert result["total_bars"] > 60

    def test_unknown_strategy_rejected(self):
        from app.backtest import run_backtest, BacktestParams
        with pytest.raises(ValueError):
            run_backtest(_make_klines(), BacktestParams(strategy="nope"))

    def test_insufficient_data_raises(self):
        from app.backtest import run_backtest, BacktestParams
        with pytest.raises(ValueError):
            run_backtest(_make_klines()[:10], BacktestParams(strategy="ma_cross"))

    def test_fees_reduce_returns(self):
        from app.backtest import run_backtest, BacktestParams
        klines = _make_klines()
        r_no_fee = run_backtest(klines, BacktestParams(strategy="ma_cross", fee_rate=0.0))
        r_fee = run_backtest(klines, BacktestParams(strategy="ma_cross", fee_rate=0.003))
        assert r_fee["metrics"]["final_equity"] <= r_no_fee["metrics"]["final_equity"]

    def test_stop_loss_limits_single_trade_loss(self):
        from app.backtest import run_backtest, BacktestParams
        result = run_backtest(
            _make_klines(),
            BacktestParams(strategy="ma_cross", stop_loss_pct=2.0),
        )
        # every closed trade loses at most ~2% price move (plus fees)
        for t in result["trades"]:
            if t["side"] == "sell" and t.get("pnl_pct") is not None:
                assert t["pnl_pct"] >= -4.0


# ===================================================================
# API layer
# ===================================================================
@pytest.fixture(autouse=True)
def _fresh_db_engine_per_test():
    """Dispose the shared asyncpg pool after each test.

    TestClient runs each request on a fresh event loop; pooled connections
    bound to a previous loop raise 'attached to a different loop'.
    """
    yield
    import app.database as database
    try:
        asyncio.run(database.close_db())
    except Exception:
        pass
    database.engine = database.create_async_engine(
        database.settings.DATABASE_URL,
        echo=False,
        pool_pre_ping=True,
    )
    database.async_session_factory = database.async_sessionmaker(
        database.engine, class_=database.AsyncSession, expire_on_commit=False,
    )


class TestBacktestAPI:
    def _client(self, klines):
        """TestClient without lifespan; failover patched to return klines."""
        from fastapi.testclient import TestClient
        from app.main import app
        import app.api.backtest as bt_api

        bt_api._failover = MockFailover(klines)
        return TestClient(app)

    def test_run_endpoint(self):
        client = self._client(_make_klines())
        resp = client.post("/api/v1/backtest/run", json={
            "symbol": "300750.SZ", "name": "宁德时代",
            "strategy": "ma_cross",
            "start_date": "2025-01-01", "end_date": "2026-06-17",
            "initial_capital": 1_000_000, "fee_rate": 0.0003,
        })
        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert data["status"] == "completed"
        assert data["result"]["symbol"] == "300750.SZ"
        assert "metrics" in data["result"]

    def test_list_endpoint(self):
        client = self._client(_make_klines())
        resp = client.get("/api/v1/backtest/list?limit=10")
        assert resp.status_code == 200
