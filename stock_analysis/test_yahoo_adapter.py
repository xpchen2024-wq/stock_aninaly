# ============================================================================
# Tests - Yahoo adapter must not use a JS engine (yfinance/mini_racer)
# ============================================================================
"""回归测试：yfinance 的 cookie/crumb 走 mini_racer(V8) 原生调用，一旦在 V8 里
卡死会持有 GIL，冻结整个 asyncio 事件循环（表现为全站接口无响应、登录失败）。
因此 Yahoo 适配器必须只走纯 HTTP。
"""
from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest

from app.adapters import YahooFinanceAdapter, _yahoo_rows, _yahoo_symbol


ADAPTER_FILE = Path(__file__).with_name("app") / "adapters.py"


# -- 源码级约束 ---------------------------------------------------------------
def test_adapters_has_no_yfinance_runtime_usage():
    """app/adapters.py 里不允许再出现 yfinance 的 import 或调用。"""
    tree = ast.parse(ADAPTER_FILE.read_text(encoding="utf-8"))
    offenders = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] == "yfinance":
                    offenders.append(f"import {alias.name} (line {node.lineno})")
        elif isinstance(node, ast.ImportFrom):
            if (node.module or "").split(".")[0] == "yfinance":
                offenders.append(f"from {node.module} import ... (line {node.lineno})")
    assert not offenders, f"adapters.py 不得再依赖 yfinance: {offenders}"


@pytest.mark.parametrize("method", ["fetch_kline", "fetch_realtime_quote",
                                    "fetch_fundamental", "health_check"])
def test_yahoo_methods_do_not_mention_yfinance(method):
    src = inspect.getsource(getattr(YahooFinanceAdapter, method))
    assert "yfinance" not in src.replace("yfinance would risk a GIL freeze", ""), \
        f"{method} 不能引用 yfinance"


def test_yahoo_health_check_is_overridden():
    """必须覆盖基类实现（基类会走 fetch_realtime_quote → 曾经的 yfinance 路径）。"""
    assert "health_check" in YahooFinanceAdapter.__dict__


# -- 纯函数 -------------------------------------------------------------------
@pytest.mark.parametrize("raw,expected", [
    ("600519", "600519.SS"),
    ("000001", "000001.SZ"),
    ("AAPL", "AAPL"),
    ("00700.HK", "00700.HK"),
])
def test_yahoo_symbol(raw, expected):
    assert _yahoo_symbol(raw) == expected


def test_yahoo_rows_parses_columnar_payload():
    payload = {
        "chart": {"result": [{
            "timestamp": [1704067200, 1704153600],
            "indicators": {"quote": [{
                "open": [1.0, 2.0], "high": [1.5, 2.5],
                "low": [0.5, 1.5], "close": [1.2, 2.2], "volume": [100.0, 200.0],
            }]},
        }]}
    }
    rows = _yahoo_rows(payload)
    assert len(rows) == 2
    assert rows[0]["close"] == 1.2 and rows[1]["close"] == 2.2
    assert rows[0]["date"] == "2024-01-01"


def test_yahoo_rows_handles_empty_and_garbage():
    assert _yahoo_rows({}) == []
    assert _yahoo_rows({"chart": {"result": []}}) == []
    assert _yahoo_rows({"chart": None}) == []
    # 缺列不应抛异常
    bad = {"chart": {"result": [{"timestamp": [1], "indicators": {"quote": [{}]}}]}}
    rows = _yahoo_rows(bad)
    assert rows and rows[0]["close"] == 0.0
