# ============================================================================
# Tests - Research report target-price logic (rating/target consistency)
# ============================================================================
"""纯函数测试：不触网、不依赖数据库。"""
from __future__ import annotations

import pytest

from app.reports import (
    DEFAULT_UPSIDE, RATING_UPSIDE, target_from_price, upside_for,
)
from app.quotes import parse_tencent_payload, to_tencent_symbol


# -- Target price derivation --------------------------------------------------
def test_buy_target_is_always_above_price():
    for i in range(200):
        t = target_from_price(100.0, "买入", f"seed-{i}")
        assert t is not None
        assert t > 100.0, "买入评级的目标价不可能低于现价"


def test_sell_target_is_always_below_price():
    for i in range(200):
        t = target_from_price(100.0, "卖出", f"seed-{i}")
        assert t is not None
        assert t < 100.0


def test_upside_within_rating_band():
    for rating, (lo, hi) in RATING_UPSIDE.items():
        for i in range(50):
            u = upside_for(rating, f"s{i}")
            assert lo - 1e-9 <= u <= hi + 1e-9


def test_unknown_rating_uses_default_band():
    lo, hi = DEFAULT_UPSIDE
    u = upside_for("强烈推荐", "x")
    assert lo - 1e-9 <= u <= hi + 1e-9


def test_upside_is_deterministic():
    assert upside_for("买入", "abc") == upside_for("买入", "abc")
    assert upside_for("买入", "abc") != upside_for("买入", "abd")


def test_target_price_rounding_and_none():
    assert target_from_price(276.86, "买入", "x") == round(276.86 * (1 + upside_for("买入", "x")), 2)
    assert target_from_price(0, "买入", "x") is None
    assert target_from_price(None, "买入", "x") is None


# -- Tencent quote helpers ----------------------------------------------------
@pytest.mark.parametrize("raw,expected", [
    ("688012.SH", "sh688012"),
    ("000001", "sz000001"),
    ("600519.SH", "sh600519"),
    ("00700.HK", "hk00700"),
    ("300124.SZ", "sz300124"),
    ("sh600000", "sh600000"),
    ("", None),
    ("abc", None),
])
def test_to_tencent_symbol(raw, expected):
    assert to_tencent_symbol(raw) == expected


def test_parse_tencent_payload():
    fields = ["1", "中微公司", "688012", "276.63", "278.50", "282.00", "5209520"]
    fields += ["x"] * 23
    fields += ["20261021103201", "-1.87", "-0.67", "282.10", "276.30", "276.63/5209520/1"]
    text = 'v_sh688012="' + "~".join(fields) + '";'
    quotes = parse_tencent_payload(text)
    assert "688012" in quotes
    q = quotes["688012"]
    assert q["price"] == 276.63
    assert q["pre_close"] == 278.50
    assert q["change_pct"] == -0.67
    assert q["high"] == 282.10
    assert q["low"] == 276.30
    assert q["volume"] == 5209520 * 100


def test_parse_tencent_payload_ignores_garbage():
    assert parse_tencent_payload("not a quote") == {}
    assert parse_tencent_payload('v_sh1="too~short";') == {}
