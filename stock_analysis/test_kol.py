# ============================================================================
# Tests - KOL direction normalization (cross-comparison consistency)
# ============================================================================
"""纯函数测试：不触网、不依赖数据库。"""
from __future__ import annotations

import pytest

from app.api.kol import normalize_kol_direction


@pytest.mark.parametrize("raw", ["看多", "bullish", "BULLISH", " 看涨 ", "bull", "buy", "positive"])
def test_bullish_labels(raw):
    assert normalize_kol_direction(raw) == "bullish"


@pytest.mark.parametrize("raw", ["看空", "bearish", "BEARISH", " 看跌 ", "bear", "sell", "negative"])
def test_bearish_labels(raw):
    assert normalize_kol_direction(raw) == "bearish"


@pytest.mark.parametrize("raw", ["", None, "中性", "neutral", "unknown"])
def test_neutral_labels(raw):
    assert normalize_kol_direction(raw) == "neutral"
