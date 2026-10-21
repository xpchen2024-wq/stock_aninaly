#!/usr/bin/env python3
"""Capital flow / fund holdings / national-fund identification tests.

Network-dependent endpoints are tested via the live API (akshare), so these
tests focus on: national-fund keyword matching, period generation, and the
route contract (registered paths).
"""
from fastapi.testclient import TestClient

from app.api.capital import _is_national_fund, _latest_periods, _strip_suffix, _market_of, _fund_trend_note


class TestNationalFundMatching:
    def test_huijin(self):
        assert _is_national_fund("中央汇金资产管理有限责任公司") == "中央汇金"

    def test_big_fund(self):
        assert _is_national_fund("国家集成电路产业投资基金二期股份有限公司") == "国家集成电路产业投资基金"

    def test_shebao(self):
        assert _is_national_fund("全国社保基金一零四组合") == "社保基金"

    def test_ordinary_holder(self):
        assert _is_national_fund("香港中央结算有限公司") is None
        assert _is_national_fund("黄世霖") is None


class TestSymbolParsing:
    def test_strip_suffix(self):
        assert _strip_suffix("300750.SZ 宁德时代") == "300750"
        assert _strip_suffix("600519.SH") == "600519"
        assert _strip_suffix("300750") == "300750"

    def test_market(self):
        assert _market_of("300750.SZ") == "sz"
        assert _market_of("600519") == "sh"
        assert _market_of("000001") == "sz"


class TestPeriods:
    def test_latest_periods_count_and_desc(self):
        p = _latest_periods(4)
        assert len(p) == 4
        assert p == sorted(p, reverse=True)
        assert all(len(x) == 8 for x in p)

    def test_quarter_ends(self):
        p = _latest_periods(6)
        assert all(x[4:6] in ("03", "06", "09", "12") for x in p)
        assert all(x[6:] == {"03": "31", "06": "30", "09": "30", "12": "31"}[x[4:6]] for x in p)


class TestFundTrendNote:
    def test_note_mentions_change(self):
        history = [
            {"period": "20260331", "fund_count": 100, "total_shares": 200_000_000,
             "change_flag": "增仓"},
            {"period": "20260630", "fund_count": 90, "total_shares": 150_000_000,
             "change_flag": "减仓"},
        ]
        note = _fund_trend_note(history)
        assert "100→90" in note
        assert "-25.0%" in note

    def test_single_period(self):
        note = _fund_trend_note([{"period": "20260630", "fund_count": 5,
                                  "total_shares": 1, "change_flag": "增仓"}])
        assert note == "增仓"


class TestRoutes:
    def test_routes_registered(self):
        from app.main import app
        paths = {r.path for r in app.routes}
        assert "/api/v1/capital/shareholders" in paths
        assert "/api/v1/capital/fund-holdings" in paths
        assert "/api/v1/capital/money-flow" in paths
        assert "/api/v1/capital/national-funds" in paths
