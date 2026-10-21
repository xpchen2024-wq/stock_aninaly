#!/usr/bin/env python3
"""
White-box unit tests for app/agents.py

Covers:
- _extract_json()         : 8 branches (exhaustive)
- BaseAgent               : init + _call_llm delegation
- FundamentalAnalyst      : success / invalid JSON / missing keys
- SentimentAnalyst        : success / invalid JSON / empty/null input
- NewsAnalyst             : success / invalid JSON / empty/null input
- TechnicalAnalyst        : success / indicator exception / invalid JSON
- BullishResearcher       : bullet parsing / fallback
- BearishResearcher       : bullet parsing / fallback
- TradingAgent            : success / invalid JSON
- RiskManager             : success / invalid JSON
- PortfolioManager        : success / invalid JSON
- AgentWorkflow           : happy path / partial failures / tie-breaking / conviction calc

Usage:
    cd /path/to/stock_analysis
    python -m pytest test_agents_whitebox.py -v
"""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from app.agents import (
    _extract_json,
    BaseAgent,
    FundamentalAnalyst,
    SentimentAnalyst,
    NewsAnalyst,
    TechnicalAnalyst,
    BullishResearcher,
    BearishResearcher,
    TradingAgent,
    RiskManager,
    PortfolioManager,
    AgentWorkflow,
    AGENT_FUNDAMENTAL,
    AGENT_SENTIMENT,
    AGENT_NEWS,
    AGENT_TECHNICAL,
    AGENT_BULL,
    AGENT_BEAR,
    AGENT_TRADER,
    AGENT_RISK,
    AGENT_PM,
)
from app.model_gateway import LLMResponse


# ===================================================================
# Helpers
# ===================================================================

class MockGateway:
    """Mock ModelGateway with configurable chat() response."""

    _default_response = LLMResponse(
        content='{"scores":{"overall":0.8},"red_flags":[],"intrinsic_value_range":{"low":100,"high":200},"confidence":0.9}',
        model="test-model",
        tokens_used=100,
        latency_ms=200.0,
        success=True,
    )

    def __init__(self, response=None):
        self.response = response or self._default_response
        self.chat = AsyncMock(return_value=self.response)

    def set_response(self, content: str):
        self.response = LLMResponse(content=content, model="x", success=True)
        self.chat = AsyncMock(return_value=self.response)

    def set_error(self, error_msg: str = "LLM error"):
        self.response = LLMResponse(content="", model="x", success=False, error=error_msg)
        self.chat = AsyncMock(return_value=self.response)


class MockFailover:
    """Mock FailoverManager for TechnicalAnalyst."""

    def __init__(self):
        self.fetch_with_failover = AsyncMock()


def make_kline(close, high, low, volume, open_, date="2024-01-01"):
    """Make a minimal KlineData object."""
    from app.adapters import KlineData
    return KlineData(symbol="000001", date=date,
                     open=open_, high=high, low=low, close=close, volume=volume)


# ===================================================================
# 1. _extract_json — 8 branches
# ===================================================================
class TestExtractJson:

    def test_wrapped_in_json_markdown(self):
        """Branch: contains ```json```"""
        text = '```json\n{"key": "value"}\n```'
        assert _extract_json(text) == '{"key": "value"}'

    def test_wrapped_in_plain_markdown(self):
        """Branch: contains ``` but no json tag"""
        text = '```\n{"key": "value"}\n```'
        assert _extract_json(text) == '{"key": "value"}'

    def test_starts_and_ends_with_braces(self):
        """Branch: text starts with { and ends with }"""
        text = '{"a": 1, "b": 2}'
        assert _extract_json(text) == '{"a": 1, "b": 2}'

    def test_embedded_json_with_prefix_suffix(self):
        """Branch: JSON embedded in other text, found by first { ... last }"""
        text = 'Prefix text {"target": 42} suffix text'
        assert _extract_json(text) == '{"target": 42}'

    def test_json_markdown_no_closing_tag(self):
        """Branch: ```json present but no closing ``` → returns text after ```json"""
        text = '```json\n{"open": "json"}'
        # start=text.find("```json")+7, end=text.find("```",start)
        # end=-1 so end>start is False, returns text (original)
        assert _extract_json(text) == text

    def test_plain_markdown_no_closing_tag(self):
        """Branch: ``` present but no closing ``` → returns text after ```"""
        text = '```\n{"open": "plain"}'
        # start=text.find("```")+3=3, end=-1 → end>start=False → returns text
        assert _extract_json(text) == text

    def test_no_json_at_all(self):
        """Branch: no markdown wrapper, no braces → returns text as-is"""
        text = "This is plain text with no JSON structure"
        assert _extract_json(text) == text

    def test_empty_string(self):
        """Edge: empty string → returns empty string"""
        assert _extract_json("") == ""

    def test_multiple_json_blocks_returns_first(self):
        """Edge: multiple ```json``` blocks → extracts first one only"""
        text = '```json\n{"first": 1}\n```\nother\n```json\n{"second": 2}\n```'
        assert _extract_json(text) == '{"first": 1}'

    def test_whitespace_and_newline_handling(self):
        """Edge: whitespace around JSON is stripped"""
        text = '  \n  {"hello": "world"}  \n  '
        assert _extract_json(text) == '{"hello": "world"}'

    def test_nested_braces(self):
        """Edge: nested JSON objects"""
        text = '{"outer": {"inner": [1,2,3]}}'
        assert _extract_json(text) == '{"outer": {"inner": [1,2,3]}}'

    def test_text_starts_with_open_brace_not_closed(self):
        """Edge: text has { but no closing } → returns from { to end"""
        text = '{"unclosed'
        assert _extract_json(text) == '{"unclosed'


# ===================================================================
# 2. BaseAgent
# ===================================================================
class TestBaseAgent:

    def test_init_stores_fields(self):
        gw = MockGateway()
        agent = BaseAgent("AGENT-TEST", "测试Agent", "tester", gw)
        assert agent.agent_id == "AGENT-TEST"
        assert agent.name == "测试Agent"
        assert agent.role == "tester"
        assert agent.gateway is gw

    @pytest.mark.asyncio
    async def test_call_llm_formats_correctly(self):
        gw = MockGateway()
        agent = BaseAgent("ID", "Name", "role", gw)
        resp = await agent._call_llm("sys prompt", "user prompt", max_tokens=500)

        gw.chat.assert_awaited_once()
        call_args = gw.chat.call_args
        messages = call_args.kwargs["messages"]
        assert messages[0] == {"role": "system", "content": "sys prompt"}
        assert messages[1] == {"role": "user", "content": "user prompt"}
        assert call_args.kwargs["max_tokens"] == 500
        assert resp is gw.response


# ===================================================================
# 3. FundamentalAnalyst — Phase 1
# ===================================================================
class TestFundamentalAnalyst:

    @pytest.mark.asyncio
    async def test_success_valid_json(self):
        gw = MockGateway()
        gw.set_response('{"scores":{"pe":0.9,"pb":0.8,"roe":0.7,"debt":0.6,"overall":0.85},'
                        '"red_flags":["高负债率"],'
                        '"intrinsic_value_range":{"low":120,"high":180},'
                        '"confidence":0.88}')
        agent = FundamentalAnalyst(gw)
        result = await agent.analyze("000001", "平安银行", fundamentals={"pe": 10})

        assert result.symbol == "000001"
        assert result.name == "平安银行"
        assert result.scores["overall"] == 0.85
        assert result.red_flags == ["高负债率"]
        assert result.intrinsic_value_range["low"] == 120
        assert result.intrinsic_value_range["high"] == 180
        assert result.confidence == 0.88

    @pytest.mark.asyncio
    async def test_success_markdown_wrapped_json(self):
        gw = MockGateway()
        gw.set_response('```json\n{"scores":{"overall":0.7},"red_flags":[],'
                        '"intrinsic_value_range":{"low":50,"high":100},"confidence":0.6}\n```')
        agent = FundamentalAnalyst(gw)
        result = await agent.analyze("000001", "平安银行")
        assert result.scores["overall"] == 0.7
        assert result.confidence == 0.6

    @pytest.mark.asyncio
    async def test_invalid_json_fallback(self):
        """JSONDecodeError → fallback to default values"""
        gw = MockGateway()
        gw.set_response("这不是JSON格式的回复")
        agent = FundamentalAnalyst(gw)
        result = await agent.analyze("000001", "平安银行")

        assert result.symbol == "000001"
        assert result.scores == {"overall": 0.5}  # default
        assert result.red_flags == []
        assert result.intrinsic_value_range == {"low": 0, "high": 0}
        assert result.confidence == 0.3

    @pytest.mark.asyncio
    async def test_missing_keys_uses_defaults(self):
        gw = MockGateway()
        gw.set_response('{"scores":{}}')  # missing red_flags, intrinsic_value_range, confidence
        agent = FundamentalAnalyst(gw)
        result = await agent.analyze("000001", "平安银行")
        assert result.red_flags == []  # .get default
        assert result.intrinsic_value_range == {}
        assert result.confidence == 0.5  # .get default

    @pytest.mark.asyncio
    async def test_fundamentals_none(self):
        """fundamentals=None → passed as {} to LLM"""
        gw = MockGateway()
        agent = FundamentalAnalyst(gw)
        result = await agent.analyze("688256", "寒武纪", fundamentals=None)
        assert result.symbol == "688256"
        # Ensure it didn't crash
        gw.chat.assert_awaited_once()


# ===================================================================
# 4. SentimentAnalyst — Phase 1
# ===================================================================
class TestSentimentAnalyst:

    @pytest.mark.asyncio
    async def test_success_valid_json(self):
        gw = MockGateway()
        gw.set_response('{"sentiment_score":0.75,"sentiment_trend":"improving",'
                        '"key_drivers":["政策利好","资金流入"],"confidence":0.80}')
        agent = SentimentAnalyst(gw)
        result = await agent.analyze("000001", "平安银行", news_headlines=["利好新闻1"])

        assert result.symbol == "000001"
        assert result.sentiment_score == 0.75
        assert result.sentiment_trend == "improving"
        assert result.key_drivers == ["政策利好", "资金流入"]
        assert result.confidence == 0.80

    @pytest.mark.asyncio
    async def test_invalid_json_fallback(self):
        gw = MockGateway()
        gw.set_response("无效回复")
        agent = SentimentAnalyst(gw)
        result = await agent.analyze("000001", "平安银行")

        assert result.sentiment_score == 0
        assert result.sentiment_trend == "stable"
        assert result.key_drivers == []
        assert result.confidence == 0.3

    @pytest.mark.asyncio
    async def test_news_headlines_none_uses_placeholder(self):
        """news_headlines=None → shows "无数据" in LLM prompt"""
        gw = MockGateway()
        agent = SentimentAnalyst(gw)
        result = await agent.analyze("000001", "平安银行", news_headlines=None)
        assert result.symbol == "000001"
        # Verify prompt contains "无数据"
        call_text = gw.chat.call_args.kwargs["messages"][1]["content"]
        assert "无数据" in call_text

    @pytest.mark.asyncio
    async def test_empty_headlines_list(self):
        gw = MockGateway()
        agent = SentimentAnalyst(gw)
        result = await agent.analyze("000001", "平安银行", news_headlines=[])
        call_text = gw.chat.call_args.kwargs["messages"][1]["content"]
        assert "无数据" in call_text


# ===================================================================
# 5. NewsAnalyst — Phase 1
# ===================================================================
class TestNewsAnalyst:

    @pytest.mark.asyncio
    async def test_success_valid_json(self):
        gw = MockGateway()
        gw.set_response('{"event_summary":"重大政策落地","impact_score":0.8,'
                        '"macro_score":0.6,"key_events":[{"event":"降准"}]}')
        agent = NewsAnalyst(gw)
        result = await agent.analyze("000001", "平安银行",
                                     news_items=[{"source": "财联社", "title": "央行降准"}])

        assert result.symbol == "000001"
        assert result.event_summary == "重大政策落地"
        assert result.impact_score == 0.8
        assert result.macro_score == 0.6
        assert result.key_events == [{"event": "降准"}]

    @pytest.mark.asyncio
    async def test_invalid_json_fallback(self):
        gw = MockGateway()
        gw.set_response("not json")
        agent = NewsAnalyst(gw)
        result = await agent.analyze("000001", "平安银行")
        assert result.event_summary == ""
        assert result.impact_score == 0
        assert result.macro_score == 0
        assert result.key_events == []

    @pytest.mark.asyncio
    async def test_news_items_none_placeholder(self):
        """news_items=None → shows "无最新新闻" in prompt"""
        gw = MockGateway()
        agent = NewsAnalyst(gw)
        await agent.analyze("000001", "平安银行", news_items=None)
        call_text = gw.chat.call_args.kwargs["messages"][1]["content"]
        assert "无最新新闻" in call_text

    @pytest.mark.asyncio
    async def test_empty_news_items_list(self):
        gw = MockGateway()
        agent = NewsAnalyst(gw)
        await agent.analyze("000001", "平安银行", news_items=[])
        call_text = gw.chat.call_args.kwargs["messages"][1]["content"]
        assert "无最新新闻" in call_text


# ===================================================================
# 6. TechnicalAnalyst — Phase 1
# ===================================================================
class TestTechnicalAnalyst:

    @pytest.fixture
    def klines_60(self):
        """Generate 60 days of K-line data."""
        return [make_kline(close=10.0 + i * 0.1, high=10.5 + i * 0.1,
                           low=9.5 + i * 0.1, volume=1000000, open_=10.0 + i * 0.1)
                for i in range(60)]

    @pytest.mark.asyncio
    async def test_success_with_indicators(self, klines_60):
        gw = MockGateway()
        gw.set_response('{"trend_direction":"bullish","support_levels":[10.5,10.0],'
                        '"resistance_levels":[15.0,16.0],'
                        '"patterns":["金叉","放量突破"],"confidence":0.85}')
        failover = MockFailover()
        failover.fetch_with_failover.return_value = klines_60

        agent = TechnicalAnalyst(gw, failover)
        result = await agent.analyze("000001", "平安银行")

        assert result.symbol == "000001"
        assert result.trend_direction == "bullish"
        assert result.support_levels == [10.5, 10.0]
        assert result.patterns == ["金叉", "放量突破"]
        assert result.confidence == 0.85
        # compute_all_indicators returns dict with keys even without talib (values may be None)
        assert isinstance(result.indicators, dict)
        assert len(result.indicators) > 0
        failover.fetch_with_failover.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_indicator_computation_exception(self):
        """Exception in indicator fetch/computation → indicators={}.
        LLM still gives a response (default gateway response parsed for trend_direction)."""
        gw = MockGateway()
        failover = MockFailover()
        failover.fetch_with_failover.side_effect = RuntimeError("Data source unavailable")

        agent = TechnicalAnalyst(gw, failover)
        result = await agent.analyze("000001", "平安银行")
        assert result.indicators == {}
        assert result.trend_direction == "sideways"  # default response has no trend_direction key
        # Prompt should show N/A for close
        call_text = gw.chat.call_args.kwargs["messages"][1]["content"]
        assert "N/A" in call_text

    @pytest.mark.asyncio
    async def test_invalid_json_fallback(self):
        gw = MockGateway()
        gw.set_response("not json")
        failover = MockFailover()
        failover.fetch_with_failover.return_value = [
            make_kline(close=10.0, high=10.5, low=9.5, volume=1000, open_=10.0)
            for _ in range(60)
        ]
        agent = TechnicalAnalyst(gw, failover)
        result = await agent.analyze("000001", "平安银行")
        assert result.trend_direction == "sideways"
        assert result.support_levels == []
        assert result.resistance_levels == []
        assert result.confidence == 0.3

    @pytest.mark.asyncio
    async def test_empty_klines_safety(self):
        """When failover returns empty list → closes is empty → N/A for close."""
        gw = MockGateway()
        failover = MockFailover()
        failover.fetch_with_failover.return_value = []

        agent = TechnicalAnalyst(gw, failover)
        # Should not crash — indicator computation handles empty arrays
        result = await agent.analyze("000001", "平安银行")
        assert result.symbol == "000001"


# ===================================================================
# 7. BullishResearcher — Phase 2
# ===================================================================
class TestBullishResearcher:

    @pytest.mark.asyncio
    async def test_bullet_points_dash(self):
        """Response with - prefixed bullet points"""
        gw = MockGateway()
        gw.set_response("- 政策利好支持芯片产业\n- 业绩超预期\n- 估值处于低位\n- 外资持续流入\n- 国产替代加速\n- extra line")
        agent = BullishResearcher(gw)
        result = await agent.debate("688256", "寒武纪", {"fundamental": {}})
        assert len(result) == 5  # capped at 5
        assert result[0] == "政策利好支持芯片产业"
        assert result[1] == "业绩超预期"

    @pytest.mark.asyncio
    async def test_bullet_points_star(self):
        """Response with * prefixed bullet points"""
        gw = MockGateway()
        gw.set_response("* 利好消息\n* 资金面改善\n* 技术突破")
        agent = BullishResearcher(gw)
        result = await agent.debate("688256", "寒武纪", {})
        assert len(result) == 3
        assert result[0] == "* 利好消息"

    @pytest.mark.asyncio
    async def test_no_bullets_fallback_first_200_chars(self):
        """Response without bullets → fallback to resp.content[:200]"""
        gw = MockGateway()
        long_text = "这是一个很长的没有项目符号的回复文本" * 20
        gw.set_response(long_text)
        agent = BullishResearcher(gw)
        result = await agent.debate("688256", "寒武纪", {})
        assert len(result) == 1
        assert result[0] == long_text[:200]

    @pytest.mark.asyncio
    async def test_mixed_format(self):
        """Some dash, some non-dash lines"""
        gw = MockGateway()
        gw.set_response("- point1\nNot a bullet\n- point2\nAlso not")
        agent = BullishResearcher(gw)
        result = await agent.debate("688256", "寒武纪", {})
        assert result == ["point1", "point2"]


# ===================================================================
# 8. BearishResearcher — Phase 2
# ===================================================================
class TestBearishResearcher:

    @pytest.mark.asyncio
    async def test_bullet_points_dash(self):
        gw = MockGateway()
        gw.set_response("- 估值过高\n- 行业竞争加剧\n- 毛利率下滑\n- 大股东减持\n- 政策风险")
        agent = BearishResearcher(gw)
        result = await agent.debate("688256", "寒武纪", {})
        assert len(result) == 5
        assert "估值过高" in result

    @pytest.mark.asyncio
    async def test_no_bullets_fallback(self):
        gw = MockGateway()
        gw.set_response("No clear bullet risks here just prose.")
        agent = BearishResearcher(gw)
        result = await agent.debate("688256", "寒武纪", {})
        assert result == ["No clear bullet risks here just prose."]

    @pytest.mark.asyncio
    async def test_response_truncated_at_3000_in_prompt(self):
        """Reports dict value is sliced to [:3000] in prompt"""
        gw = MockGateway()
        agent = BearishResearcher(gw)
        large_reports = {"fundamental": {"x": "y" * 5000}}
        await agent.debate("688256", "寒武纪", large_reports)
        call_text = gw.chat.call_args.kwargs["messages"][1]["content"]
        # The json.dumps output should be truncated
        assert len(call_text) <= 4000  # rough bound (prompt + truncated json)


# ===================================================================
# 9. TradingAgent — Phase 3
# ===================================================================
class TestTradingAgent:

    @pytest.mark.asyncio
    async def test_success_buy_order(self):
        gw = MockGateway()
        gw.set_response('{"side":"buy","quantity":1000,"stop_loss":9.5,'
                        '"take_profit":12.0,"rationale":"强势突破","confidence":0.82}')
        agent = TradingAgent(gw)
        result = await agent.generate_proposal("000001", "平安银行", {"final_stance": "bullish"})

        assert result.side == "buy"
        assert result.quantity == 1000
        assert result.stop_loss == 9.5
        assert result.take_profit == 12.0
        assert result.rationale == "强势突破"
        assert result.confidence == 0.82

    @pytest.mark.asyncio
    async def test_invalid_json_fallback_hold(self):
        gw = MockGateway()
        gw.set_response("invalid response")
        agent = TradingAgent(gw)
        result = await agent.generate_proposal("000001", "平安银行", {})

        assert result.side == "hold"
        assert result.quantity == 0
        assert result.stop_loss == 0
        assert result.take_profit == 0
        assert result.rationale == "无法生成交易提案"
        assert result.confidence == 0.1

    @pytest.mark.asyncio
    async def test_missing_side_defaults_to_hold(self):
        gw = MockGateway()
        gw.set_response('{"quantity":500,"confidence":0.6}')
        agent = TradingAgent(gw)
        result = await agent.generate_proposal("000001", "平安银行", {})
        assert result.side == "hold"  # .get default


# ===================================================================
# 10. RiskManager — Phase 4
# ===================================================================
class TestRiskManager:

    @pytest.mark.asyncio
    async def test_success_high_risk(self):
        gw = MockGateway()
        gw.set_response('{"var_value":-5000.0,"cvar_value":-8000.0,'
                        '"concentration_risk":0.35,'
                        '"stress_test_results":{"recession":-15.0},'
                        '"risk_level":"high","recommendations":["降低仓位"]}')
        agent = RiskManager(gw)

        from app.schemas import TradeProposalPayload
        proposal = TradeProposalPayload(
            symbol="000001", name="平安银行", side="buy", quantity=1000,
            stop_loss=9.5, take_profit=12.0
        )
        result = await agent.assess("000001", "平安银行", proposal)

        assert result.symbol == "000001"
        assert result.var_value == -5000.0
        assert result.cvar_value == -8000.0
        assert result.concentration_risk == 0.35
        assert result.stress_test_results == {"recession": -15.0}
        assert result.risk_level == "high"
        assert result.recommendations == ["降低仓位"]

    @pytest.mark.asyncio
    async def test_invalid_json_fallback(self):
        gw = MockGateway()
        gw.set_response("bad")
        agent = RiskManager(gw)

        from app.schemas import TradeProposalPayload
        proposal = TradeProposalPayload(
            symbol="000001", name="平安银行", side="hold", quantity=0,
            stop_loss=0, take_profit=0
        )
        result = await agent.assess("000001", "平安银行", proposal)

        assert result.var_value == 0
        assert result.cvar_value == 0
        assert result.concentration_risk == 0
        assert result.risk_level == "medium"
        assert result.recommendations == []


# ===================================================================
# 11. PortfolioManager — Phase 4
# ===================================================================
class TestPortfolioManager:

    @pytest.mark.asyncio
    async def test_success_approval(self):
        gw = MockGateway()
        gw.set_response('{"status":"approved","allocated_amount":50000.0,'
                        '"rationale":"风险可控","conditions":["设置止损"]}')
        agent = PortfolioManager(gw)

        from app.schemas import TradeProposalPayload, RiskAssessmentPayload
        proposal = TradeProposalPayload(
            symbol="000001", name="平安银行", side="buy", quantity=1000,
            stop_loss=9.5, take_profit=12.0
        )
        risk = RiskAssessmentPayload(symbol="000001", risk_level="low")

        result = await agent.approve("000001", "平安银行", proposal, risk)

        assert result.status == "approved"
        assert result.allocated_amount == 50000.0
        assert result.rationale == "风险可控"
        assert result.conditions == ["设置止损"]

    @pytest.mark.asyncio
    async def test_invalid_json_fallback_rejected(self):
        gw = MockGateway()
        gw.set_response("garbage")
        agent = PortfolioManager(gw)

        from app.schemas import TradeProposalPayload, RiskAssessmentPayload
        proposal = TradeProposalPayload(
            symbol="000001", name="平安银行", side="buy", quantity=1000,
            stop_loss=9.5, take_profit=12.0
        )
        risk = RiskAssessmentPayload(symbol="000001", risk_level="high")

        result = await agent.approve("000001", "平安银行", proposal, risk)

        assert result.status == "rejected"
        assert result.allocated_amount == 0
        assert result.rationale == "审批失败"
        assert result.conditions == []


# ===================================================================
# 12. AgentWorkflow — 5-Phase Orchestrator
# ===================================================================
class TestAgentWorkflow:

    def _make_gateway_with_responses(self, responses: list):
        """Create gateway that returns responses in sequence.
        Order in run():
          [fundamental, sentiment, news, technical] — phase 1 parallel
          [bull_open, bear_open] — phase 2 round 1 parallel
          [bull_rebuttal, bear_rebuttal] — phase 2 round 2 parallel
          [judge] — research manager synthesize
          [trader] — phase 3
          [risk] — phase 4
          [pm] — phase 4
        """
        gw = MockGateway()
        gw.chat = AsyncMock()
        gw.chat.side_effect = [
            LLMResponse(content=r, model="x", success=True)
            for r in responses
        ]
        return gw

    def _default_responses(self):
        return [
            # Phase 1: 4 analysts
            '{"scores":{"overall":0.8},"red_flags":[],"intrinsic_value_range":{"low":100,"high":200},"confidence":0.9}',
            '{"sentiment_score":0.6,"sentiment_trend":"stable","key_drivers":[],"confidence":0.7}',
            '{"event_summary":"无","impact_score":0.0,"macro_score":0.0,"key_events":[]}',
            '{"trend_direction":"bullish","support_levels":[10],"resistance_levels":[20],"patterns":[],"confidence":0.8}',
            # Phase 2 round 1: bull + bear openings
            "- 政策利好\n- 资金流入\n- 估值合理",
            "- 行业风险\n- 竞争加剧",
            # Phase 2 round 2: rebuttals
            "- 流动性支撑估值\n- 资金持续流入",
            "- 估值陷阱风险\n- 资金反转风险",
            # Research manager judgement
            '{"final_stance":"bullish","conviction_level":0.6,"key_risks":["竞争加剧","估值偏高"],"judge_summary":"多方论据更扎实"}',
            # Phase 3: trader
            '{"side":"buy","quantity":500,"stop_loss":9.5,"take_profit":15.0,"rationale":"综合看涨","confidence":0.75}',
            # Phase 4: risk
            '{"var_value":-2000,"cvar_value":-5000,"concentration_risk":0.2,"stress_test_results":{},"risk_level":"medium","recommendations":[]}',
            # Phase 4: pm
            '{"status":"approved","allocated_amount":30000,"rationale":"批准","conditions":[]}',
        ]

    def _make_klines(self):
        failover = MockFailover()
        failover.fetch_with_failover.return_value = [
            make_kline(close=10.0, high=11.0, low=9.0, volume=1000, open_=10.0)
            for _ in range(60)
        ]
        return failover

    def _routed_gateway(self, overrides=None):
        """Content-routed gateway: maps system-prompt substrings to responses.

        Robust against call-order changes and parse-failure retries (which
        consume extra calls and would misalign index-based side_effect lists).
        """
        d = self._default_responses()
        routes = {
            "基本面分析师": d[0],
            "市场情绪分析专家": d[1],
            "宏观新闻分析专家": d[2],
            "专业技术分析师": d[3],
            "从积极面评估": d[4],
            "从风险面寻找": d[5],
            "请针对对方的看跌论点": d[6],
            "请针对对方的看涨论点": d[7],
            "研究经理": d[8],
            "专业交易员": d[9],
            "风险管理专家": d[10],
            "投资组合经理": d[11],
        }
        if overrides:
            routes.update(overrides)

        gw = MockGateway()

        async def route_chat(*args, **kwargs):
            messages = kwargs.get("messages") or (args[0] if args else [])
            system = messages[0].get("content", "") if messages else ""
            for key, content in routes.items():
                if key in system:
                    return LLMResponse(content=content, model="x", success=True)
            raise AssertionError(f"No route matched system prompt: {system[:60]}")

        gw.chat = AsyncMock(side_effect=route_chat)
        return gw

    @pytest.mark.asyncio
    async def test_happy_path_all_phases(self):
        """Complete workflow with all agents returning valid results."""
        gw = self._make_gateway_with_responses(self._default_responses())
        failover = self._make_klines()

        workflow = AgentWorkflow(gw, failover)
        result = await workflow.run("000001", "平安银行")

        assert result["symbol"] == "000001"
        assert result["name"] == "平安银行"
        assert result["phase"] == 5
        assert result["status"] == "completed"

        # Phase 1 reports
        assert result["fundamental_report"] is not None
        assert result["sentiment_report"] is not None
        assert result["news_report"] is not None
        assert result["technical_report"] is not None

        # Phase 2 debate — stance decided by research manager
        debate = result["debate_conclusion"]
        assert debate["final_stance"] == "bullish"
        assert len(debate["bull_arguments"]) == 3
        assert len(debate["bear_arguments"]) == 2
        # conviction comes from the judge now
        assert debate["conviction_level"] == pytest.approx(0.6)
        # key_risks from the judge
        assert debate["key_risks"] == ["竞争加剧", "估值偏高"]

        # Two-round debate record
        judgement = result["research_judgement"]
        assert judgement["rounds"] == 2
        assert len(judgement["bull_rebuttals"]) > 0
        assert len(judgement["bear_rebuttals"]) > 0

        # Phase 3 — quantity overwritten by deterministic risk model:
        # risk_amount = 1_000_000 * 0.02 * 0.75 = 15000; per-share risk = max(10-9.5, 10*0.01) = 0.5
        assert result["trade_proposal"]["side"] == "buy"
        assert result["trade_proposal"]["quantity"] == 30000
        assert result["trade_proposal"]["quantity_method"] == "risk_parity_deterministic"

        # Phase 4 — VaR/CVaR computed deterministically from flat closes → 0
        assert result["risk_assessment"]["risk_level"] == "medium"
        assert result["risk_assessment"]["var_value"] == 0.0
        assert result["risk_assessment"]["cvar_value"] == 0.0
        assert result["approval_decision"]["status"] == "approved"

        # Meta
        assert result["errors"] == []
        assert "completed_at" in result
        assert len(result["session_id"]) > 0
        assert len(result["correlation_id"]) > 0

    @pytest.mark.asyncio
    async def test_tie_breaker_bearish_wins(self):
        """Judge unparseable → heuristic fallback: BEAR has more arguments → bearish"""
        gw = self._routed_gateway(overrides={
            "从积极面评估": "- 一点利好",                                # bull 1 argument
            "从风险面寻找": "- 风险1\n- 风险2\n- 风险3\n- 风险4",       # bear 4 arguments
            "研究经理": "研究经理暂时无法给出裁决",                      # judge fails → heuristic
        })
        failover = self._make_klines()

        workflow = AgentWorkflow(gw, failover)
        result = await workflow.run("000001", "平安银行")

        debate = result["debate_conclusion"]
        assert debate["final_stance"] == "bearish"
        # conviction = |1-4| / max(5, 1) = 3/5 = 0.6
        assert debate["conviction_level"] == pytest.approx(0.6)
        assert any("研究经理" in e for e in result["errors"])

    @pytest.mark.asyncio
    async def test_tie_breaker_equal_neutral(self):
        """Judge unparseable + equal argument counts → neutral"""
        gw = self._routed_gateway(overrides={
            "从积极面评估": "- arg1\n- arg2",
            "从风险面寻找": "- arg1\n- arg2",
            "研究经理": "无法裁决",
        })
        failover = self._make_klines()

        workflow = AgentWorkflow(gw, failover)
        result = await workflow.run("000001", "平安银行")

        debate = result["debate_conclusion"]
        assert debate["final_stance"] == "neutral"
        assert debate["conviction_level"] == 0.0

    @pytest.mark.asyncio
    async def test_conviction_zero_when_both_zero(self):
        """Both bull and bear return 0 arguments."""
        gw = self._routed_gateway(overrides={
            "从积极面评估": "no bullets at all",
            "从风险面寻找": "no bullets either",
            "研究经理": "无法裁决",
        })
        failover = self._make_klines()

        workflow = AgentWorkflow(gw, failover)
        result = await workflow.run("000001", "平安银行")

        debate = result["debate_conclusion"]
        # each gets 1 fallback → 1 vs 1 → neutral
        assert debate["final_stance"] == "neutral"
        assert debate["conviction_level"] == 0.0

    @pytest.mark.asyncio
    async def test_phase1_partial_failure(self):
        """Some analyst agents throw exceptions → errors populated, reports are None."""
        gw = self._make_gateway_with_responses(self._default_responses())

        gw2 = MockGateway()
        call_count = [0]

        async def selective_fail(*args, **kwargs):
            idx = call_count[0]
            call_count[0] += 1
            if idx == 0:  # fundamental fails
                raise RuntimeError("Fundamental data error")
            if idx == 2:  # news fails
                raise RuntimeError("News source error")
            return LLMResponse(content=self._default_responses()[idx], model="x", success=True)

        gw2.chat = AsyncMock(side_effect=selective_fail)

        failover = self._make_klines()

        workflow = AgentWorkflow(gw2, failover)
        result = await workflow.run("000001", "平安银行")

        assert result["status"] == "completed_with_errors"
        assert len(result["errors"]) == 2  # fundamental + news failed
        assert result["fundamental_report"] is None
        assert result["news_report"] is None
        assert result["sentiment_report"] is not None
        assert result["technical_report"] is not None

    @pytest.mark.asyncio
    async def test_phase2_debate_exception(self):
        """Bull debate agent throws → bull args empty; judge unparseable →
        heuristic picks bearish (0 vs 2)."""
        gw = self._routed_gateway(overrides={"研究经理": "无法裁决"})

        inner_route = gw.chat.side_effect

        async def fail_bull(*args, **kwargs):
            messages = kwargs.get("messages") or (args[0] if args else [])
            if messages and "从积极面评估" in messages[0].get("content", ""):
                raise RuntimeError("Debate timeout")
            return await inner_route(*args, **kwargs)

        gw.chat = AsyncMock(side_effect=fail_bull)
        failover = self._make_klines()

        workflow = AgentWorkflow(gw, failover)
        result = await workflow.run("000001", "平安银行")

        debate = result["debate_conclusion"]
        assert debate["bull_arguments"] == []  # failed → empty
        assert len(debate["bear_arguments"]) == 2  # bear succeeded
        assert debate["final_stance"] == "bearish"  # heuristic: bear 2 > bull 0

    @pytest.mark.asyncio
    async def test_phase1_all_fail_still_completes(self):
        """All 4 phase-1 agents fail → all reports None, debate gets empty dicts.
        Phases 2-4 still proceed since debate/generate/assess use same gateway."""
        responses = self._default_responses()
        # Make phase 1 (first 4 calls) fail, rest succeed
        call_count = [0]

        async def selective_fail(*args, **kwargs):
            idx = call_count[0]
            call_count[0] += 1
            if idx < 4:  # first 4 calls = phase 1 analysts → fail
                raise RuntimeError(f"Analyst {idx} failed")
            return LLMResponse(content=responses[idx], model="x", success=True)

        gw = MockGateway()
        gw.chat = AsyncMock(side_effect=selective_fail)

        failover = self._make_klines()

        workflow = AgentWorkflow(gw, failover)
        result = await workflow.run("000001", "平安银行")

        assert result["status"] == "completed_with_errors"
        assert len(result["errors"]) == 4  # 4 analysts failed
        assert result["fundamental_report"] is None
        assert result["sentiment_report"] is None
        assert result["news_report"] is None
        assert result["technical_report"] is None
        # Debate gets empty dicts
        debate = result["debate_conclusion"]
        assert debate["bull_arguments"] != []  # debate succeeds with empty but real response
        assert debate["bear_arguments"] != []

    @pytest.mark.asyncio
    async def test_empty_name_defaults_to_empty_string(self):
        """name="" is valid → workflow still runs."""
        gw = self._make_gateway_with_responses(self._default_responses())
        failover = self._make_klines()
        workflow = AgentWorkflow(gw, failover)
        result = await workflow.run("000001", "")
        assert result["name"] == ""
        assert result["status"] == "completed"

    @pytest.mark.asyncio
    async def test_news_items_passed_to_news_analyst(self):
        """Verify that news_items parameter reaches NewsAnalyst.analyze()"""
        gw = self._make_gateway_with_responses(self._default_responses())
        failover = self._make_klines()
        workflow = AgentWorkflow(gw, failover)
        news_items = [{"source": "测试", "title": "标题"}]
        result = await workflow.run("000001", "平安银行", news_items=news_items)
        assert result["news_report"] is not None

    @pytest.mark.asyncio
    async def test_sentiment_receives_headlines(self):
        """BUG-2 regression: sentiment analyst must receive real headlines."""
        gw = self._make_gateway_with_responses(self._default_responses())
        captured_prompts = []

        async def capturing_chat(*args, **kwargs):
            messages = kwargs.get("messages") or (args[0] if args else [])
            for m in messages:
                if m.get("role") == "user":
                    captured_prompts.append(m.get("content", ""))
            return await AsyncMock(return_value=LLMResponse(
                content='{"sentiment_score":0.5,"sentiment_trend":"up","key_drivers":[],"confidence":0.8}',
                model="x", success=True))()

        gw.chat = AsyncMock(side_effect=capturing_chat)
        failover = MockFailover()
        failover.fetch_with_failover.return_value = [
            make_kline(close=10.0, high=11.0, low=9.0, volume=1000, open_=10.0)
            for _ in range(60)
        ]
        # Also serve a NEWS fetch returning dict items
        news_payload = [{"source": "财联社", "title": "重大利好政策出台"}]

        async def route_fetch(data_type, symbol, **kwargs):
            from app.adapters import DataType as DT
            if data_type == DT.KLINE:
                return [make_kline(close=10.0, high=11.0, low=9.0, volume=1000, open_=10.0)
                        for _ in range(60)]
            if data_type == DT.NEWS:
                return news_payload
            return []

        failover.fetch_with_failover = AsyncMock(side_effect=route_fetch)

        workflow = AgentWorkflow(gw, failover)
        await workflow.run("000001", "平安银行")

        sentiment_prompt = next((p for p in captured_prompts if "情绪分析" in p), "")
        assert "重大利好政策出台" in sentiment_prompt

    @pytest.mark.asyncio
    async def test_run_stream_yields_phases_then_result(self):
        """run_stream emits progress events then a final result event."""
        gw = self._make_gateway_with_responses(self._default_responses())
        failover = self._make_klines()
        workflow = AgentWorkflow(gw, failover)

        events = []
        async for kind, payload in workflow.run_stream("000001", "平安银行"):
            events.append((kind, payload))

        kinds = [k for k, _ in events]
        assert kinds[0] == "phase"
        assert kinds[-1] == "result"
        assert kinds.count("result") == 1
        phases = [p for k, p in events if k == "phase"]
        assert [p["phase"] for p in phases] == [0, 1, 1, 2, 2, 3, 3, 4, 4, 5]
        result = events[-1][1]
        assert result["status"] == "completed"
        assert result["trade_proposal"]["quantity_method"] == "risk_parity_deterministic"

    @pytest.mark.asyncio
    async def test_session_and_correlation_ids_are_unique(self):
        """Each run() call generates new unique UUIDs."""
        gw1 = self._make_gateway_with_responses(self._default_responses())
        gw2 = self._make_gateway_with_responses(self._default_responses())
        failover = self._make_klines()

        w1 = AgentWorkflow(gw1, failover)
        w2 = AgentWorkflow(gw2, failover)
        r1 = await w1.run("000001", "平安银行")
        r2 = await w2.run("000002", "万科A")

        assert r1["session_id"] != r2["session_id"]
        assert r1["correlation_id"] != r2["correlation_id"]


# ===================================================================
# 13. Agent ID Constants
# ===================================================================
class TestAgentIds:

    def test_all_ids_are_unique(self):
        ids = [AGENT_FUNDAMENTAL, AGENT_SENTIMENT, AGENT_NEWS, AGENT_TECHNICAL,
               AGENT_BULL, AGENT_BEAR, AGENT_TRADER, AGENT_RISK, AGENT_PM]
        assert len(ids) == len(set(ids))

    def test_all_ids_follow_pattern(self):
        ids = [AGENT_FUNDAMENTAL, AGENT_SENTIMENT, AGENT_NEWS, AGENT_TECHNICAL,
               AGENT_BULL, AGENT_BEAR, AGENT_TRADER, AGENT_RISK, AGENT_PM]
        for aid in ids:
            assert aid.startswith("AGENT-")
            assert len(aid) > 7


# ===================================================================
# 14. FallbackAnalyzer — indicator argument-order regression (BUG-1)
# ===================================================================
class TestFallbackAnalyzerIndicators:

    def _make_workflow(self, klines):
        failover = MockFailover()

        async def route_fetch(data_type, symbol, **kwargs):
            from app.adapters import DataType as DT
            if data_type == DT.KLINE:
                return klines
            if data_type == DT.NEWS:
                return []
            return []

        failover.fetch_with_failover = AsyncMock(side_effect=route_fetch)
        return AgentWorkflow(MockGateway(), failover)

    @pytest.mark.asyncio
    async def test_indicators_computed_from_correctly_ordered_arrays(self):
        """BUG-1: closes were previously passed as `high` → every indicator
        was computed on the wrong series. ma5 must equal MA5(closes)."""
        pytest.importorskip("talib")
        from app.schemas import ApprovalDecisionPayload, TradeProposalPayload
        from app.adapters import KlineData

        n = 60
        klines = [
            KlineData(symbol="000001", date=f"2024-{(i // 28) + 1:02d}-{(i % 28) + 1:02d}",
                      open=float(i), high=float(i) + 10.0,
                      low=float(i), close=float(i), volume=1000)
            for i in range(n)
        ]
        workflow = self._make_workflow(klines)

        approval = ApprovalDecisionPayload(symbol="000001", status="approved",
                                           allocated_amount=0, rationale="", conditions=[])
        trade = TradeProposalPayload(symbol="000001", name="测试股", side="hold", quantity=0,
                                     stop_loss=0, take_profit=0, rationale="", confidence=0.5)

        from app.schemas import DebateConclusionPayload
        debate_payload = DebateConclusionPayload(
            symbol="000001", name="x", bull_arguments=[], bear_arguments=[],
            final_stance="neutral", conviction_level=0.0, key_risks=[])

        report = await workflow.fallback_analyzer.generate(
            "000001", "测试股", approval, trade, debate_payload,
        )

        indicators = (report.get("trend") or {}).get("indicators") or {}
        # closes are 0..59 → MA5 of last five = mean(55..59) = 57.0
        expected_ma5 = sum(range(55, 60)) / 5
        assert indicators.get("ma5") == pytest.approx(expected_ma5, abs=0.01)
        # If the bug were present (highs passed as first arg), ma5 would be 67.0
        assert indicators.get("ma5") != pytest.approx(expected_ma5 + 10, abs=0.01)


# ===================================================================
# 15. risk_metrics — deterministic VaR/CVaR & position sizing
# ===================================================================
class TestRiskMetrics:

    def test_var_cvar_known_sample(self):
        from app.risk_metrics import historical_var_cvar
        # 30 returns: 27 flat days + three -10% days (tail > 5% so the
        # 95% quantile lands inside the loss region)
        closes = [100.0]
        for r in [0.0] * 27 + [-0.10] * 3:
            closes.append(closes[-1] * (1 + r))
        result = historical_var_cvar(np.array(closes), position_value=100_000,
                                     confidence_level=0.95, horizon_days=1)
        assert not result["insufficient_history"]
        assert result["sample_size"] >= 30
        # The 5% quantile return is -10% → VaR = 10% of position
        assert result["var_value"] == pytest.approx(10_000, rel=0.05)
        assert result["cvar_value"] >= result["var_value"]

    def test_insufficient_history_flag(self):
        from app.risk_metrics import historical_var_cvar
        result = historical_var_cvar(np.array([100.0] * 10), position_value=50_000)
        assert result["insufficient_history"] is True
        assert result["var_value"] == 0.0
        assert result["cvar_value"] == 0.0

    def test_higher_confidence_never_lower_var(self):
        from app.risk_metrics import historical_var_cvar
        rng = np.random.default_rng(42)
        closes = 100 * np.cumprod(1 + rng.normal(0, 0.02, 120))
        v95 = historical_var_cvar(closes, 1_000_000, confidence_level=0.95)
        v99 = historical_var_cvar(closes, 1_000_000, confidence_level=0.99)
        assert v99["var_value"] >= v95["var_value"]

    def test_position_size_math(self):
        from app.risk_metrics import position_size
        # conf .75, price 10, stop 9.5 → 15000 / 0.5 = 30000 shares
        assert position_size(0.75, 10.0, 9.5) == 30000

    def test_position_size_guards(self):
        from app.risk_metrics import position_size
        assert position_size(0.75, 0.0, 9.5) == 0
        assert position_size(0.0, 10.0, 20.0) > 0
        assert position_size(-5.0, 10.0, 9.5) > 0
        # tighter stop → smaller per-share risk → larger size
        assert position_size(0.9, 10.0, 9.999) > position_size(0.9, 10.0, 9.0)


# ===================================================================
# 16. ResearchManager — judge synthesis with fallback
# ===================================================================
class TestResearchManager:

    @pytest.mark.asyncio
    async def test_judge_success_synthesizes_stance(self):
        gw = MockGateway(response=LLMResponse(
            content='{"final_stance":"bullish","conviction_level":0.7,"key_risks":["r1"],"judge_summary":"多方胜出"}',
            model="x", success=True))
        rm = __import__("app.agents", fromlist=["ResearchManager"]).ResearchManager(gw)
        payload = await rm.synthesize("000001", "X", ["b1"], ["s1"], ["br1"], ["sr1"])
        assert payload is not None
        assert payload.final_stance == "bullish"
        assert payload.conviction_level == pytest.approx(0.7)
        assert payload.key_risks == ["r1"]
        assert rm.last_parse_ok is True

    @pytest.mark.asyncio
    async def test_judge_failure_returns_none_for_heuristic_fallback(self):
        gw = MockGateway()
        gw.set_response("完全无法解析的输出")
        rm = __import__("app.agents", fromlist=["ResearchManager"]).ResearchManager(gw)
        payload = await rm.synthesize("000001", "X", ["b1"], ["s1"], [], [])
        assert payload is None
        assert rm.last_parse_ok is False

    @pytest.mark.asyncio
    async def test_judge_invalid_stance_rejected(self):
        gw = MockGateway(response=LLMResponse(
            content='{"final_stance":"sideways_up","conviction_level":0.5,"key_risks":[]}',
            model="x", success=True))
        rm = __import__("app.agents", fromlist=["ResearchManager"]).ResearchManager(gw)
        assert await rm.synthesize("000001", "X", [], [], [], []) is None
