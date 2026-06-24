# ============================================================================
# AI Stock Analysis Platform - Analysis Engines
# ============================================================================
from __future__ import annotations

import asyncio
import json
import logging
import math
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any, AsyncGenerator

import numpy as np
import pandas as pd

from app.adapters import FailoverManager, DataType, KlineData, RealtimeQuote
from app.indicators import compute_all_indicators
from app.model_gateway import ModelGateway, LLMResponse
from app.cache import (
    cache_realtime_quote, get_cached_quote,
    cache_hot_topics, get_cached_hot_topics,
)

logger = logging.getLogger(__name__)


# -- Hot Topic Tracking Engine (HT-001 ~ HT-004) -------------------------------
class HotTopicEngine:
    """Real-time hot topic tracking and ranking engine."""

    def __init__(self, gateway: ModelGateway):
        self.gateway = gateway

    @staticmethod
    def compute_heat_index(news_count: int, spread_rate: float,
                           market_cap_coverage: float) -> float:
        """Compute heat index: HT-003 formula."""
        return (
            0.4 * (news_count / 10.0) +
            0.35 * spread_rate +
            0.25 * market_cap_coverage
        ) * 100

    async def generate_ai_conclusion(self, topic_name: str, news_summaries: List[str],
                                     related_stocks: List[Dict]) -> str:
        """Generate AI conclusion for hot topic (AC-001)."""
        prompt = f"""你是一位专业的金融分析师。请基于以下热点话题信息，生成200-500字的结构化分析结论。

## 热点话题
{topic_name}

## 相关新闻摘要
{chr(10).join(f'- {s}' for s in news_summaries[:5])}

## 关联股票
{chr(10).join(f'- {s.get("name", "")} ({s.get("symbol", "")})' for s in related_stocks[:5])}

## 请输出：
1. 热点事件摘要
2. 影响分析
3. 关联标的推荐
4. 操作建议
5. 风险提示

使用结构化 Markdown 格式输出，风险用 🔴🟡🟢 标注。
"""
        response = await self.gateway.chat(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=800,
        )
        return response.content if response.success else "AI 分析暂不可用"


# -- AI Trend Analysis Engine (TA-001 ~ TA-005) --------------------------------
class TrendAnalysisEngine:
    """AI-driven trend analysis using LLM + technical indicators."""

    def __init__(self, gateway: ModelGateway, failover: FailoverManager):
        self.gateway = gateway
        self.failover = failover

    async def analyze(self, symbol: str, name: str = "",
                      frequency: str = "D") -> Dict[str, Any]:
        """Run full AI trend analysis for a stock."""

        # 1. Fetch K-line data
        klines = await self.failover.fetch_with_failover(
            DataType.KLINE, symbol, frequency=frequency
        )
        if not klines or len(klines) < 20:
            return {"error": "Insufficient K-line data", "symbol": symbol}

        # 2. Compute technical indicators
        closes = np.array([k.close for k in klines])
        highs = np.array([k.high for k in klines])
        lows = np.array([k.low for k in klines])
        volumes = np.array([k.volume for k in klines])
        opens = np.array([k.open for k in klines])

        indicators = compute_all_indicators(highs, lows, closes, volumes, opens)

        # 3. Build analysis context
        recent_klines = klines[-20:]
        kline_table = "\n".join(
            f"| {k.date} | {k.open:.2f} | {k.high:.2f} | {k.low:.2f} | {k.close:.2f} | {k.volume} |"
            for k in recent_klines
        )

        prompt = f"""你是一位专业的技术分析师。请基于以下数据进行分析：

## K线数据（最近20个交易日）
| 日期 | 开盘 | 最高 | 最低 | 收盘 | 成交量 |
|------|------|------|------|------|--------|
{kline_table}

## 技术指标
- MA5: {indicators.get('ma5', 'N/A')}, MA10: {indicators.get('ma10', 'N/A')}, MA20: {indicators.get('ma20', 'N/A')}, MA60: {indicators.get('ma60', 'N/A')}
- MACD: DIF={indicators.get('macd_dif', 'N/A')}, DEA={indicators.get('macd_dea', 'N/A')}, 信号={indicators.get('macd_signal', 'N/A')}
- RSI(6): {indicators.get('rsi_6', 'N/A')}, RSI(14): {indicators.get('rsi_14', 'N/A')}, RSI(24): {indicators.get('rsi_24', 'N/A')}
- KDJ: K={indicators.get('kdj_k', 'N/A')}, D={indicators.get('kdj_d', 'N/A')}, J={indicators.get('kdj_j', 'N/A')}
- CCI(14): {indicators.get('cci_14', 'N/A')}
- 布林带: 上轨={indicators.get('bollinger_upper', 'N/A')}, 中轨={indicators.get('bollinger_middle', 'N/A')}, 下轨={indicators.get('bollinger_lower', 'N/A')}
- ATR(14): {indicators.get('atr_14', 'N/A')}
- 成交量: 最近量比={indicators.get('volume_ratio', 'N/A')}

## 请输出：
1. **趋势判断**（看涨/看跌/震荡）及置信度（高/中/低）
2. **关键支撑位**和**阻力位**
3. **识别到的技术形态**
4. **判断依据**（列举2-3个关键技术信号）
5. **风险提示**
"""
        # 4. Call AI model
        response = await self.gateway.chat(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1000,
        )

        # 5. Parse trend direction from response
        content = response.content if response.success else ""
        trend_direction = "sideways"
        if "看涨" in content and "看跌" not in content:
            trend_direction = "bullish"
        elif "看跌" in content:
            trend_direction = "bearish"

        confidence = "medium"
        if "高置信" in content or "高 置信" in content:
            confidence = "high"
        elif "低置信" in content or "低 置信" in content:
            confidence = "low"

        return {
            "symbol": symbol,
            "name": name,
            "frequency": frequency,
            "trend_direction": trend_direction,
            "confidence": confidence,
            "ai_conclusion": content,
            "raw_indicators": indicators,
            "support_level": None,
            "resistance_level": None,
        }


# -- Serenity Analysis Engine (SA-001 ~ SA-003) --------------------------------
class SerenityEngine:
    """Serenity 5-step bottleneck hunter analysis."""

    def __init__(self, gateway: ModelGateway):
        self.gateway = gateway

    async def analyze(self, symbol: str, name: str = "",
                      sector: str = "") -> Dict[str, Any]:
        """Run the full 5-step Serenity analysis."""

        # Step 1: Supply Chain BOM
        step1 = await self._step1_bom(symbol, name, sector)

        # Step 2: Bottleneck Audit
        step2 = await self._step2_bottleneck(symbol, name, step1)

        # Step 3: Adversarial Test
        step3 = await self._step3_adversarial(symbol, name, step1, step2)

        # Step 4: Float Dynamics
        step4 = await self._step4_float(symbol, name)

        # Step 5: Execution Matrix
        step5 = await self._step5_matrix(symbol, name, step1, step2, step3, step4)

        # Generate final AI conclusion
        conclusion = await self._generate_conclusion(
            symbol, name, step1, step2, step3, step4, step5
        )

        # Check bottleneck conditions
        conditions = self._check_bottleneck_conditions(step2)

        return {
            "symbol": symbol,
            "name": name,
            "step1_bom": step1,
            "step2_bottleneck": step2,
            "step3_adversarial": step3,
            "step4_float": step4,
            "step5_matrix": step5,
            "conditions_met": conditions,
            "ai_conclusion": conclusion,
        }

    async def analyze_stream(self, symbol: str, name: str = "",
                             sector: str = "") -> AsyncGenerator[str, None]:
        """Run the full 5-step Serenity analysis with SSE progress events."""
        # Step 1: Supply Chain BOM
        yield self._sse_event("progress", {"step": 1, "status": "running",
            "label": "BOM剥洋葱", "message": "正在进行供应链分层分析..."})
        step1 = await self._step1_bom(symbol, name, sector)
        yield self._sse_event("progress", {"step": 1, "status": "done",
            "label": "BOM剥洋葱", "message": "供应链BOM分析完成 ✓"})

        # Step 2: Bottleneck Audit
        yield self._sse_event("progress", {"step": 2, "status": "running",
            "label": "卡脖子审计", "message": "正在盘点核心供应商与技术壁垒..."})
        step2 = await self._step2_bottleneck(symbol, name, step1)
        yield self._sse_event("progress", {"step": 2, "status": "done",
            "label": "卡脖子审计", "message": "卡脖子卡点审计完成 ✓"})

        # Step 3: Adversarial Test
        yield self._sse_event("progress", {"step": 3, "status": "running",
            "label": "逆向证伪", "message": "正在从做空视角进行对抗性审查..."})
        step3 = await self._step3_adversarial(symbol, name, step1, step2)
        yield self._sse_event("progress", {"step": 3, "status": "done",
            "label": "逆向证伪", "message": "逆向证伪分析完成 ✓"})

        # Step 4: Float Dynamics
        yield self._sse_event("progress", {"step": 4, "status": "running",
            "label": "筹码测算", "message": "正在分析筹码结构与持仓分布..."})
        step4 = await self._step4_float(symbol, name)
        yield self._sse_event("progress", {"step": 4, "status": "done",
            "label": "筹码测算", "message": "筹码结构分析完成 ✓"})

        # Step 5: Execution Matrix
        yield self._sse_event("progress", {"step": 5, "status": "running",
            "label": "执行矩阵", "message": "正在制定投资执行策略与仓位管理..."})
        step5 = await self._step5_matrix(symbol, name, step1, step2, step3, step4)
        yield self._sse_event("progress", {"step": 5, "status": "done",
            "label": "执行矩阵", "message": "执行矩阵制定完成 ✓"})

        # Generate final AI conclusion
        yield self._sse_event("progress", {"step": "conclusion", "status": "running",
            "label": "AI 结论", "message": "AI 正在综合所有分析生成最终投资结论..."})
        conclusion = await self._generate_conclusion(
            symbol, name, step1, step2, step3, step4, step5
        )

        # Check bottleneck conditions
        conditions = self._check_bottleneck_conditions(step2)

        result = {
            "symbol": symbol, "name": name,
            "step1_bom": step1, "step2_bottleneck": step2,
            "step3_adversarial": step3, "step4_float": step4,
            "step5_matrix": step5,
            "conditions_met": conditions,
            "ai_conclusion": conclusion,
        }

        yield self._sse_event("result", result)

    @staticmethod
    def _sse_event(event: str, data: Any) -> str:
        """Format data as a Server-Sent Events message."""
        return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"

    async def _step1_bom(self, symbol: str, name: str, sector: str) -> str:
        prompt = f"""你是供应链分析专家。请对{symbol} {name}进行供应链BOM剥洋葱分析。

## 要求：
将市场热点穿透至少3层：
- 第1层：终端需求与产品
- 第2层：中游集成与模组
- 第3层：底层材料、零部件、设备

请输出结构化的供应链分层分析。"""
        resp = await self.gateway.chat(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1500,
        )
        return resp.content if resp.success else "Step 1 analysis unavailable"

    async def _step2_bottleneck(self, symbol: str, name: str, step1: str) -> str:
        prompt = f"""你是半导体/硬科技产业专家。基于以下供应链分析，进行卡脖子卡点审计。

## 供应链分析
{step1[:2000]}

## 要求：
- 盘点核心供应商数量（是否<5家）
- 评估技术门槛
- 评估扩产周期（是否>12个月）
- 分析该环节的市场集中度"""
        resp = await self.gateway.chat(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1500,
        )
        return resp.content if resp.success else "Step 2 analysis unavailable"

    async def _step3_adversarial(self, symbol: str, name: str,
                                  step1: str, step2: str) -> str:
        prompt = f"""你是做空机构分析师。请从以下三个维度对{symbol} {name}进行逆向证伪：

## 背景信息
{step2[:1500]}

## 三个毁灭性地雷：
1. **专利高墙陷阱**：底层技术是否踩到海外巨头的垄断专利雷区？
2. **重资产苦力陷阱**：是高毛利IP垄断者还是重资产代工厂？
3. **应用概念偷换陷阱**：当前订单是AI级还是消费电子级？

请逐项分析，诚实面对风险。"""
        resp = await self.gateway.chat(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1200,
        )
        return resp.content if resp.success else "Step 3 analysis unavailable"

    async def _step4_float(self, symbol: str, name: str) -> str:
        prompt = f"""你是筹码结构分析专家。请对{symbol} {name}进行筹码结构分析。

## 要求：
- 国家队/战略底成本区间
- 机构主力成本区
- 量化/游资情绪区
- 关键支撑位和压力位"""
        resp = await self.gateway.chat(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1000,
        )
        return resp.content if resp.success else "Step 4 analysis unavailable"

    async def _step5_matrix(self, symbol: str, name: str,
                             step1: str, step2: str, step3: str, step4: str) -> str:
        prompt = f"""你是投资策略专家。基于以下分析，为{symbol} {name}制定执行矩阵。

## 分析摘要
- 供应链：{step1[:300]}
- 卡脖子：{step2[:300]}
- 风险：{step3[:300]}
- 筹码：{step4[:300]}

## 要求：
- 左侧潜伏点参数（什么情况下可以开始分批买入）
- 右侧加仓催化剂（什么信号出现后可以加仓）
- 仓位管理建议
- 止损策略"""
        resp = await self.gateway.chat(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1200,
        )
        return resp.content if resp.success else "Step 5 analysis unavailable"

    async def _generate_conclusion(self, symbol: str, name: str,
                                    step1, step2, step3, step4, step5) -> str:
        prompt = f"""你是投资决策专家。基于完整的Serenity 5步分析，为{symbol} {name}生成最终投资建议。

## 5步分析摘要
1. 供应链：{step1[:500]}
2. 卡脖子：{step2[:500]}
3. 逆向证伪：{step3[:500]}
4. 筹码结构：{step4[:500]}
5. 执行矩阵：{step5[:500]}

## 请输出：
### 🔍 核心竞争力总结
### ⚠️ 风险清单（🔴高 🟡中 🟢低）
### 📊 投资时间窗口
### 💰 仓位建议
### 📡 下阶段监测哨位"""
        resp = await self.gateway.chat(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1500,
        )
        return resp.content if resp.success else "AI 结论暂不可用"

    @staticmethod
    def _check_bottleneck_conditions(step2: str) -> Dict[str, bool]:
        """Check 3 bottleneck conditions from step 2 analysis."""
        conditions = {
            "extreme_scarcity": False,    # <5 core players
            "high_verification_barrier": False,  # >12 months import
            "tech_architecture_disruption": False,  # new architecture only solution
        }
        text = step2.lower()
        if "少于5家" in text or "<5" in text or "寡头" in text or "垄断" in text:
            conditions["extreme_scarcity"] = True
        if "12个月" in text or ">12" in text or "验证周期" in text:
            conditions["high_verification_barrier"] = True
        if "技术更迭" in text or "新架构" in text or "唯一解" in text:
            conditions["tech_architecture_disruption"] = True

        conditions["passed"] = sum(conditions.values()) >= 2
        return conditions


# -- KOL Opinion Engine (KV-001 ~ KV-005) --------------------------------------
class KOLEngine:
    """Key Opinion Leader opinion aggregation and analysis engine."""

    def __init__(self, gateway: ModelGateway):
        self.gateway = gateway

    @staticmethod
    def compute_heat_score(likes: int, comments: int, shares: int,
                           followers: int, max_followers: int) -> float:
        """Compute KOL opinion heat score."""
        kol_influence = math.log(followers + 1) / math.log(max_followers + 1)
        return (
            0.25 * math.log(likes + 1) +
            0.30 * math.log(comments + 1) +
            0.20 * math.log(shares + 1) +
            0.25 * kol_influence * 100
        )

    async def extract_opinion(self, raw_text: str) -> Dict[str, Any]:
        """AI extract opinion direction and mentioned stocks (KV-003)."""
        prompt = f"""你是一位金融NLP专家。请从以下财经大V的观点中提取关键信息。

## 原文
{raw_text[:2000]}

## 请提取并返回JSON格式：
{{
  "summary": "不超过150字的观点摘要",
  "direction": "bullish/bearish/neutral",
  "mentioned_stocks": [{{"code": "股票代码", "name": "股票名称", "direction": "看多/看空"}}],
  "topic_tags": ["标签1", "标签2"],
  "confidence": 0.0-1.0
}}

只返回JSON，不要其他内容。"""
        resp = await self.gateway.chat(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500,
        )
        if resp.success and resp.content:
            try:
                import json
                # Try to extract JSON from response
                content = resp.content.strip()
                if "```" in content:
                    content = content.split("```")[1]
                    if content.startswith("json"):
                        content = content[4:]
                return json.loads(content)
            except (json.JSONDecodeError, IndexError):
                pass
        return {
            "summary": raw_text[:150],
            "direction": "neutral",
            "mentioned_stocks": [],
            "topic_tags": [],
            "confidence": 0.5,
        }

    async def generate_consensus(self, opinions: List[Dict],
                                  date_str: str) -> Dict[str, Any]:
        """Generate daily consensus/divergence summary (KV-005, AC-008)."""
        # Count stock mentions
        stock_mentions: Dict[str, Dict] = {}
        for op in opinions:
            for stock in op.get("mentioned_stocks", []):
                code = stock.get("code", "")
                if code not in stock_mentions:
                    stock_mentions[code] = {
                        "code": code,
                        "name": stock.get("name", ""),
                        "bullish": 0,
                        "bearish": 0,
                        "total": 0,
                    }
                sm = stock_mentions[code]
                sm["total"] += 1
                if stock.get("direction") == "看多":
                    sm["bullish"] += 1
                else:
                    sm["bearish"] += 1

        # Hot stocks (≥3 mentions)
        hot_stocks = [
            s for s in stock_mentions.values() if s["total"] >= 3
        ]

        # AI consensus summary
        stock_list = "\n".join(
            f"- {s['name']}({s['code']}): 看多{s['bullish']}位, 看空{s['bearish']}位, 共{s['total']}位提及"
            for s in hot_stocks[:10]
        )

        prompt = f"""你是财经媒体分析专家。请基于Top 40财经大V的观点聚合数据，生成每日共识/分歧摘要。

## 热议标的（≥3位大V提及）
{stock_list}

## 请输出：
### 📊 共识方向
### ⚡ 分歧焦点
### 🏆 热议标的排名
### 💡 投资启示"""
        resp = await self.gateway.chat(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=800,
        )

        return {
            "summary_date": date_str,
            "hot_stocks": hot_stocks,
            "ai_summary": resp.content if resp.success else "",
        }
