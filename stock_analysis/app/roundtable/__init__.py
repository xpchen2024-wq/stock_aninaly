# ============================================================================
# 投研圆桌 · TradingAgents 风格多智能体圆桌
# ============================================================================
"""功能包说明（与 GitHub TauricResearch/TradingAgents 的对应关系）：

| TradingAgents            | 本项目实现                                      |
|--------------------------|------------------------------------------------|
| LangGraph StateGraph     | ``graph.RoundtableGraph`` + 条件路由函数         |
| ConditionalLogic         | ``should_continue_debate`` / ``should_continue_risk`` |
| 四分析师 + 各自 tools     | ``agents.*Analyst`` + ``toolkit.RoundtableToolkit`` |
| Bull/Bear N 轮辩论        | ``max_debate_rounds``，逐轮互相反驳              |
| Research Manager 裁决     | ``ResearchManager.judge``（深思考档模型）        |
| Trader 投资计划           | ``Trader.plan``                                 |
| 三方风控 + Portfolio Mgr  | ``RiskAnalyst`` ×3 + ``PortfolioManager.decide`` |
| memory log + reflection   | ``memory.py``（决策落库 → 到期结算 → 复盘反思）   |
| verified data snapshot    | ``toolkit`` 生成的事实锚，禁止模型编造数值        |

不引入 LangChain/LangGraph：项目网关只提供 chat completion，且已有统一的
模型配置、故障转移与 A 股数据链路，复刻编排逻辑比引入依赖更稳。
"""
from app.roundtable.graph import (
    RoundtableGraph, should_continue_debate, should_continue_risk,
)
from app.roundtable.state import RoundtableConfig

__all__ = ["RoundtableGraph", "RoundtableConfig",
           "should_continue_debate", "should_continue_risk"]
