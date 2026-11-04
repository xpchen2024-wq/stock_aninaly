# ============================================================================
# 投研圆桌 · 图编排（TradingAgents 的 LangGraph 结构复刻）
# ============================================================================
"""用纯 Python 实现 TradingAgents 的状态图语义：

    Analysts ──► Bull ⇄ Bear ──► Research Manager ──► Trader
        └─► Aggressive ⇄ Conservative ⇄ Neutral ──► Portfolio Manager ──► END

其中节点之间是「条件边」：多空辩论是否继续由 ``should_continue_debate``
判定（``count >= 2 * max_debate_rounds`` 时交给研究经理），风控辩论由
``should_continue_risk`` 判定（``count >= 3 * max_risk_rounds`` 时交给组合
经理）。这两个函数与 TradingAgents ``ConditionalLogic`` 的判定条件一一对应，
便于单测覆盖。

每完成一个节点都会写一个 checkpoint，状态全程可序列化落库（roundtable_runs.state），
因此即便进程重启也能从数据库还原整场圆桌。
"""
from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime, timezone
from typing import Any, AsyncIterator, Awaitable, Callable, Dict, List, Optional
from uuid import uuid4

from app.model_gateway import ModelGateway
from app.roundtable.agents import (
    BearResearcher, BullResearcher, FundamentalsAnalyst, LLMFormatError,
    LLMUnavailable, MarketAnalyst, NewsAnalyst, PortfolioManager,
    ResearchManager, RiskAnalyst, SentimentAnalyst, Trader,
)
from app.roundtable.state import (
    PHASE_ANALYSTS, PHASE_DECISION, PHASE_MEMORY, PHASE_RESEARCH,
    PHASE_RESEARCH_JUDGE, PHASE_RISK, PHASE_TRADER,
    RoundtableConfig, add_error, append_transcript, new_state,
    summarize_reports,
)
from app.roundtable.toolkit import RoundtableToolkit

logger = logging.getLogger(__name__)

NODE_ANALYSTS = "Analysts"
NODE_BULL = "Bull Researcher"
NODE_BEAR = "Bear Researcher"
NODE_RESEARCH_MANAGER = "Research Manager"
NODE_TRADER = "Trader"
NODE_AGGRESSIVE = "Aggressive Analyst"
NODE_CONSERVATIVE = "Conservative Analyst"
NODE_NEUTRAL = "Neutral Analyst"
NODE_PM = "Portfolio Manager"
NODE_END = "END"

RISK_ROTATION = {
    NODE_AGGRESSIVE: NODE_CONSERVATIVE,
    NODE_CONSERVATIVE: NODE_NEUTRAL,
    NODE_NEUTRAL: NODE_AGGRESSIVE,
}

RISK_ROLE = {
    NODE_AGGRESSIVE: "aggressive_analyst",
    NODE_CONSERVATIVE: "conservative_analyst",
    NODE_NEUTRAL: "neutral_analyst",
}

NODE_STAGE_PCT = {
    NODE_ANALYSTS: 18,
    NODE_BULL: 30,
    NODE_BEAR: 38,
    NODE_RESEARCH_MANAGER: 50,
    NODE_TRADER: 60,
    NODE_AGGRESSIVE: 70,
    NODE_CONSERVATIVE: 76,
    NODE_NEUTRAL: 82,
    NODE_PM: 94,
}


# -- 条件路由（与 TradingAgents ConditionalLogic 对齐） ----------------------
def should_continue_debate(state: Dict[str, Any],
                           max_debate_rounds: int) -> str:
    """多空辩论的下一个节点。"""
    debate = state.get("investment_debate_state") or {}
    if debate.get("count", 0) >= 2 * max_debate_rounds:
        return NODE_RESEARCH_MANAGER
    if str(debate.get("current_response", "")).startswith("Bull"):
        return NODE_BEAR
    return NODE_BULL


def should_continue_risk(state: Dict[str, Any], max_risk_rounds: int) -> str:
    """三方风控讨论的下一个节点。"""
    risk = state.get("risk_debate_state") or {}
    if risk.get("count", 0) >= 3 * max_risk_rounds:
        return NODE_PM
    return RISK_ROTATION.get(risk.get("latest_speaker", ""), NODE_AGGRESSIVE)


class RoundtableGraph:
    """一次投研圆桌的完整图执行器。"""

    def __init__(self, gateway: ModelGateway, failover,
                 config: Optional[RoundtableConfig] = None):
        self.gateway = gateway
        self.failover = failover
        self.config = (config or RoundtableConfig()).clamp()

        self.market_analyst = MarketAnalyst(gateway, self.config)
        self.fundamentals_analyst = FundamentalsAnalyst(gateway, self.config)
        self.news_analyst = NewsAnalyst(gateway, self.config)
        self.sentiment_analyst = SentimentAnalyst(gateway, self.config)
        self.bull = BullResearcher(gateway, self.config)
        self.bear = BearResearcher(gateway, self.config)
        self.research_manager = ResearchManager(gateway, self.config)
        self.trader = Trader(gateway, self.config)
        self.risk_agents = {
            NODE_AGGRESSIVE: RiskAnalyst(gateway, self.config, "aggressive"),
            NODE_CONSERVATIVE: RiskAnalyst(gateway, self.config, "conservative"),
            NODE_NEUTRAL: RiskAnalyst(gateway, self.config, "neutral"),
        }
        self.portfolio_manager = PortfolioManager(gateway, self.config)

    # -- 工具 ---------------------------------------------------------------
    @staticmethod
    def _checkpoint(state: Dict[str, Any], node: str) -> None:
        state["node"] = node
        state.setdefault("checkpoints", []).append({
            "node": node,
            "ts": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
            "debate_count": (state.get("investment_debate_state") or {}).get("count", 0),
            "risk_count": (state.get("risk_debate_state") or {}).get("count", 0),
        })

    async def _safe(self, name: str, runner: Callable[[], Awaitable[str]],
                    fallback: Callable[[], str],
                    state: Dict[str, Any]) -> str:
        """单个角色失败不拖垮整场圆桌：记录降级原因并用确定性内容兜底。"""
        try:
            return await runner()
        except LLMFormatError as exc:
            add_error(state, f"{name} 输出格式不合规：{exc}")
        except LLMUnavailable as exc:
            add_error(state, f"{name} 模型不可用：{exc}")
        except Exception as exc:
            logger.warning(f"[roundtable] {name} failed: {exc}", exc_info=True)
            add_error(state, f"{name} 执行失败：{exc}")
        try:
            return fallback()
        except Exception as exc:  # 兜底也失败时给出可读占位
            add_error(state, f"{name} 兜底失败：{exc}")
            return f"（{name} 未产出内容：{exc}）"

    # -- 节点：分析师 -------------------------------------------------------
    async def _node_analysts(self, state: Dict[str, Any], tk: RoundtableToolkit,
                             kol_text: str) -> List[Dict[str, Any]]:
        # 四位分析师互相独立，各自只拿到自己的数据块（共享同一份事实锚）
        market, funda, news, senti = await asyncio.gather(
            self._safe("市场技术分析师", lambda: self.market_analyst.run(tk),
                       lambda: self.market_analyst.fallback(tk), state),
            self._safe("基本面分析师", lambda: self.fundamentals_analyst.run(tk),
                       lambda: self.fundamentals_analyst.fallback(tk), state),
            self._safe("新闻分析师", lambda: self.news_analyst.run(tk),
                       lambda: self.news_analyst.fallback(tk), state),
            self._safe("情绪分析师", lambda: self.sentiment_analyst.run(tk, kol_text),
                       lambda: self.sentiment_analyst.fallback(tk), state),
        )

        state["market_report"] = market
        state["fundamentals_report"] = funda
        state["news_report"] = news
        state["sentiment_report"] = senti

        return [
            append_transcript(state, "market_analyst", PHASE_ANALYSTS, market, "report"),
            append_transcript(state, "fundamentals_analyst", PHASE_ANALYSTS, funda, "report"),
            append_transcript(state, "news_analyst", PHASE_ANALYSTS, news, "report"),
            append_transcript(state, "sentiment_analyst", PHASE_ANALYSTS, senti, "report"),
        ]

    # -- 节点：多空研究员 ---------------------------------------------------
    def _debate_round(self, state: Dict[str, Any]) -> int:
        count = (state["investment_debate_state"])["count"]
        return count // 2 + 1

    async def _node_bull(self, state: Dict[str, Any], tk: RoundtableToolkit
                         ) -> List[Dict[str, Any]]:
        debate = state["investment_debate_state"]
        round_no = self._debate_round(state)
        reports = summarize_reports(state)
        text = await self._safe(
            "看涨研究员",
            lambda: self.bull.argue(
                tk, reports, debate.get("current_response", ""),
                "\n".join(debate.get("bull_history") or []), round_no,
            ),
            lambda: self.bull.fallback(tk), state,
        )
        debate["bull_history"].append(text)
        debate["history"].append({"agent": "看涨研究员", "content": text, "round": round_no})
        debate["current_response"] = f"Bull Researcher: {text}"
        debate["count"] = debate.get("count", 0) + 1
        return [append_transcript(state, "bull_researcher", PHASE_RESEARCH, text,
                                  "argument" if round_no <= 1 else "rebuttal", round_no)]

    async def _node_bear(self, state: Dict[str, Any], tk: RoundtableToolkit
                         ) -> List[Dict[str, Any]]:
        debate = state["investment_debate_state"]
        round_no = self._debate_round(state)
        reports = summarize_reports(state)
        text = await self._safe(
            "看跌研究员",
            lambda: self.bear.argue(
                tk, reports, debate.get("current_response", ""),
                "\n".join(debate.get("bear_history") or []), round_no,
            ),
            lambda: self.bear.fallback(tk), state,
        )
        debate["bear_history"].append(text)
        debate["history"].append({"agent": "看跌研究员", "content": text, "round": round_no})
        debate["current_response"] = f"Bear Researcher: {text}"
        debate["count"] = debate.get("count", 0) + 1
        return [append_transcript(state, "bear_researcher", PHASE_RESEARCH, text,
                                  "argument" if round_no <= 1 else "rebuttal", round_no)]

    # -- 节点：研究经理 ----------------------------------------------------
    async def _node_research_manager(self, state: Dict[str, Any],
                                     tk: RoundtableToolkit) -> List[Dict[str, Any]]:
        debate = state["investment_debate_state"]
        history_text = "\n\n".join(
            f"[{h['agent']}] {h['content']}" for h in debate.get("history") or []
        )
        reports = summarize_reports(state)

        def _fallback() -> str:
            bull_n = len(debate.get("bull_history") or [])
            bear_n = len(debate.get("bear_history") or [])
            tech = (tk.forecast or {}).get("technical") or {}
            direction = ("看多" if (tech.get("score") or 0) > 10 else
                         "看空" if (tech.get("score") or 0) < -10 else "中性")
            return (
                "## 裁决\n（模型不可用，回退为规则裁决）\n"
                f"- 量化技术评分 {tech.get('score', '—')} → {direction}\n"
                f"- 多空发言次数 {bull_n} : {bear_n}\n"
                f"- 信心：低（仅规则合成）\n"
                "## 给交易员的指示\n按量化方向执行，仓位不超过 20%，严格执行止损。"
            )

        text = await self._safe(
            "研究经理",
            lambda: self.research_manager.judge(tk, history_text, reports),
            _fallback, state,
        )
        debate["judge_decision"] = text
        return [append_transcript(state, "research_manager",
                                  PHASE_RESEARCH_JUDGE, text, "judge")]

    # -- 节点：交易员 ------------------------------------------------------
    async def _node_trader(self, state: Dict[str, Any], tk: RoundtableToolkit
                           ) -> List[Dict[str, Any]]:
        judgement = (state["investment_debate_state"].get("judge_decision") or "")
        reports = summarize_reports(state)

        def _fallback() -> str:
            levels = (tk.forecast or {}).get("levels") or {}

            def _lv(items) -> str:
                if not items:
                    return "—"
                first = items[0]
                if isinstance(first, dict):
                    return f"{first.get('label', '位')} {first.get('price')}"
                return str(first)

            price = (tk.snapshot_core() or {}).get("price")
            return (
                "## 交易方向\n（模型不可用，按规则生成）\n"
                f"- 现价 {price}，支撑 {_lv(levels.get('supports'))}，"
                f"阻力 {_lv(levels.get('resistances'))}\n"
                "## 执行计划\n- 若放量突破阻力再介入，否则观望\n"
                "## 仓位与节奏\n- 首笔不超过 20%，分两笔\n"
                "## 失效条件\n- 跌破首个支撑位即离场"
            )

        text = await self._safe("交易员", lambda: self.trader.plan(tk, reports, judgement),
                                _fallback, state)
        state["trader_investment_plan"] = text
        return [append_transcript(state, "trader", PHASE_TRADER, text, "plan")]

    # -- 节点：三方风控 ----------------------------------------------------
    def _risk_round(self, state: Dict[str, Any]) -> int:
        return (state["risk_debate_state"])["count"] // 3 + 1

    async def _node_risk(self, node: str, state: Dict[str, Any],
                         tk: RoundtableToolkit) -> List[Dict[str, Any]]:
        risk = state["risk_debate_state"]
        agent = self.risk_agents[node]
        round_no = self._risk_round(state)
        label = {
            NODE_AGGRESSIVE: "激进风控", NODE_CONSERVATIVE: "保守风控",
            NODE_NEUTRAL: "中性风控",
        }[node]
        history = "\n\n".join(
            f"[{h['agent']}] {h['content']}" for h in risk.get("history") or []
        )
        reports = summarize_reports(state)

        def _fallback() -> str:
            snap = tk.snapshot_core() or {}
            return (
                f"（{label}模型不可用）参考：年化波动率 {snap.get('volatility')}%，"
                f"ATR占价比 {snap.get('atr_pct')}%，建议仓位控制在 20% 以内并设置硬止损。"
            )

        text = await self._safe(
            label,
            lambda: agent.speak(tk, state.get("trader_investment_plan", ""),
                                reports, history, round_no),
            _fallback, state,
        )
        key = {
            NODE_AGGRESSIVE: "aggressive_history",
            NODE_CONSERVATIVE: "conservative_history",
            NODE_NEUTRAL: "neutral_history",
        }[node]
        risk[key].append(text)
        risk["history"].append({"agent": label, "content": text, "round": round_no})
        risk["latest_speaker"] = node
        risk["count"] = risk.get("count", 0) + 1
        return [append_transcript(state, RISK_ROLE[node], PHASE_RISK, text,
                                  "argument" if round_no <= 1 else "rebuttal", round_no)]

    # -- 节点：组合经理 ----------------------------------------------------
    async def _node_pm(self, state: Dict[str, Any], tk: RoundtableToolkit
                       ) -> List[Dict[str, Any]]:
        risk = state["risk_debate_state"]
        risk_text = "\n\n".join(
            f"[{h['agent']}] {h['content']}" for h in risk.get("history") or []
        )
        judgement = state["investment_debate_state"].get("judge_decision") or ""
        plan = state.get("trader_investment_plan") or ""
        reports = summarize_reports(state)

        try:
            decision = await self.portfolio_manager.decide(
                tk, reports, judgement, plan, risk_text,
                state.get("past_memory_str", ""),
            )
        except LLMFormatError as exc:
            # 模型有响应但没给出合法 JSON（推理模型思维链占满预算被截断）：
            # 这是格式问题，不是服务不可用，提示必须分开，否则会误导排查方向。
            logger.warning(f"[roundtable] portfolio manager format error: {exc}")
            add_error(state, f"组合经理输出格式不合规（已按量化规则降级）：{exc}")
            decision = self.portfolio_manager.fallback_decision(tk, str(exc))
        except LLMUnavailable as exc:
            logger.warning(f"[roundtable] portfolio manager model unavailable: {exc}")
            add_error(state, f"组合经理模型不可用（已按量化规则降级）：{exc}")
            decision = self.portfolio_manager.fallback_decision(tk, str(exc))
        except Exception as exc:
            logger.warning(f"[roundtable] portfolio manager failed: {exc}")
            add_error(state, f"组合经理决策失败：{exc}")
            decision = self.portfolio_manager.fallback_decision(tk, str(exc))

        state["final_trade_decision"] = decision
        risk["judge_decision"] = decision.get("rationale", "")

        summary = self._render_decision(decision)
        return [append_transcript(state, "portfolio_manager", PHASE_DECISION,
                                  summary, "decision")]

    @staticmethod
    def _render_decision(decision: Dict[str, Any]) -> str:
        """把终审结果渲染成发言。

        价格必须带方向词：SELL 的 target 是「下跌目标」（低于现价才对），
        写成中性的「目标」会让人以为系统把止盈止损算反了。
        """
        action = decision.get("action")
        entry = decision.get("entry_price") or "—"
        target = decision.get("target_price")
        stop = decision.get("stop_loss")

        if action == "buy":
            price_line = (f"入场 {entry}｜上涨目标 {target or '—'}"
                          f"｜止损 {stop or '—'}")
        elif action == "sell":
            price_line = (f"参考价 {entry}｜下跌目标 {target or '—'}"
                          f"｜上方止损 {stop or '—'}")
        else:
            price_line = f"参考价 {entry}｜无目标价（观望）"
        notes = [n for n in (decision.get("target_note"), decision.get("stop_note")) if n]
        if notes:
            price_line += f"（依据：{' / '.join(notes)}）"

        head = (
            f"**最终决策：{str(action or '').upper()}**"
            f"（信心 {decision.get('confidence')}，"
            f"建议仓位 {decision.get('position_pct')}%）\n\n"
            f"- {price_line}\n"
        )
        if decision.get("risk_reward") is not None:
            head += f"- 盈亏比 {decision['risk_reward']}:1\n"
        head += f"- 理由：{decision.get('rationale') or '—'}\n"
        head += (f"- 关键风险：{'；'.join(decision.get('key_risks') or [])}\n"
                 if decision.get("key_risks") else "")
        head += (f"- 参考的历史教训：{'；'.join(decision.get('memory_used') or [])}\n"
                 if decision.get("memory_used") else "")
        head += (f"- 本次要记住的教训：{decision['lessons']}"
                 if decision.get("lessons") else "")
        return head

    # -- 主流程 ------------------------------------------------------------
    async def run_stream(
        self,
        symbol: str,
        name: str = "",
        trade_date: Optional[str] = None,
        kol_context: str = "",
        memory_text: str = "",
        settle: Optional[Callable[[], Awaitable[Dict[str, Any]]]] = None,
        refresh_memory: Optional[Callable[[], Awaitable[str]]] = None,
        session_id: Optional[str] = None,
    ) -> AsyncIterator[tuple]:
        """执行圆桌，逐事件产出 ``("phase"|"agent"|"result", payload)``。"""
        session_id = session_id or str(uuid4())
        started = time.time()
        trade_date = trade_date or datetime.now().strftime("%Y-%m-%d")
        state = new_state(symbol, name, trade_date, self.config)
        state["past_memory_str"] = memory_text or ""
        state["session_id"] = session_id
        state["kol_context"] = kol_context

        tk = RoundtableToolkit(self.failover, symbol, name,
                               trade_date=trade_date,
                               horizon_days=self.config.horizon_days)

        yield "phase", {
            "node": NODE_ANALYSTS, "stage": "数据锚定", "status": "running",
            "pct": 6,
            "message": f"拉取 {symbol} 行情/基本面/新闻/资金流并做量化研判…",
        }
        try:
            state["snapshot"] = await tk.prepare()
            for error in tk.errors:
                add_error(state, error)
        except Exception as exc:
            add_error(state, f"数据快照构建失败：{exc}")
            state["snapshot"] = {
                "bars": 0, "price": None, "trade_date": trade_date,
                "note": "数据源全部不可用，本次结论仅基于模型推理",
            }
        yield "phase", {
            "node": NODE_ANALYSTS, "stage": "数据锚定", "status": "done", "pct": 12,
            "message": (
                f"数据就绪：{state['snapshot'].get('bars', 0)} 根日线，"
                f"新闻 {state['snapshot'].get('news_count', 0)} 条"
                if state["snapshot"].get("bars") else "数据源异常，已进入降级模式"
            ),
        }

        # 记忆结算与分析师并行（对齐 TradingAgents：分析师工作时结算历史）
        settle_task: Optional[asyncio.Task] = None
        if settle is not None:
            settle_task = asyncio.create_task(self._run_settle(settle))

        node: str = NODE_ANALYSTS
        guard = 0
        while node != NODE_END and guard < 40:
            guard += 1
            self._checkpoint(state, node)
            yield "phase", {
                "node": node,
                "stage": self._stage_label(node),
                "status": "running",
                "pct": NODE_STAGE_PCT.get(node, 50),
                "message": self._stage_message(node),
            }

            try:
                if node == NODE_ANALYSTS:
                    entries = await self._node_analysts(state, tk, kol_context)
                    next_node = NODE_BULL
                elif node == NODE_BULL:
                    entries = await self._node_bull(state, tk)
                    next_node = should_continue_debate(state, self.config.max_debate_rounds)
                elif node == NODE_BEAR:
                    entries = await self._node_bear(state, tk)
                    next_node = should_continue_debate(state, self.config.max_debate_rounds)
                elif node == NODE_RESEARCH_MANAGER:
                    entries = await self._node_research_manager(state, tk)
                    next_node = NODE_TRADER
                elif node == NODE_TRADER:
                    entries = await self._node_trader(state, tk)
                    next_node = NODE_AGGRESSIVE
                elif node in RISK_ROTATION:
                    entries = await self._node_risk(node, state, tk)
                    next_node = should_continue_risk(state, self.config.max_risk_rounds)
                elif node == NODE_PM:
                    # 组合经理发言前确保记忆已结算完毕
                    if settle_task is not None:
                        yield "phase", {
                            "node": NODE_PM, "stage": "记忆结算", "status": "running",
                            "pct": 88, "message": "结算历史决策并生成复盘教训…",
                        }
                        settle_stats = await self._await_settle(settle_task, state)
                        state["memory_stats"] = settle_stats
                        if refresh_memory is not None:
                            try:
                                state["past_memory_str"] = await refresh_memory() or state["past_memory_str"]
                            except Exception as exc:
                                logger.warning(f"[roundtable] refresh memory failed: {exc}")
                    entries = await self._node_pm(state, tk)
                    next_node = NODE_END
                else:
                    break
            except Exception as exc:
                logger.error(f"[roundtable] node {node} crashed: {exc}", exc_info=True)
                add_error(state, f"{node} 节点异常：{exc}")
                entries, next_node = [], NODE_END

            for payload in entries:
                yield "agent", payload

            yield "phase", {
                "node": node,
                "stage": self._stage_label(node),
                "status": "done",
                "pct": NODE_STAGE_PCT.get(node, 50),
                "message": self._done_message(node, state),
            }
            node = next_node

        if settle_task is not None and not settle_task.done():
            settle_task.cancel()

        elapsed = int((time.time() - started) * 1000)
        result = self._build_result(state, tk, elapsed)
        yield "result", result

    async def _run_settle(self, settle: Callable[[], Awaitable[Dict[str, Any]]]
                          ) -> Dict[str, Any]:
        try:
            return await settle()
        except Exception as exc:
            logger.warning(f"[roundtable] settle task failed: {exc}")
            return {"checked": 0, "settled": 0, "skipped": 0, "failed": 1, "error": str(exc)}

    async def _await_settle(self, task: asyncio.Task,
                            state: Dict[str, Any]) -> Dict[str, Any]:
        try:
            return await asyncio.wait_for(asyncio.shield(task), timeout=45.0)
        except asyncio.TimeoutError:
            add_error(state, "记忆结算超时，本次使用已有记忆")
            return {"checked": 0, "settled": 0, "skipped": 0, "failed": 0, "timeout": True}
        except Exception as exc:
            add_error(state, f"记忆结算失败：{exc}")
            return {"checked": 0, "settled": 0, "skipped": 0, "failed": 1}

    @staticmethod
    def _stage_label(node: str) -> str:
        return {
            NODE_ANALYSTS: "分析师团队", NODE_BULL: "多空辩论",
            NODE_BEAR: "多空辩论", NODE_RESEARCH_MANAGER: "研究经理裁决",
            NODE_TRADER: "交易计划", NODE_AGGRESSIVE: "三方风控辩论",
            NODE_CONSERVATIVE: "三方风控辩论", NODE_NEUTRAL: "三方风控辩论",
            NODE_PM: "组合经理终审",
        }.get(node, node)

    @staticmethod
    def _stage_message(node: str) -> str:
        return {
            NODE_ANALYSTS: "四位分析师并行工作中（各自独立数据集）…",
            NODE_BULL: "看涨研究员发言…",
            NODE_BEAR: "看跌研究员发言…",
            NODE_RESEARCH_MANAGER: "研究经理综合多空论点裁决中…",
            NODE_TRADER: "交易员制定执行计划…",
            NODE_AGGRESSIVE: "激进风控评审计划…",
            NODE_CONSERVATIVE: "保守风控评审计划…",
            NODE_NEUTRAL: "中性风控补充意见…",
            NODE_PM: "组合经理终审…",
        }.get(node, node)

    def _done_message(self, node: str, state: Dict[str, Any]) -> str:
        if node in (NODE_BULL, NODE_BEAR):
            cnt = state["investment_debate_state"]["count"]
            return f"已完成 {cnt}/{2 * self.config.max_debate_rounds} 次发言"
        if node in RISK_ROTATION:
            cnt = state["risk_debate_state"]["count"]
            return f"已完成 {cnt}/{3 * self.config.max_risk_rounds} 次发言"
        if node == NODE_PM:
            d = state.get("final_trade_decision") or {}
            return f"决策：{d.get('action', 'hold')}（信心 {d.get('confidence', '—')}）"
        return "完成"

    def _build_result(self, state: Dict[str, Any], tk: RoundtableToolkit,
                      elapsed_ms: int) -> Dict[str, Any]:
        debate = state["investment_debate_state"]
        risk = state["risk_debate_state"]
        decision = state.get("final_trade_decision") or {}
        degraded = bool(decision.get("degraded")) or bool(state.get("errors"))
        return {
            "session_id": state.get("session_id"),
            "symbol": state["company_of_interest"],
            "name": state["company_name"],
            "trade_date": state["trade_date"],
            "status": "completed_with_errors" if state.get("errors") else "completed",
            "degraded": degraded,
            "config": state["config"],
            "snapshot": state["snapshot"],
            "analyst_reports": {
                "market": state["market_report"],
                "fundamentals": state["fundamentals_report"],
                "news": state["news_report"],
                "sentiment": state["sentiment_report"],
            },
            "research_debate": {
                "rounds": self.config.max_debate_rounds,
                "count": debate.get("count", 0),
                "history": debate.get("history") or [],
                "judgement": debate.get("judge_decision") or "",
            },
            "trader_plan": state.get("trader_investment_plan") or "",
            "risk_debate": {
                "rounds": self.config.max_risk_rounds,
                "count": risk.get("count", 0),
                "history": risk.get("history") or [],
                "judgement": risk.get("judge_decision") or "",
            },
            "decision": decision,
            "transcript": state.get("transcript") or [],
            "checkpoints": state.get("checkpoints") or [],
            "errors": state.get("errors") or [],
            "memory_stats": state.get("memory_stats") or {},
            "elapsed_ms": elapsed_ms,
            "grounding": tk.grounding_text()[:20000],
        }
