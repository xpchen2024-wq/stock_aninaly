# ============================================================================
# 投研圆桌 · 角色智能体
# ============================================================================
"""对齐 TradingAgents 的角色分工：

    分析师团队（4）→ 多空研究员（N 轮辩论）→ 研究经理裁决
    → 交易员（投资计划）→ 三方风控（N 轮辩论）→ 组合经理终审 → 反思

与 TradingAgents 的差异：LangChain 的 tool-calling 循环在这里换成
「事实锚 + 单轮强约束提示」——因为项目网关只暴露 chat completion。
所有数值必须来自 RoundtableToolkit 生成的数据块，模型不允许自行估算，
这样既保留 TradingAgents 的「分析有据可查」，又避免多轮工具调用的延迟。
"""
from __future__ import annotations

import inspect
import logging
from typing import Any, Dict, List, Optional

from app.model_gateway import LLMResponse, ModelGateway
from app.roundtable.state import RoundtableConfig
from app.roundtable.toolkit import RoundtableToolkit
from app.utils.json_extract import build_retry_prompt, parse_llm_json

logger = logging.getLogger(__name__)

_NO_FABRICATE = (
    "硬性规则：\n"
    "1) 只能引用「数据块」中出现的数字，缺失就写“数据缺失”，严禁编造价格、指标、财务或新闻；\n"
    "2) 结论必须带明确方向（看多/看空/中性）与置信度（高/中/低），并给出可被数据证伪的理由；\n"
    "3) 用中文输出，结构化 Markdown，不超过 500 字，不要客套话与免责声明。\n"
)

# 结构化输出角色专用：推理型模型（Nemotron / o1 类）默认会把思维链写进 content，
# 一旦 token 预算被思考过程吃光就拿不到最终 JSON，因此必须显式禁止。
_JSON_ONLY = (
    "输出规范（强制遵守）：\n"
    "- 只输出一个 JSON 对象本身，第一个字符必须是 {，最后一个字符必须是 }；\n"
    "- 禁止输出思考过程、分析步骤、前言、解释，禁止 markdown 代码块与注释；\n"
    "- 数值字段必须是数字字面量（不要写 \"8.45元\" 这类带单位的字符串）。\n"
)


class LLMUnavailable(RuntimeError):
    """模型调用失败（全部候选模型不可用、超时、鉴权失败等）。"""


class LLMFormatError(RuntimeError):
    """模型返回了内容但不是合法 JSON（多为思维链占满 token 被截断）。

    与 ``LLMUnavailable`` 严格区分：这不是服务不可用，模型是有响应的，
    只是输出不合规。混为一谈会让前端把「格式问题」显示成「模型挂了」，
    误导使用者去排查不存在的服务故障。
    """


class RoundtableAgent:
    role = "agent"

    # 结构化输出的最低 token 预算：PM 的 JSON 正文只有 ~200 token，
    # 剩余预算留给推理型模型的思维链，避免在写出 JSON 前就被截断。
    JSON_MIN_TOKENS = 1600
    JSON_MAX_TOKENS = 4096

    def __init__(self, gateway: ModelGateway, config: RoundtableConfig):
        self.gateway = gateway
        self.config = config
        self._accepts: Dict[str, bool] = {}

    # -- 底层调用 -----------------------------------------------------------
    def _supports(self, kwarg: str) -> bool:
        """网关是否接受某个可选参数（测试替身/自定义网关可能没有）。"""
        if kwarg not in self._accepts:
            try:
                params = inspect.signature(self.gateway.chat).parameters
                self._accepts[kwarg] = kwarg in params
            except (TypeError, ValueError):
                self._accepts[kwarg] = False
        return self._accepts[kwarg]

    async def _call(self, system: str, user: str, deep: bool = False,
                    max_tokens: int = 900, json_mode: bool = False,
                    exclude_config_ids: Optional[List[str]] = None) -> LLMResponse:
        prefer = self.config.deep_model_id if deep else self.config.quick_model_id
        kwargs: Dict[str, Any] = {
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "prefer_config_id": prefer,
            "temperature": 0.3 if deep else 0.5,
            "max_tokens": max_tokens,
        }
        if json_mode and self._supports("response_format"):
            kwargs["response_format"] = {"type": "json_object"}
        if exclude_config_ids and self._supports("exclude_config_ids"):
            kwargs["exclude_config_ids"] = exclude_config_ids
        return await self.gateway.chat(**kwargs)

    @staticmethod
    def _text(resp: LLMResponse) -> str:
        if not resp.success or not (resp.content or "").strip():
            raise LLMUnavailable(resp.error or "模型返回空内容")
        return resp.content.strip()

    # -- 文本型 -------------------------------------------------------------
    async def chat(self, system: str, user: str, deep: bool = False,
                   max_tokens: int = 900) -> str:
        resp = await self._call(system, user, deep=deep, max_tokens=max_tokens)
        return self._text(resp)

    # -- JSON 型 ------------------------------------------------------------
    async def chat_json(self, system: str, user: str, deep: bool = True,
                        max_tokens: int = 900) -> Dict[str, Any]:
        """强制拿到 JSON 对象；拿不到就抛 ``LLMFormatError``（绝不返回 None）。

        重试策略针对真实故障设计：
        1. 首次用 ``response_format=json_object`` + 不低于 ``JSON_MIN_TOKENS`` 的预算；
        2. 失败后把预算翻倍、并避开刚失败的那个模型重试一次
           （推理型模型截断是最常见根因，原样重试必然复现同样的失败）；
        3. 仍然失败则抛出带「模型名 + finish_reason + 原始输出片段」的异常。
        """
        system = f"{system}\n{_JSON_ONLY}"
        budget = min(self.JSON_MAX_TOKENS, max(max_tokens, self.JSON_MIN_TOKENS))

        first = await self._call(system, user, deep=deep,
                                 max_tokens=budget, json_mode=True)
        text = self._text(first)
        data = parse_llm_json(text)
        if isinstance(data, dict):
            return data

        retry_budget = min(self.JSON_MAX_TOKENS, max(budget * 2, 2048))
        # 换一个模型重试：同一个推理模型在同样预算下只会再吐一遍思维链，
        # 原样重试等于白等 30 秒（真实案例：Nemotron 连续两次 finish_reason=length）。
        avoid = [first.config_id] if first.config_id else None
        logger.warning(
            f"[roundtable] {self.role} 首次输出不是合法 JSON "
            f"(model={first.model}, finish_reason={first.finish_reason or 'unknown'})，"
            f"提升预算 {budget}→{retry_budget} 并避开该模型重试"
        )
        second = await self._call(
            system, build_retry_prompt(user, text),
            deep=deep, max_tokens=retry_budget, json_mode=True,
            exclude_config_ids=avoid,
        )
        text2 = self._text(second)
        data = parse_llm_json(text2)
        if isinstance(data, dict):
            return data

        truncated = second.finish_reason == "length"
        raise LLMFormatError(
            f"模型 {second.model} 未返回合法 JSON；"
            f"finish_reason={second.finish_reason or 'unknown'}"
            + (f"，输出在 {retry_budget} token 处被截断，疑似思维链占满预算"
               if truncated else "")
            + f"；原始输出前 120 字：{text2[:120]!r}"
        )


# -- 分析师团队 ---------------------------------------------------------------
class MarketAnalyst(RoundtableAgent):
    role = "market_analyst"

    async def run(self, tk: RoundtableToolkit) -> str:
        system = (
            "你是买方机构的 A 股市场技术分析师，擅长趋势结构、关键位与量价关系。"
            "你的判断将作为圆桌会议的第一手材料。\n" + _NO_FABRICATE
        )
        user = (
            f"标的：{tk.name}（{tk.symbol}），分析基准日 {tk.trade_date}。\n\n"
            f"{tk.market_brief()}\n{tk.technical_brief()}\n"
            "请输出：\n"
            "## 趋势结构\n（均线排列、多周期共振、趋势阶段）\n"
            "## 量能与情绪\n（量比、OBV、资金配合度）\n"
            "## 关键位与风险\n（支撑/阻力、ATR 波动区间）\n"
            "## 结论\n（方向 + 置信度 + 一句话操作含义）"
        )
        return await self.chat(system, user)

    def fallback(self, tk: RoundtableToolkit) -> str:
        from app.roundtable.toolkit import _fmt_levels
        fc = tk.forecast or {}
        tech = fc.get("technical") or {}
        levels = fc.get("levels") or {}
        prob = (fc.get("probability") or {}).get("probability")
        return (
            "## 趋势结构\n（模型不可用，以下为系统量化结论）\n"
            f"- 方向：{tech.get('direction', '未知')}，技术评分 {tech.get('score', '—')}\n"
            f"- 多周期：{(fc.get('multi_timeframe') or {}).get('resonance', '—')}\n"
            "## 关键位与风险\n"
            f"- 支撑 {_fmt_levels(levels.get('supports'))}，"
            f"阻力 {_fmt_levels(levels.get('resistances'))}\n"
            f"- 未来 {fc.get('horizon', '—')} 日上涨概率 {prob if prob is not None else '—'}%\n"
            "## 结论\n量化模型输出，未经语言模型加工。"
        )


class FundamentalsAnalyst(RoundtableAgent):
    role = "fundamentals_analyst"

    async def run(self, tk: RoundtableToolkit) -> str:
        system = (
            "你是基本面分析师，擅长估值与盈利质量判断，习惯用可验证的财务事实说话。\n"
            + _NO_FABRICATE
        )
        user = (
            f"标的：{tk.name}（{tk.symbol}），基准日 {tk.trade_date}。\n\n"
            f"{tk.fundamentals_brief()}\n{tk.market_brief()}\n"
            "请输出：\n## 估值水平\n（PE/PB 与市值结构，缺少对比数据时明确说明）\n"
            "## 盈利质量\n（ROE 等可得指标）\n"
            "## 基本面结论\n（方向 + 置信度 + 主要不确定性）"
        )
        return await self.chat(system, user)

    def fallback(self, tk: RoundtableToolkit) -> str:
        f = tk.fundamental or {}
        if not any(f.values()):
            return "## 基本面结论\n数据源未返回财务数据，基本面维度按“未知”处理，不参与打分。"
        return (
            "## 估值水平\n（模型不可用，仅列原始数据）\n"
            f"- PE {f.get('pe_ratio')}，PB {f.get('pb_ratio')}，"
            f"总市值 {f.get('market_cap')} 亿元\n"
            f"## 盈利质量\n- ROE {f.get('roe')}%\n"
            "## 基本面结论\n中性（数据未经语言模型解读）。"
        )


class NewsAnalyst(RoundtableAgent):
    role = "news_analyst"

    async def run(self, tk: RoundtableToolkit) -> str:
        system = (
            "你是新闻与事件驱动分析师，只根据给定新闻列表判断催化剂与风险，"
            "绝不虚构未列出的消息。\n" + _NO_FABRICATE
        )
        user = (
            f"标的：{tk.name}（{tk.symbol}），基准日 {tk.trade_date}。\n\n"
            f"{tk.news_brief()}\n{tk.market_brief()}\n"
            "请输出：\n## 关键事件\n（按影响力排序，标注是利好/利空）\n"
            "## 事件与股价的匹配度\n（消息是否已被 price in）\n"
            "## 新闻面结论\n（方向 + 置信度）"
        )
        return await self.chat(system, user)

    def fallback(self, tk: RoundtableToolkit) -> str:
        if not tk.news:
            return "## 新闻面结论\n近 72 小时无可用个股新闻，新闻面按中性处理。"
        lines = "\n".join(f"- {n['title']}" for n in tk.news[:6])
        return f"## 关键事件\n（模型不可用，仅列标题）\n{lines}\n\n## 新闻面结论\n中性。"


class SentimentAnalyst(RoundtableAgent):
    role = "sentiment_analyst"

    async def run(self, tk: RoundtableToolkit, kol_context: str = "") -> str:
        system = (
            "你是市场情绪分析师，用资金流、量价与社媒/大V观点综合判断短期情绪温度。\n"
            + _NO_FABRICATE
        )
        user = (
            f"标的：{tk.name}（{tk.symbol}），基准日 {tk.trade_date}。\n\n"
            f"{tk.capital_brief()}\n{tk.news_brief()}\n{tk.market_brief()}\n"
            f"{kol_context or '（无大V观点数据）'}\n"
            "请输出：\n## 情绪温度\n（0-100 打分 + 依据）\n"
            "## 资金与舆情的一致性\n## 情绪面结论\n（方向 + 置信度）"
        )
        return await self.chat(system, user)

    def fallback(self, tk: RoundtableToolkit) -> str:
        c = tk.capital or {}
        if not c.get("rows"):
            return "## 情绪面结论\n资金流数据缺失，情绪面按中性处理。"
        return (
            "## 情绪温度\n（模型不可用，以资金流替代）\n"
            f"- 5日主力净流入合计 {(c.get('main_net_total_5d') or 0) / 1e8:+.2f} 亿元，"
            f"净流入 {c.get('inflow_days')}/5 天\n"
            "## 情绪面结论\n中性。"
        )


# -- 多空研究员 ---------------------------------------------------------------
class BullResearcher(RoundtableAgent):
    role = "bull_researcher"

    async def argue(self, tk: RoundtableToolkit, reports: str,
                    opponent_last: str, own_history: str,
                    round_no: int) -> str:
        system = (
            "你是看涨研究员（多头）。你的任务不是盲目乐观，而是找出被市场低估的赔率："
            "用分析师事实构建最强的看多论证，并逐条反驳空头的质疑。\n" + _NO_FABRICATE
        )
        if round_no <= 1:
            user = (
                f"标的：{tk.name}（{tk.symbol}），{tk.trade_date}。\n\n{reports}\n\n"
                "请给出看多论证：\n"
                "1) 三个最强的看多理由（每条必须引用数据块中的具体数字）；\n"
                "2) 你认为空头最可能攻击什么，你如何提前防守；\n"
                "3) 一句话立场（看多强度：强/中/弱）。"
            )
        else:
            user = (
                f"标的：{tk.name}（{tk.symbol}），{tk.trade_date}。这是第 {round_no} 轮辩论。\n\n"
                f"{reports}\n\n【你的既往发言】\n{own_history or '（无）'}\n\n"
                f"【空头刚才说】\n{opponent_last or '（无）'}\n\n"
                "请：\n"
                "1) 逐条反驳空头的核心质疑（不要重复你上一轮已说过的内容）；\n"
                "2) 补充一个新的看多论据；\n"
                "3) 若某个质疑确实成立，明确承认并修正你的立场强度。"
            )
        return await self.chat(system, user, max_tokens=700)

    def fallback(self, tk: RoundtableToolkit) -> str:
        tech = (tk.forecast or {}).get("technical") or {}
        return (
            f"（模型不可用）系统量化信号：技术评分 {tech.get('score', '—')}，"
            f"方向 {tech.get('direction', '未知')}。看多论证暂缺。"
        )


class BearResearcher(RoundtableAgent):
    role = "bear_researcher"

    async def argue(self, tk: RoundtableToolkit, reports: str,
                    opponent_last: str, own_history: str,
                    round_no: int) -> str:
        system = (
            "你是看跌研究员（空头）。你的职责是压力测试：找出逻辑漏洞、估值透支、"
            "拥挤交易与下行风险，并逐条质疑多头的乐观假设。\n" + _NO_FABRICATE
        )
        if round_no <= 1:
            user = (
                f"标的：{tk.name}（{tk.symbol}），{tk.trade_date}。\n\n{reports}\n\n"
                "请给出看空/风险论证：\n"
                "1) 三个最有力的风险或看空理由（引用具体数字）；\n"
                "2) 多头最可能忽略的尾部风险；\n"
                "3) 一句话立场（看空强度：强/中/弱）。"
            )
        else:
            user = (
                f"标的：{tk.name}（{tk.symbol}），{tk.trade_date}。这是第 {round_no} 轮辩论。\n\n"
                f"{reports}\n\n【你的既往发言】\n{own_history or '（无）'}\n\n"
                f"【多头刚才说】\n{opponent_last or '（无）'}\n\n"
                "请：\n"
                "1) 逐条质疑多头的核心论据（指出其数据解读或假设的问题）；\n"
                "2) 补充一个新的风险点；\n"
                "3) 若多头说服了你，明确让步并修正立场强度。"
            )
        return await self.chat(system, user, max_tokens=700)

    def fallback(self, tk: RoundtableToolkit) -> str:
        pop = (tk.snapshot_core() or {}).get("pos_250")
        return (
            f"（模型不可用）系统量化提示：250日价格分位 {pop if pop is not None else '—'}%。"
            "风险论证暂缺。"
        )


# -- 研究经理 -----------------------------------------------------------------
class ResearchManager(RoundtableAgent):
    role = "research_manager"

    async def judge(self, tk: RoundtableToolkit, debate_text: str,
                    reports: str) -> str:
        system = (
            "你是研究经理，主持多空辩论并作出裁决。你不站队，只按证据强度裁决，"
            "明确指出哪一方的哪条论据最有说服力、哪一方出现了逻辑跳跃。\n" + _NO_FABRICATE
        )
        user = (
            f"标的：{tk.name}（{tk.symbol}），{tk.trade_date}。\n\n"
            f"【多空辩论全文】\n{debate_text}\n\n"
            f"【分析师报告摘要】\n{reports[:2000]}\n\n"
            "请输出：\n"
            "## 裁决\n（看多 / 看空 / 中性 + 信心 0-100）\n"
            "## 决定性论据\n（最多3条，标注来自多头还是空头）\n"
            "## 被驳回的论点\n（说明为什么证据不足）\n"
            "## 给交易员的指示\n（建议仓位方向与需要重点监控的变量）"
        )
        return await self.chat(system, user, deep=True, max_tokens=900)


# -- 交易员 -------------------------------------------------------------------
class Trader(RoundtableAgent):
    role = "trader"

    async def plan(self, tk: RoundtableToolkit, reports: str,
                   judgement: str) -> str:
        system = (
            "你是执行交易员，把研究结论转化为可执行的交易计划。"
            "你只对赔率与执行细节负责，不重新做基本面判断。\n" + _NO_FABRICATE
        )
        user = (
            f"标的：{tk.name}（{tk.symbol}），{tk.trade_date}。\n\n"
            f"【研究经理裁决】\n{judgement}\n\n"
            f"【当前持仓与资金约束】\n{self.config.portfolio_note or '（未提供）'}\n\n"
            f"计划持有期：{tk.horizon} 个交易日。\n\n"
            f"【分析师要点】\n{reports[:1800]}\n\n"
            f"{tk.market_brief()}\n"
            "请输出：\n"
            "## 交易方向\n（买入 / 观望 / 卖出）\n"
            "## 执行计划\n（入场区间、止损位、目标位，必须落在数据块给出的关键位附近）\n"
            "## 仓位与节奏\n（建议仓位百分比、分几笔建仓）\n"
            "## 失效条件\n（什么情况证明这笔交易错了）"
        )
        return await self.chat(system, user, deep=True, max_tokens=800)


# -- 三方风控 -----------------------------------------------------------------
class RiskAnalyst(RoundtableAgent):
    """激进 / 中性 / 保守 三位风控，共用一套辩论逻辑，立场由 ``stance`` 决定。"""

    def __init__(self, gateway: ModelGateway, config: RoundtableConfig,
                 stance: str):
        super().__init__(gateway, config)
        self.stance = stance
        self.role = {
            "aggressive": "aggressive_analyst",
            "conservative": "conservative_analyst",
            "neutral": "neutral_analyst",
        }[stance]

    _PERSONA = {
        "aggressive": (
            "你是激进风控，主张在赔率合适时承担波动换取收益，反对过度保守错失机会。"
            "你会质疑保守方“为了不亏而放弃所有机会”。"
        ),
        "conservative": (
            "你是保守风控，优先保护本金，强调回撤、流动性与尾部风险。"
            "你会质疑激进方低估了极端行情下的损失。"
        ),
        "neutral": (
            "你是中性风控，在收益与风险之间寻找平衡，用仓位调节而不是非黑即白。"
            "你会指出激进方和保守方各自的偏误。"
        ),
    }

    async def speak(self, tk: RoundtableToolkit, trader_plan: str,
                    reports: str, history: str, round_no: int) -> str:
        system = (
            f"{self._PERSONA[self.stance]}\n"
            "你评审的是交易员给出的具体计划，必须落到仓位、止损与最坏情形上。\n"
            + _NO_FABRICATE
        )
        user = (
            f"标的：{tk.name}（{tk.symbol}），{tk.trade_date}。第 {round_no} 轮风控讨论。\n\n"
            f"【交易员计划】\n{trader_plan[:1800]}\n\n"
            f"【分析师要点】\n{reports[:1200]}\n\n"
            f"【风控讨论记录】\n{history or '（你是第一位发言者）'}\n\n"
            "请输出：\n"
            "1) 你对这个计划的核心意见（最多3条，引用数据块中的波动率/关键位）；\n"
            "2) 针对其他风控立场的反驳或让步（首轮可跳过）；\n"
            "3) 你建议的仓位与止损调整（给出具体数字或调整方向）。"
        )
        return await self.chat(system, user, max_tokens=600)


# -- 组合经理 -----------------------------------------------------------------
class PortfolioManager(RoundtableAgent):
    role = "portfolio_manager"

    #: 目标位选择的最小空间（单位：ATR 倍数）。低于这个距离的「目标价」在
    #: 10 日持有期里等于没有交易价值，却会给出一个看起来合理的假信号。
    MIN_TARGET_ATR = 1.0
    #: 目标位至少要离开现价的百分比（ATR 过小的低波动股用这条兜底）
    MIN_TARGET_PCT = 0.01
    #: 止损位同理：贴着现价的止损会被日内噪声直接打掉，而且会把盈亏比算得虚高
    #: （风险被低估 → RR 虚高 → 看起来是好机会，实际是必被打的止损）。
    MIN_STOP_ATR = 0.5
    MIN_STOP_PCT = 0.005
    #: 关键位不够用时的合成倍数（必须 ≥ 上面的最小距离，否则等于没兜住）
    SYNTH_TARGET_ATR = 1.5
    SYNTH_STOP_ATR = 1.0
    #: 可接受的最低盈亏比（盈利空间 / 止损空间）
    MIN_RR = 1.0

    async def decide(self, tk: RoundtableToolkit, reports: str, judgement: str,
                     trader_plan: str, risk_text: str,
                     memory_text: str) -> Dict[str, Any]:
        system = (
            "你是组合经理，拥有最终决策权。你要综合研究裁决、交易计划、三方风控意见，"
            "并参考历史决策的复盘教训，给出可执行的最终决定。\n" + _NO_FABRICATE
        )
        user = (
            f"标的：{tk.name}（{tk.symbol}），基准日 {tk.trade_date}，"
            f"最新价 {(tk.snapshot_core() or {}).get('price')} 元。\n\n"
            f"【研究经理裁决】\n{judgement[:1400]}\n\n"
            f"【交易员计划】\n{trader_plan[:1400]}\n\n"
            f"【三方风控讨论】\n{risk_text[:2000]}\n\n"
            f"【当前持仓与资金约束】\n{self.config.portfolio_note or '（未提供）'}\n\n"
            f"持有期固定为 {tk.horizon} 个交易日，仓位建议须遵守上述约束。\n\n"
            f"【历史决策与教训】\n{memory_text or '（无历史记忆，这是该标的首次圆桌）'}\n\n"
            "请只输出一个 JSON 对象：\n"
            '{"action":"buy|hold|sell","confidence":0.0-1.0,"position_pct":0-100,'
            '"entry_price":数字,"target_price":数字,"stop_loss":数字,'
            '"horizon_days":整数,"rationale":"不超过200字的决策理由",'
            '"key_risks":["风险1","风险2"],"memory_used":["参考的历史教训，可为空"],'
            '"lessons":"这次决策希望未来被记住的一条教训"}\n'
            "要求：entry/target/stop 必须是数字；若 action=hold，position_pct 可为 0 或维持现有仓位；"
            "confidence 反映你的真实把握，不要一律给 0.7。\n"
            "价格方向必须自洽：action=buy 时 target_price > entry_price > stop_loss；"
            "action=sell 时 target_price < entry_price < stop_loss（目标指下行目标，"
            "止损指出现反弹时的认错位）；并且务必保证盈亏比 "
            "(|target_price-entry_price| / |stop_loss-entry_price|) 不低于 1.2。"
        )
        data = await self.chat_json(system, user, deep=True, max_tokens=800)
        return self._normalize(data, tk)

    def _normalize(self, data: Dict[str, Any], tk: RoundtableToolkit) -> Dict[str, Any]:
        # 没有行情时保持 None，不要写成 0.0（0.0 会绕过方向校验并污染落库数据）
        price = (tk.snapshot_core() or {}).get("price") or None
        action = str(data.get("action", "hold")).lower()
        if action not in ("buy", "hold", "sell"):
            action = "hold"

        def _f(key: str, default: Optional[float] = None) -> Optional[float]:
            try:
                v = float(data.get(key))
                return round(v, 3) if v > 0 else default
            except (TypeError, ValueError):
                return default

        conf = data.get("confidence")
        try:
            conf = max(0.0, min(1.0, float(conf)))
        except (TypeError, ValueError):
            conf = 0.5

        pos = data.get("position_pct")
        try:
            pos = max(0.0, min(100.0, float(pos)))
        except (TypeError, ValueError):
            pos = 0.0

        risks = data.get("key_risks")
        if not isinstance(risks, list):
            risks = [str(risks)] if risks else []
        mem = data.get("memory_used")
        if not isinstance(mem, list):
            mem = [str(mem)] if mem else []

        entry = _f("entry_price", price)
        target = _f("target_price")
        stop = _f("stop_loss")

        # 方向自洽校验：模型给反了就直接丢弃该价格，而不是把错误数字落库
        # （「SELL 却给了一个高于现价的目标价」这类输出在四舍五入后极难人工发现）
        notes: List[str] = []
        if entry:
            if action == "buy":
                if target and target <= entry:
                    notes.append("模型给的目标价不高于入场价，已忽略目标价")
                    target = None
                if stop and stop >= entry:
                    notes.append("模型给的止损价不低于入场价，已忽略止损价")
                    stop = None
            elif action == "sell":
                if target and target >= entry:
                    notes.append("模型给的（下跌）目标价不低于现价，已忽略目标价")
                    target = None
                if stop and stop <= entry:
                    notes.append("模型给的（上方）止损价不高于现价，已忽略止损价")
                    stop = None
        if action == "hold":
            target = stop = None

        return {
            "action": action,
            "confidence": round(conf, 2),
            "position_pct": round(pos, 1),
            "entry_price": entry,
            "target_price": target,
            "stop_loss": stop,
            "risk_reward": self._rr(entry, target, stop),
            "horizon_days": tk.horizon,
            "rationale": str(data.get("rationale") or "")[:600],
            # 模型给过不可用的价格时，把原因并进风险提示，避免静默丢字段
            "key_risks": [str(x)[:160] for x in (notes + risks)[:5]],
            "memory_used": [str(x)[:200] for x in mem[:4]],
            "lessons": str(data.get("lessons") or "")[:300],
            "price_notes": notes,
        }

    # -- 兜底决策 -----------------------------------------------------------
    @staticmethod
    def _rr(entry: Optional[float], target: Optional[float],
            stop: Optional[float]) -> Optional[float]:
        """盈亏比 = 盈利空间 / 止损空间。任一价格缺失时返回 None。"""
        if not entry or not target or not stop:
            return None
        reward = abs(target - entry)
        risk = abs(stop - entry)
        if risk <= 0:
            return None
        return round(reward / risk, 2)

    @staticmethod
    def _parse_levels(items: Any) -> List[Dict[str, Any]]:
        """关键位统一成 [{"price": float, "label": str, "distance_pct": float|None}]。

        研判引擎给的是 dict 列表，旧代码与测试里也可能出现裸 float。
        """
        out: List[Dict[str, Any]] = []
        for it in items or []:
            if isinstance(it, dict):
                try:
                    px = float(it.get("price"))
                except (TypeError, ValueError):
                    continue
                out.append({"price": px, "label": str(it.get("label") or "关键位"),
                            "distance_pct": it.get("distance_pct")})
            else:
                try:
                    out.append({"price": float(it), "label": "关键位", "distance_pct": None})
                except (TypeError, ValueError):
                    continue
        return out

    @classmethod
    def _pick_level(cls, levels: List[Dict[str, Any]], price: float,
                    below: bool, min_move: float,
                    strict: bool = False) -> Optional[Dict[str, Any]]:
        """在价格某一侧挑一档关键位。

        只接受「距离 ≥ ``min_move``（价格百分比）」的档位，避免目标价/止损价贴脸。
        ``strict=True`` 时若没有档位满足距离要求就返回 None（由调用方改用
        ATR 合成）；``strict=False`` 时退回最近一档，保证用的是真实技术位
        而不是凭空造价格（项目原则：数值必须可追溯到数据）。
        """
        side: List[Dict[str, Any]] = []
        for lv in levels:
            gap = (price - lv["price"]) if below else (lv["price"] - price)
            if gap > 0:
                side.append(lv)
        if not side:
            return None
        if not price:
            return side[0]
        far = [lv for lv in side if abs(lv["price"] - price) / price >= min_move]
        if far:
            return far[0]
        return None if strict else side[0]

    @staticmethod
    def _atr(fc: Dict[str, Any], price: Optional[float]) -> float:
        """ATR 绝对金额：优先研判引擎给出的 levels.atr，其次用快照 atr_pct 反推。"""
        levels = fc.get("levels") or {}
        try:
            v = float(levels.get("atr"))
            if v > 0:
                return v
        except (TypeError, ValueError):
            pass
        tech = fc.get("technical") or {}
        try:
            pct = float((tech.get("snapshot") or {}).get("atr_pct"))
            if pct > 0 and price:
                return price * pct / 100.0
        except (TypeError, ValueError):
            pass
        return (price or 0.0) * 0.02

    def fallback_decision(self, tk: RoundtableToolkit, reason: str) -> Dict[str, Any]:
        """确定性兜底：用量化评分映射方向，并按 ATR 选目标位/止损位。

        关键约束（修复「目标价贴脸、盈亏比 0.2:1」的问题）：
        - 目标位必须离现价至少 ``MIN_TARGET_ATR`` 个 ATR（且不低于 1%），
          否则退到更远的一档关键位；都太近则用 ATR 合成；
        - 止损位必须离现价至少 ``MIN_STOP_ATR`` 个 ATR（且不低于 0.5%），
          否则改用 ``suggested_stop`` / ATR 合成——贴着现价的止损会把风险算小、
          把盈亏比算高，比目标贴脸更危险；
        - 盈亏比低于 ``MIN_RR`` 时不改方向（方向来自评分），但写进 key_risks
          与 rationale，让使用者看见这是「盈亏比不划算」而不是「强烈看空」。
        """
        fc = tk.forecast or {}
        tech = fc.get("technical") or {}
        score = tech.get("score") or 0
        prob = (fc.get("probability") or {}).get("probability")
        price = (tk.snapshot_core() or {}).get("price")
        levels = fc.get("levels") or {}

        action = "buy" if score >= 20 else ("sell" if score <= -20 else "hold")
        supports = self._parse_levels(levels.get("supports"))
        resistances = self._parse_levels(levels.get("resistances"))

        atr = self._atr(fc, price)
        px = float(price) if price else None
        # 距离一律换算成「占现价的比例」，与 _pick_level 的口径保持一致
        target_move = max(
            (self.MIN_TARGET_ATR * atr) / px if px else 0.0,
            self.MIN_TARGET_PCT,
        )
        stop_move = max(
            (self.MIN_STOP_ATR * atr) / px if px else 0.0,
            self.MIN_STOP_PCT,
        )

        target: Optional[float] = None
        stop: Optional[float] = None
        target_note = ""
        stop_note = ""

        if px and action == "buy":
            lv = self._pick_level(resistances, px, below=False, min_move=target_move)
            if lv:
                target, target_note = lv["price"], lv["label"]
            elif atr:
                target = round(px + self.SYNTH_TARGET_ATR * atr, 3)
                target_note = f"{self.SYNTH_TARGET_ATR:g}×ATR"
            sl = self._pick_level(supports, px, below=True,
                                  min_move=stop_move, strict=True)
            if sl:
                stop, stop_note = sl["price"], sl["label"]
            elif levels.get("suggested_stop"):
                stop, stop_note = float(levels["suggested_stop"]), "建议止损"
            elif atr:
                stop = round(px - self.SYNTH_STOP_ATR * atr, 3)
                stop_note = f"{self.SYNTH_STOP_ATR:g}×ATR"
        elif px and action == "sell":
            # 卖出方向：目标是下方支撑（跌到哪），止损是上方阻力（反弹到哪算看错）
            lv = self._pick_level(supports, px, below=True, min_move=target_move)
            if lv:
                target, target_note = lv["price"], lv["label"]
            elif atr:
                target = round(px - self.SYNTH_TARGET_ATR * atr, 3)
                target_note = f"{self.SYNTH_TARGET_ATR:g}×ATR"
            rl = self._pick_level(resistances, px, below=False,
                                  min_move=stop_move, strict=True)
            if rl:
                stop, stop_note = rl["price"], rl["label"]
            elif atr:
                stop = round(px + self.SYNTH_STOP_ATR * atr, 3)
                stop_note = f"{self.SYNTH_STOP_ATR:g}×ATR"

        target = round(target, 3) if target else None
        stop = round(stop, 3) if stop else None
        rr = self._rr(px, target, stop)

        risks: List[str] = [f"决策链路降级：{str(reason)[:120]}，结论仅由量化规则生成"]
        if action != "hold" and rr is not None and rr < self.MIN_RR:
            risks.append(f"盈亏比仅 {rr}:1（低于 {self.MIN_RR}:1），并非高胜算机会")
        if action != "hold" and (target is None or stop is None):
            risks.append("关键位数据不全，目标价/止损价缺失")

        rr_text = f"，盈亏比 {rr}:1" if rr is not None else ""
        return {
            "action": action,
            "confidence": 0.4,
            "position_pct": 20.0 if action == "buy" else 0.0,
            "entry_price": px,
            "target_price": target,
            "stop_loss": stop,
            "risk_reward": rr,
            "target_note": target_note,
            "stop_note": stop_note,
            "horizon_days": tk.horizon,
            "rationale": (
                f"决策链路降级（{str(reason)[:120]}）：语言模型未产出可用结论，"
                f"按量化技术评分 {score} 与上涨概率 "
                f"{prob if prob is not None else '—'}% 生成规则决策{rr_text}。"
            ),
            "key_risks": risks,
            "memory_used": [],
            "lessons": "",
            "degraded": True,
        }


# -- 反思 ---------------------------------------------------------------------
class Reflector(RoundtableAgent):
    role = "reflector"

    async def reflect(self, symbol: str, name: str, signal: str,
                      entry: Optional[float], exit_price: Optional[float],
                      ret: Optional[float], alpha: Optional[float],
                      rationale: str) -> str:
        system = (
            "你是交易复盘教练，负责把一次已结算的决策写成一段可复用的教训。"
            "不要评价对错本身，而要提炼下次可以改进的判断依据。"
        )
        user = (
            f"标的：{name}（{symbol}）。当时的决策：{signal}，"
            f"入场 {entry}，持有期结束价 {exit_price}，实际收益 {ret}%，"
            f"相对沪深300超额 {alpha}%。\n"
            f"当时理由：{rationale[:500]}\n\n"
            "请用 100-150 字写一段第一人称复盘（“我当时……”），"
            "指出哪个判断被验证、哪个假设被证伪，以及下次遇到类似形态该怎么做。"
        )
        try:
            return await self.chat(system, user, deep=False, max_tokens=300)
        except LLMUnavailable:
            return (
                f"持有期内收益 {ret}%，相对基准超额 {alpha}%。"
                f"决策方向 {signal}，后续需复核当时的核心理由是否仍然成立。"
            )
