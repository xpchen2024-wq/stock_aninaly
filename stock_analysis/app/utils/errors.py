# ============================================================================
# AI Stock Analysis Platform - LLM Error Explanation
# ============================================================================
"""把各服务商返回的原始错误翻译成「中文原因 + 修复建议」。

模型连接测试失败时，litellm / openrouter / opencode 返回的是英文异常堆栈
（例如 ``RateLimitError: ... free-models-per-day``），使用者很难判断到底是
配额用完、Key 过期还是平台策略不允许。这里做一次归一化，让界面直接给出
可执行动作——尤其是下面两类「看起来像 Bug、其实是上游策略」的情况：

* OpenCode Zen 的 ``-free`` / ``big-pickle`` 免费模型：官方 v2 文档明确只允许
  在 OpenCode 客户端内使用，其他客户端即使带有效 Console Key 也返回 403；
* OpenRouter 免费模型：按账户共享 50 次/天，额度用尽后 429
  ``free-models-per-day``，需等重置或充值 $10 解锁 1000 次/天。
"""
from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from typing import Optional, Tuple

# 用户与服务器都在 UTC+8，直接按北京时间展示重置时刻
_CST = timezone(timedelta(hours=8))

_RESET_PATTERNS = (
    # OpenRouter 429 响应里的 X-RateLimit-Reset（毫秒时间戳）
    re.compile(r'x-ratelimit-reset"?\s*:\s*"?(\d{10,13})', re.I),
    # 其他写法：reset_at=1793232000 / "reset": 1793232000000
    re.compile(r'reset[\w\-]*"?\s*[:=]\s*"?(\d{10,13})', re.I),
)


def parse_rate_limit_reset(error: str) -> Optional[datetime]:
    """从错误文本里解析「配额重置时刻」（返回北京时间）。

    例如 OpenRouter 的 429 会带上 ``"X-RateLimit-Reset":"1793232000000"``，
    直接用它算真实重置时刻，比写死「次日 08:00」更准确，也让网关能把该配置
    精确冷却到额度恢复，而不是反复发起注定失败的请求。
    """
    if not error:
        return None
    for pattern in _RESET_PATTERNS:
        match = pattern.search(error)
        if not match:
            continue
        try:
            ts = int(match.group(1))
        except (TypeError, ValueError):
            continue
        if ts > 10 ** 11:                 # 毫秒 → 秒
            ts //= 1000
        if not (1_600_000_000 <= ts <= 4_100_000_000):   # 明显不是时间戳就跳过
            continue
        try:
            return datetime.fromtimestamp(ts, _CST)
        except (OverflowError, OSError, ValueError):
            continue
    return None


def _fmt_reset(moment: Optional[datetime]) -> str:
    return f"{moment:%m-%d %H:%M}（北京时间）" if moment else ""


def _explain(error: str) -> Tuple[str, str]:
    """返回 (中文原因, 建议动作)。所有分支集中在此，避免两套文案互相走偏。"""
    if not error:
        return "未知错误", "查看服务端日志"

    low = error.lower()

    # ------------------------------------------------------------------
    # OpenCode Zen / Console：免费模型禁止外部调用（官方 v2 策略）
    # ------------------------------------------------------------------
    if ("free usage limit reached" in low
            or "freeusagelimiterror" in low
            or "free models work in opencode" in low
            or "free-tier chat models" in low
            or "only available from within opencode" in low
            or ("403" in low and "free" in low and "opencode" in low)):
        return (
            "OpenCode Zen 的免费模型（*-free、big-pickle）只允许在 OpenCode 客户端内使用。"
            "官方 v2 文档明确：其他客户端即使带有效 Console API Key 也会收到 403，"
            "这是平台策略，不是 Key 写错或代码问题。外部集成请改用付费模型"
            "（如 glm-5.3-flash、deepseek-v4.1-flash，Console workspace 需有 credits）"
            "或订阅 OpenCode Go。",
            "改用 OpenCode Zen 付费模型、订阅 OpenCode Go，或切换到其他平台",
        )

    # Go 订阅要求稳定的会话 ID，缺失时上游直接返回 400
    if "missingsessionid" in low or "x-opencode-session" in low:
        return (
            "OpenCode Go 订阅要求每个请求携带稳定的 x-opencode-session 请求头，"
            "并使用自定义 User-Agent 标识客户端；缺失该头时上游拒绝路由。",
            "在该配置的 extra_headers 中补充 x-opencode-session 与 User-Agent",
        )

    # ------------------------------------------------------------------
    # OpenRouter：免费模型按账户共享的每日额度
    # ------------------------------------------------------------------
    if "free-models-per-day" in low:
        reset = parse_rate_limit_reset(error)
        when = f"，额度将于 {_fmt_reset(reset)}重置" if reset else \
            "，额度按 UTC 自然日重置（北京时间次日 08:00）"
        return (
            f"该账户的 OpenRouter 免费模型每日额度（50 次/天）已用尽{when}。"
            "额度是账户级共享的，换一个 :free 模型也不会增加次数；"
            "账户累计充值满 10 美元可解锁 1000 次/天。若账户余额为 0，"
            "直接改用付费模型会返回 402，需先充值。",
            "等待额度重置或充值（≥$10 解锁 1000 次/天）；未充值前请切换到其他平台",
        )

    # ------------------------------------------------------------------
    # 鉴权类
    # ------------------------------------------------------------------
    if "invalid credential" in low or "invalid_credential" in low:
        return (
            "该平台不认识这个 API Key：多半是 Key 与所选平台不匹配"
            "（例如把 OpenRouter 的 sk-or-v1-… 填进了 OpenCode Zen 配置），"
            "或该 Key 已在控制台被删除。",
            "到对应平台重新生成 Key 并按其前缀重新填写",
        )
    if "api_key_expired" in low or "api key has expired" in low or "key expired" in low:
        return (
            "API Key 已过期，服务商拒绝鉴权。请到平台控制台重新生成 Key，"
            "在「AI 模型配置管理」里更新该配置并重新启用。",
            "重新生成 API Key 并更新配置",
        )
    if ("authentication" in low or "401" in low or "invalid api key" in low
            or "missing authentication" in low or "unauthorized" in low):
        return (
            "API Key 无效或未正确填写（服务商返回 401）。请确认 Key 完整、"
            "未被截断，且与所选平台一致。",
            "核对 API Key 是否属于该平台并重新填写",
        )

    # ------------------------------------------------------------------
    # 限流 / 额度
    # ------------------------------------------------------------------
    if "temporarily rate-limited" in low or "provider returned error" in low:
        return (
            "上游服务商临时限流（不是你的账户额度问题），稍后重试通常即可恢复。",
            "稍后重试即可（上游临时限流）",
        )
    if ("rate limit" in low or "rate_limit" in low or "429" in low
            or "too many requests" in low):
        return (
            "触发服务商的调用频率限制（429）。免费/低配额账户的阈值较低，"
            "建议降低并发或稍后重试。",
            "稍后重试或降低并发",
        )
    if ("quota" in low or "insufficient_credits" in low
            or "insufficient credit" in low or "add credits" in low
            or "billing" in low):
        return (
            "账户额度/余额不足。请前往服务商控制台充值后重试。",
            "前往服务商控制台充值",
        )

    # ------------------------------------------------------------------
    # 模型本身 / 地区
    # ------------------------------------------------------------------
    if "not available in your region" in low or "not available in your country" in low:
        return (
            "该模型在你所在地区不可用（部分免费模型明确排除中国大陆及港澳地区）。"
            "请改选其他模型。",
            "改选其他地区可用的模型",
        )
    if "model is unavailable" in low:
        return (
            "该模型在服务商侧不可用。常见原因：模型 ID 已下线，或该配置的 API Key 已失效"
            "（部分服务商对失效 Key 也返回此提示）。建议先确认 Key 状态，再改选可用的模型 ID。",
            "确认 API Key 状态并改选可用模型 ID",
        )
    if "model_not_found" in low or "not found" in low or "does not exist" in low:
        return (
            "模型 ID 不存在或已被下线，请在模型配置中核对模型名称。",
            "在模型配置中改选可用模型 ID",
        )
    if "unavailable" in low:
        return (
            "该模型在服务商侧已下线或暂不可用，请改选当前可用的模型 ID。",
            "改选可用模型 ID",
        )
    if "not supported" in low or "unsupported" in low:
        return (
            "该服务商不支持此模型或该请求方式，请更换模型或平台。",
            "更换模型或平台",
        )

    # ------------------------------------------------------------------
    # 本地状态
    # ------------------------------------------------------------------
    if "disabled or cooling down" in low:
        return (
            "该配置已停用，或刚失败进入冷却期（默认 10~30 分钟）。"
            "可稍后重试，或在模型配置中重新启用。",
            "稍后重试或重新启用该配置",
        )
    if "no usable model" in low or ("all" in low and "model(s) failed" in low):
        return (
            "所有候选模型都调用失败，请检查各模型配置与账户状态。",
            "逐个「测试」模型配置定位失败原因",
        )

    # ------------------------------------------------------------------
    # 网络
    # ------------------------------------------------------------------
    if "timeout" in low or "timed out" in low:
        return (
            "请求超时，服务商响应过慢或网络不通，请稍后重试。",
            "稍后重试；长期超时请检查网络/代理",
        )
    if "connection" in low or "dns" in low or "ssl" in low:
        return (
            "网络连接失败，无法访问服务商接口，请检查网络或代理设置。",
            "检查网络、代理或 Base URL 是否正确",
        )

    # 兜底：截断原始信息
    cleaned = " ".join(error.split())
    return (cleaned[:260] if cleaned else "未知错误"), "查看下方原始错误信息"


def explain_llm_error(error: str) -> str:
    """返回可直接展示给用户的中文原因说明。"""
    return _explain(error)[0]


def explain_llm_error_with_hint(error: str) -> Tuple[str, str]:
    """返回 (原因, 建议动作) 二元组，便于界面分开展示。"""
    return _explain(error)
