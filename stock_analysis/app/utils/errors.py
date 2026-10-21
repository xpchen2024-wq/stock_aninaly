# ============================================================================
# AI Stock Analysis Platform - LLM Error Explanation
# ============================================================================
"""把各服务商返回的原始错误翻译成「中文原因 + 修复建议」。

模型连接测试失败时，litellm / openrouter / opencode 返回的是英文异常堆栈
（例如 ``RateLimitError: ... free-models-per-day``），使用者很难判断到底是
配额用完、Key 过期还是模型下线。这里做一次归一化，让界面直接给出可执行动作。
"""
from __future__ import annotations

from typing import Tuple


def explain_llm_error(error: str) -> str:
    """返回可直接展示给用户的中文原因说明。"""
    if not error:
        return "未知错误"

    low = error.lower()

    # 限流 / 配额
    if "free-models-per-day" in low:
        return ("该模型的每日免费调用额度已用尽（openrouter 每个免费模型 50 次/天，"
                "次日 08:00 北京时间重置）。建议改选其他模型，或等待额度恢复。")
    if "temporarily rate-limited" in low or "provider returned error" in low:
        return ("上游服务商临时限流（不是你的账户额度问题），稍后重试通常即可恢复。")
    if "rate limit" in low or "rate_limit" in low or "429" in low:
        return "触发服务商调用频率限制。建议降低并发或稍后重试。"
    if "quota" in low or "insufficient_credits" in low or "insufficient credit" in low:
        return "账户额度不足。请前往服务商控制台充值后重试。"

    # 鉴权
    if "api_key_expired" in low or "api key has expired" in low or "key expired" in low:
        return ("API Key 已过期。请到 OpenCode Console 重新生成 Key，"
                "在「AI 模型配置管理」中更新并重新启用该配置。")
    if "authentication" in low or "401" in low or "invalid api key" in low or "missing authentication" in low:
        return "API Key 无效或未正确填写，请检查 Key 是否完整。"

    # 模型本身
    if "not available in your region" in low or "region" in low:
        return "该模型在当前地区不可用，请改选其他模型。"
    if "model is unavailable" in low:
        return ("该模型在服务商侧不可用。常见原因：模型 ID 已下线，或该配置的 API Key 已失效"
                "（部分服务商对失效 Key 也返回此提示）。建议先到服务商控制台确认 Key 状态，"
                "再改选当前可用的模型 ID。")
    if "unavailable" in low:
        return "该模型在服务商侧已下线，请在模型配置中改选当前可用的模型 ID。"
    if "not found" in low or "does not exist" in low:
        return "模型 ID 不存在，请在模型配置中核对模型名称。"
    if "not supported" in low or "unsupported" in low:
        return "该服务商不支持此模型或该请求方式，请更换模型或平台。"

    # 本地状态
    if "disabled or cooling down" in low:
        return ("该配置已停用，或刚失败进入冷却期（默认 10~30 分钟）。"
                "可稍后重试，或在模型配置中重新启用。")
    if "no usable model" in low or "all" in low and "model(s) failed" in low:
        return "所有候选模型都调用失败，请检查各模型配置与账户状态。"

    # 网络
    if "timeout" in low or "timed out" in low:
        return "请求超时，服务商响应过慢或网络不通，请稍后重试。"
    if "connection" in low or "dns" in low or "ssl" in low:
        return "网络连接失败，无法访问服务商接口，请检查网络或代理设置。"

    # 兜底：截断原始信息
    cleaned = " ".join(error.split())
    return cleaned[:260] if cleaned else "未知错误"


def explain_llm_error_with_hint(error: str) -> Tuple[str, str]:
    """返回 (原因, 建议动作) 二元组，便于界面分开展示。"""
    reason = explain_llm_error(error)
    low = (error or "").lower()
    if "free-models-per-day" in low:
        action = "改选其他免费模型或等待次日额度重置"
    elif "temporarily rate-limited" in low or "provider returned error" in low:
        action = "稍后重试即可（上游临时限流）"
    elif "api_key_expired" in low or "api key has expired" in low:
        action = "重新生成 API Key 并更新配置"
    elif "model is unavailable" in low:
        action = "确认 API Key 状态并改选可用模型 ID"
    elif "unavailable" in low or "not found" in low:
        action = "在模型配置中改选可用模型 ID"
    elif "rate limit" in low or "429" in low:
        action = "稍后重试或降低并发"
    elif "region" in low:
        action = "改选其他地区可用的模型"
    else:
        action = "查看下方原始错误信息"
    return reason, action
