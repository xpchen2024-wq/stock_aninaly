#!/usr/bin/env python3
"""Regression tests for LLM failure diagnosis (AI 模型配置页).

覆盖三处「看起来像 Bug、其实是上游策略或填错 Key」的判定：

- ``app.utils.errors``   : 中文原因翻译 + 配额重置时间解析
- ``app.model_gateway.detect_key_mismatch`` : Key 归属 / 平台一致性
- ``ModelGateway``        : Key 错配短路、配额类错误按真实重置时刻冷却
"""
import asyncio
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

from app.utils.errors import (
    explain_llm_error,
    explain_llm_error_with_hint,
    parse_rate_limit_reset,
)
from app.api.models import AvailableModelsRequest, fetch_available_models
from app.config import get_settings
from app.model_gateway import (
    LLMConfig,
    LLMProvider,
    ModelGateway,
    detect_key_mismatch,
    normalize_extra_headers,
    normalize_opencode_model,
    encrypt_api_key,
    register_model_config,
)

# OpenRouter 在 429 响应里带回的重置时间戳（2026-10-29 08:00 北京时间）
RESET_MS = '"X-RateLimit-Reset":"1793232000000"'
ZEN_FREE_ERROR = (
    'litellm.RateLimitError: RateLimitError: OpenAIException - Free usage limit reached. '
    'Free models work in OpenCode (https://opencode.ai); subscribe to OpenCode Go'
)
OPENROUTER_QUOTA_ERROR = (
    'litellm.RateLimitError: RateLimitError: OpenrouterException - {"error":{"message":'
    '"Rate limit exceeded: free-models-per-day. Add 10 credits to unlock 1000 free model '
    f'requests per day","code":429,"metadata":{{"headers":{{"X-RateLimit-Limit":"50",{RESET_MS}}}}}}}'
)


def _quota_error_resetting_in(hours: float = 12) -> str:
    """构造一个「配额在 N 小时后重置」的 429 报文。

    不能用写死的 RESET_MS 来测冷却时长：那个时间戳一旦过去，
    冷却就会被算成 0（重置时间已到），测试会随日期推移凭空变红。
    """
    ts = int((datetime.now(timezone.utc) + timedelta(hours=hours)).timestamp())
    return OPENROUTER_QUOTA_ERROR.replace(RESET_MS, f'"X-RateLimit-Reset":"{ts * 1000}"')


# ===================================================================
# 配额重置时间解析
# ===================================================================
class TestParseRateLimitReset:
    def test_parses_millisecond_timestamp(self):
        moment = parse_rate_limit_reset(OPENROUTER_QUOTA_ERROR)
        assert moment is not None
        assert moment.astimezone(timezone(timedelta(hours=8))).strftime("%Y-%m-%d %H:%M") == \
            "2026-10-29 08:00"

    def test_parses_second_timestamp(self):
        moment = parse_rate_limit_reset('"reset": 1793232000')
        assert moment is not None and moment.year == 2026

    def test_ignores_non_timestamp_and_garbage(self):
        assert parse_rate_limit_reset("plain failure without reset") is None
        assert parse_rate_limit_reset("reset: 12345") is None      # 位数不足
        assert parse_rate_limit_reset("") is None


# ===================================================================
# 错误翻译
# ===================================================================
class TestExplainLlmError:
    def test_zen_free_tier_is_upstream_policy_not_key_problem(self):
        reason, hint = explain_llm_error_with_hint(ZEN_FREE_ERROR)
        assert "OpenCode" in reason
        assert "客户端" in reason            # 说明只允许在客户端内使用
        assert "OpenCode Go" in hint

    def test_openrouter_daily_quota_mentions_real_reset_time(self):
        reason, hint = explain_llm_error_with_hint(OPENROUTER_QUOTA_ERROR)
        assert "10-29 08:00" in reason       # 用真实重置时间，而不是写死「次日」
        assert "50 次/天" in reason
        assert "10 美元" in reason
        assert "重置" in hint or "充值" in hint

    def test_invalid_credential_points_to_platform_mismatch(self):
        reason, _ = explain_llm_error_with_hint("OpenAIException - Invalid credential")
        assert "不匹配" in reason

    def test_expired_key_asks_for_new_key(self):
        reason, hint = explain_llm_error_with_hint("ZenException - api_key_expired")
        assert "过期" in reason
        assert "重新生成" in hint

    def test_region_restriction(self):
        reason, _ = explain_llm_error_with_hint("model not available in your region")
        assert "地区" in reason

    def test_unknown_error_falls_back_to_raw_text(self):
        raw = "SomeWeirdFailure: boom"
        assert "SomeWeirdFailure" in explain_llm_error(raw)

    def test_empty_error(self):
        assert explain_llm_error("") == "未知错误"


# ===================================================================
# Key 归属 / 平台一致性
# ===================================================================
class TestDetectKeyMismatch:
    def test_openrouter_key_on_opencodezen_is_flagged(self):
        warning = detect_key_mismatch("opencodezen", "sk-or-v1-0123456789abcdef")
        assert warning and "OpenRouter" in warning and "不匹配" in warning

    def test_zen_shaped_key_on_openrouter_is_flagged(self):
        warning = detect_key_mismatch("openrouter", "sk-3lrfMaBxxxxxxxx")
        assert warning and "sk-or-v1-" in warning

    def test_matching_pairs_are_silent(self):
        assert detect_key_mismatch("openrouter", "sk-or-v1-abc") is None
        assert detect_key_mismatch("opencodezen", "sk-3lrfMaBabc") is None
        assert detect_key_mismatch("openai", "sk-proj-abc") is None

    def test_custom_platform_is_never_judged(self):
        assert detect_key_mismatch("custom", "sk-or-v1-abc") is None
        assert detect_key_mismatch("custom", "") is None


# ===================================================================
# 网关：错配短路 + 配额冷却
# ===================================================================
class TestGatewayFailureHandling:
    def test_mismatched_key_short_circuits_without_network_call(self):
        gw = ModelGateway(encryption_secret="s")
        gw.add_config(LLMConfig(
            id="zen", name="Zen", provider=LLMProvider.OPENCODEZEN,
            base_url="https://opencode.ai/inference/openai/v1",
            api_key="sk-or-v1-0123456789abcdef", model_name="deepseek-v4.1-flash-free",
        ))
        import litellm
        with patch.object(litellm, "acompletion", side_effect=AssertionError("must not be called")):
            resp = asyncio.run(gw.chat(
                messages=[{"role": "user", "content": "hi"}], config_id="zen"
            ))
        assert not resp.success
        assert "归属异常" in resp.error

    def test_quota_cooldown_uses_real_reset_time(self):
        gw = ModelGateway(encryption_secret="s")
        gw._mark_failure("openrouter-cfg", _quota_error_resetting_in(12))
        remaining = gw._cooldown_remaining("openrouter-cfg")
        # 距重置还有 12 小时，远大于固定 600s 的限流冷却
        assert remaining > 600
        assert remaining <= 86400

    def test_plain_error_keeps_short_cooldown(self):
        gw = ModelGateway(encryption_secret="s")
        gw._mark_failure("cfg", "upstream timeout")
        assert 0 < gw._cooldown_remaining("cfg") <= 1800

    def test_cooldown_config_is_skipped_as_candidate(self):
        gw = ModelGateway(encryption_secret="s")
        gw.add_config(LLMConfig(id="a", name="A", api_key="sk-proj-a", model_name="gpt-4o"))
        gw.add_config(LLMConfig(id="b", name="B", api_key="sk-proj-b", model_name="gpt-4o-mini"))
        gw.set_default("a")
        gw._mark_failure("a", OPENROUTER_QUOTA_ERROR)
        candidates = [c.id for c in gw._candidate_configs(None)]
        assert candidates == ["b"]

    def test_probe_reports_mismatch_instead_of_calling_provider(self):
        gw = ModelGateway(encryption_secret="s")
        import litellm
        cfg = LLMConfig(
            id="x", name="X", provider=LLMProvider.OPENCODEZEN,
            api_key="sk-or-v1-0123456789abcdef", model_name="glm-5.3-flash-free",
        )
        with patch.object(litellm, "acompletion", side_effect=AssertionError("must not be called")):
            result = asyncio.run(gw.probe(cfg))
        assert result["success"] is False
        assert "归属异常" in result["error"]


# ===================================================================
# 网关：结构化输出与「避开某模型重试」
# ===================================================================
def _completion(model: str, finish_reason: str = "stop") -> MagicMock:
    resp = MagicMock()
    resp.choices = [MagicMock()]
    resp.choices[0].message.content = f"ok:{model}"
    resp.choices[0].finish_reason = finish_reason
    resp.usage.total_tokens = 11
    return resp


class TestStructuredOutput:
    def test_exclude_config_ids_switches_to_next_model(self):
        """JSON 失败后换模型重试：必须能跳过刚失败的那个配置。"""
        import litellm
        gw = ModelGateway(encryption_secret="s")
        gw.add_config(LLMConfig(id="a", name="A", api_key="sk-proj-a", model_name="model-a"))
        gw.add_config(LLMConfig(id="b", name="B", api_key="sk-proj-b", model_name="model-b"))
        gw.set_default("a")

        calls = []

        async def _fake(**kwargs):
            calls.append(kwargs["model"])
            return _completion(kwargs["model"])

        with patch.object(litellm, "acompletion", side_effect=_fake):
            resp = asyncio.run(gw.chat(
                messages=[{"role": "user", "content": "hi"}],
                prefer_config_id="a", exclude_config_ids=["a"],
            ))
        assert resp.success and resp.model == "model-b"
        assert resp.config_id == "b"
        assert calls == ["model-b"]

    def test_excluding_every_model_is_ignored(self):
        """排除后无候选时必须忽略排除，而不是把自己变成「无模型可用」。"""
        import litellm
        gw = ModelGateway(encryption_secret="s")
        gw.add_config(LLMConfig(id="a", name="A", api_key="sk-proj-a", model_name="model-a"))

        async def _fake(**kwargs):
            return _completion(kwargs["model"])

        with patch.object(litellm, "acompletion", side_effect=_fake):
            resp = asyncio.run(gw.chat(
                messages=[{"role": "user", "content": "hi"}],
                prefer_config_id="a", exclude_config_ids=["a"],
            ))
        assert resp.success and resp.config_id == "a"

    def test_response_format_is_sent_and_finish_reason_returned(self):
        import litellm
        gw = ModelGateway(encryption_secret="s")
        gw.add_config(LLMConfig(id="a", name="A", api_key="sk-proj-a", model_name="model-a"))

        seen = {}

        async def _fake(**kwargs):
            seen.update(kwargs)
            return _completion(kwargs["model"], finish_reason="length")

        with patch.object(litellm, "acompletion", side_effect=_fake):
            resp = asyncio.run(gw.chat(
                messages=[{"role": "user", "content": "hi"}],
                response_format={"type": "json_object"},
            ))
        assert seen["response_format"] == {"type": "json_object"}
        assert resp.finish_reason == "length"      # 截断必须能被上层看见

    def test_model_rejecting_response_format_is_retried_without_it(self):
        """免费/自建模型报 400 不认识 response_format 时，降级为纯提示词约束。"""
        import litellm
        gw = ModelGateway(encryption_secret="s")
        gw.add_config(LLMConfig(id="a", name="A", api_key="sk-proj-a", model_name="model-a"))

        seen = []
        attempts = {"n": 0}

        async def _fake(**kwargs):
            seen.append(dict(kwargs))
            attempts["n"] += 1
            if attempts["n"] == 1:
                raise Exception('Error code: 400 - {"error":{"message":'
                                '"response_format is not supported by this model"}}')
            return _completion(kwargs["model"])

        with patch.object(litellm, "acompletion", side_effect=_fake):
            resp = asyncio.run(gw.chat(
                messages=[{"role": "user", "content": "hi"}],
                response_format={"type": "json_object"},
            ))
        assert resp.success is True
        assert "response_format" in seen[0]
        assert "response_format" not in seen[1]

    def test_oversized_max_tokens_is_retried_with_smaller_budget(self):
        """部分模型限制输出长度，4096 被拒时应降配重试而不是整条链路失败。"""
        import litellm
        gw = ModelGateway(encryption_secret="s")
        gw.add_config(LLMConfig(id="a", name="A", api_key="sk-proj-a", model_name="model-a"))

        seen = []
        attempts = {"n": 0}

        async def _fake(**kwargs):
            seen.append(kwargs.get("max_tokens"))
            attempts["n"] += 1
            if attempts["n"] == 1:
                raise Exception('Error code: 400 - max_tokens is too large: '
                                'maximum context length is 2048')
            return _completion(kwargs["model"])

        with patch.object(litellm, "acompletion", side_effect=_fake):
            resp = asyncio.run(gw.chat(
                messages=[{"role": "user", "content": "hi"}], max_tokens=4096,
            ))
        assert resp.success is True
        assert seen[0] == 4096 and seen[1] < 4096

    def test_both_degradations_apply_to_the_same_model(self):
        """既拒 response_format 又限输出长度的模型：两次降配要能叠加。"""
        import litellm
        gw = ModelGateway(encryption_secret="s")
        gw.add_config(LLMConfig(id="a", name="A", api_key="sk-proj-a", model_name="model-a"))

        seen = []
        attempts = {"n": 0}

        async def _fake(**kwargs):
            attempts["n"] += 1
            seen.append(("response_format" in kwargs, kwargs.get("max_tokens")))
            if "response_format" in kwargs:
                raise Exception('Error code: 400 - {"message":"response_format unsupported"}')
            if kwargs.get("max_tokens") == 3200:
                raise Exception('Error code: 400 - max_tokens too large')
            return _completion(kwargs["model"])

        with patch.object(litellm, "acompletion", side_effect=_fake):
            resp = asyncio.run(gw.chat(
                messages=[{"role": "user", "content": "hi"}],
                max_tokens=3200, response_format={"type": "json_object"},
            ))
        assert resp.success is True
        assert seen == [(True, 3200), (False, 3200), (False, 1600)]


# ===================================================================
# OpenCode Zen / Console 官方 v2 API 兼容
# 参考 https://opencode.ai/v2/docs/console/{models,harnesses,go}
# ===================================================================
class TestOpenCodeZenV2Compat:
    def test_model_id_prefix_is_stripped(self):
        """opencode/ 前缀只用于客户端配置，直接调 API 必须去掉。"""
        assert normalize_opencode_model("opencode/glm-5.3-flash") == "glm-5.3-flash"
        assert normalize_opencode_model("opencode-go/kimi-k3") == "kimi-k3"
        assert normalize_opencode_model("glm-5.3-flash") == "glm-5.3-flash"

    def test_console_key_belongs_to_zen(self):
        warning = detect_key_mismatch("openai", "oc_sk_1a2b3c4d5e6f_abc")
        assert warning and "OpenCode Zen" in warning and "不匹配" in warning

    def test_placeholder_key_is_flagged(self):
        warning = detect_key_mismatch("opencodezen", "oc_sk_your_personal_api_key")
        assert warning and "占位符" in warning and "oc_sk_" in warning

    def test_legacy_zen_key_still_accepted(self):
        """旧形态 Key 不能被强前缀校验误伤。"""
        assert detect_key_mismatch("opencodezen", "sk-3lrfMaBabc") is None

    def test_free_tier_403_is_explained(self):
        reason, hint = explain_llm_error_with_hint(
            'litellm.PermissionDeniedError: 403 {"error":"Free-tier chat models '
            'are only available from within OpenCode"}'
        )
        assert "403" in reason and "付费模型" in reason
        assert "OpenCode Go" in hint

    def test_go_session_header_error_is_explained(self):
        reason, hint = explain_llm_error_with_hint(
            'MissingSessionID: Request is missing x-opencode-session'
        )
        assert "x-opencode-session" in reason
        assert "extra_headers" in hint

    def test_extra_headers_registered_and_forwarded(self):
        """Go 订阅的 session 头必须一路透传到 LiteLLM，且模型名去掉客户端前缀。"""
        import litellm

        secret = "test-secret"
        gw = ModelGateway(secret)
        cfg = MagicMock()
        cfg.id = "zen-go-1"
        cfg.name = "Zen Go"
        cfg.platform = "opencodezen"
        cfg.base_url = "https://opencode.ai/inference/go/openai/v1"
        cfg.api_key_encrypted = encrypt_api_key("oc_sk_realkey", secret)
        cfg.model_name = "opencode-go/glm-5.3-flash"
        cfg.max_tokens = 64
        cfg.temperature = 0.3
        cfg.is_enabled = True
        cfg.extra_headers = {
            "x-opencode-session": "ses_demo_1",
            "User-Agent": "stock-analysis/1.0",
        }
        assert register_model_config(gw, cfg, secret) is True
        loaded = gw.get_config("zen-go-1")
        assert loaded.extra_headers["x-opencode-session"] == "ses_demo_1"

        seen = {}

        async def _fake(**kwargs):
            seen.update(kwargs)
            return _completion(kwargs["model"])

        with patch.object(litellm, "acompletion", side_effect=_fake):
            resp = asyncio.run(gw._chat_one(
                loaded, [{"role": "user", "content": "hi"}], None, 10, False,
            ))
        assert resp.success is True
        assert seen["model"] == "openai/glm-5.3-flash"
        assert seen["extra_headers"]["x-opencode-session"] == "ses_demo_1"
        assert seen["api_base"] == "https://opencode.ai/inference/go/openai/v1"


# ===================================================================
# 请求头归一化 / 加载容错 / 模型目录过滤
# ===================================================================
def _fake_httpx_client(payload, seen):
    """构造假的 httpx.AsyncClient，避免测试真的打上游。"""
    class _Resp:
        status_code = 200

        def json(self):
            return payload

    class _Client:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def get(self, url, headers=None):
            seen["url"] = url
            return _Resp()

    return _Client


class TestExtraHeadersSafety:
    def test_non_mapping_is_ignored(self):
        assert normalize_extra_headers(None) == {}
        assert normalize_extra_headers(["a", "b"]) == {}
        assert normalize_extra_headers("x-opencode-session: ses_1") == {}

    def test_reserved_headers_are_dropped(self):
        headers = normalize_extra_headers({
            "Authorization": "Bearer evil",
            "content-type": "application/json",
            "x-opencode-session": "ses_1",
        })
        assert headers == {"x-opencode-session": "ses_1"}

    def test_values_are_stringified_and_empties_skipped(self):
        headers = normalize_extra_headers({
            "Retry": 3, "Empty": "", "None": None, "UA": " app/1.0 ",
        })
        assert headers == {"Retry": "3", "UA": "app/1.0"}

    def test_bad_headers_do_not_break_registration(self):
        """DB 里 extra_headers 被写成数组时不能抛，否则会拖垮整个网关装载。"""
        secret = "test-secret"
        gw = ModelGateway(secret)
        cfg = MagicMock()
        cfg.id = "bad-1"
        cfg.name = "Bad Headers"
        cfg.platform = "opencodezen"
        cfg.base_url = "https://opencode.ai/inference/openai/v1"
        cfg.api_key_encrypted = encrypt_api_key("oc_sk_realkey", secret)
        cfg.model_name = "glm-5.3-flash"
        cfg.max_tokens = 64
        cfg.temperature = 0.2
        cfg.is_enabled = True
        cfg.extra_headers = ["x-opencode-session", "ses_1"]
        assert register_model_config(gw, cfg, secret) is True
        assert gw.get_config("bad-1").extra_headers == {}

    def test_prefix_match_is_case_insensitive(self):
        assert normalize_opencode_model("OpenCode/glm-5.3-flash") == "glm-5.3-flash"
        assert normalize_opencode_model("OPENCODE-GO/kimi-k3") == "kimi-k3"

    def test_long_random_key_with_your_is_not_placeholder(self):
        long_key = "sk-or-v1-" + "your" + "a" * 62
        assert len(long_key) > 48
        assert detect_key_mismatch("openrouter", long_key) is None

    def test_short_example_key_is_still_flagged(self):
        warning = detect_key_mismatch("openai", "sk-your-openai-api-key")
        assert warning and "占位符" in warning


class TestAvailableModelsFiltering:
    def test_free_and_non_chat_models_are_excluded(self):
        seen = {}
        payload = {"data": [
            {"id": "glm-5.3-flash", "type": "chat"},
            {"id": "glm-5.3-flash-free", "type": "chat"},
            {"id": "big-pickle", "type": "chat"},
            {"id": "some-embedding-model", "type": "embedding"},
        ]}
        with patch("app.api.models.httpx.AsyncClient", _fake_httpx_client(payload, seen)):
            result = asyncio.run(fetch_available_models(
                AvailableModelsRequest(platform="opencodezen"), None,
            ))
        assert result["models"] == ["glm-5.3-flash"]
        assert set(result["blocked_models"]) == {"glm-5.3-flash-free", "big-pickle"}
        assert seen["url"] == "https://opencode.ai/inference/v1/models"

    def test_stored_go_base_url_wins_over_default(self):
        """编辑 Go 配置且未传 base_url 时，必须拉 Go 目录而非 pay-as-you-go。"""
        secret = get_settings().ENCRYPTION_KEY
        stored = MagicMock()
        stored.platform = "opencodezen"
        stored.base_url = "https://opencode.ai/inference/go/openai/v1"
        stored.api_key_encrypted = encrypt_api_key("oc_sk_realkey", secret)
        db_result = MagicMock()
        db_result.scalar_one_or_none.return_value = stored
        db = MagicMock()
        db.execute = AsyncMock(return_value=db_result)

        seen = {}
        payload = {"data": [{"id": "glm-5.3-flash", "type": "chat"}]}
        with patch("app.api.models.httpx.AsyncClient", _fake_httpx_client(payload, seen)):
            asyncio.run(fetch_available_models(
                AvailableModelsRequest(platform="opencodezen", config_id="cfg-1"), db,
            ))
        assert seen["url"] == "https://opencode.ai/inference/go/v1/models"
