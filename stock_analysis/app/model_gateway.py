# ============================================================================
# AI Stock Analysis Platform - AI Model Gateway (LiteLLM Wrapper)
# ============================================================================
from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import logging
import time
from base64 import b64encode, b64decode
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Optional, Dict, Any, List, AsyncIterator

from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.backends import default_backend

logger = logging.getLogger(__name__)


# -- Encryption Utilities -----------------------------------------------------
def _derive_key(secret: str, salt: bytes = b"aistock_salt") -> bytes:
    """Derive AES-256 key from secret."""
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=100000,
        backend=default_backend(),
    )
    return b64encode(kdf.derive(secret.encode()))


def encrypt_api_key(api_key: str, secret: str) -> str:
    """Encrypt API key using AES-256."""
    key = _derive_key(secret)
    f = Fernet(key)
    return f.encrypt(api_key.encode()).decode()


def decrypt_api_key(encrypted: str, secret: str) -> str:
    """Decrypt API key using AES-256."""
    key = _derive_key(secret)
    f = Fernet(key)
    return f.decrypt(encrypted.encode()).decode()


def mask_api_key(key: str) -> str:
    """Mask API key for display: sk-****xxxx."""
    if len(key) <= 8:
        return "****"
    return key[:4] + "****" + key[-4:]


# -- API Key / Platform Consistency -------------------------------------------
# 各平台 Key 的前缀是稳定的，可以据此在「调用之前」发现填错平台的 Key，
# 避免用户对着 401 / Invalid credential 反复改模型名（最常见的误配置就是
# 把 OpenRouter 的 sk-or-v1-… 填进 OpenCode Zen 配置）。
PLATFORM_LABELS = {
    "openai": "OpenAI",
    "deepseek": "DeepSeek",
    "openrouter": "OpenRouter",
    "opencodezen": "OpenCode Zen",
    "custom": "自定义服务",
    "anthropic": "Anthropic",
    "groq": "Groq",
    "xai": "xAI",
}

_KEY_PREFIX_OWNERS = (
    ("sk-or-v1-", "openrouter"),
    ("sk-ant-", "anthropic"),
    ("sk-proj-", "openai"),
    ("gsk_", "groq"),
    ("xai-", "xai"),
    # OpenCode Console 个人 API Key（v2 文档 console/harnesses#get-an-api-key）
    ("oc_sk_", "opencodezen"),
)

# 这些头由 LiteLLM / httpx 自行生成，配置里覆盖会破坏鉴权或请求签名
_RESERVED_HEADERS = frozenset({
    "authorization", "content-type", "content-length",
    "host", "accept-encoding", "connection",
})


def _looks_like_placeholder(key: str) -> bool:
    """判断是否为示例占位 Key。

    真实凭据是长随机串（OpenRouter 73 位，OpenCode ``oc_sk_…`` 也偏长），
    占位符则是可读短语。因此 ``your`` 只在短 Key 里视为占位，避免长随机串
    偶然包含这四个字母就被误判；``xxxx`` / ``placeholder`` / ``changeme``
    不可能出现在真实 Key 里，直接判定。
    """
    low = key.lower()
    if "xxxx" in low or "placeholder" in low or "changeme" in low:
        return True
    return "your" in low and len(key) <= 48


def normalize_extra_headers(raw: Any) -> Dict[str, str]:
    """把自定义请求头归一化成安全的 ``str -> str`` 字典。

    数据来源可能是 DB 的 JSON 列（历史数据可能写成数组/字符串）或 API 入参，
    因此一律容错：非映射返回空字典，跳过保留头、空键与空值，值统一转字符串。
    OpenCode Go 依赖 ``x-opencode-session`` 与自定义 ``User-Agent``；过滤保留头
    是为了避免配置里塞的 Authorization 覆盖由 api_key 生成的鉴权头。
    """
    if not isinstance(raw, dict):
        return {}
    headers: Dict[str, str] = {}
    for key, value in raw.items():
        if not isinstance(key, str) or not key.strip():
            continue
        name = key.strip()
        if name.lower() in _RESERVED_HEADERS:
            logger.warning(f"Ignoring reserved header in extra_headers: {name}")
            continue
        if value is None:
            continue
        text = str(value).strip()
        if text:
            headers[name] = text
    return headers


def detect_key_mismatch(platform: str, api_key: str) -> Optional[str]:
    """检查 API Key 是否与所选平台匹配，返回中文警告；正常时返回 None。"""
    key = (api_key or "").strip()
    if not key or platform == "custom":
        return None

    for prefix, owner in _KEY_PREFIX_OWNERS:
        if key.startswith(prefix) and owner != platform:
            return (
                f"Key 归属异常：该 Key 属于 {PLATFORM_LABELS.get(owner, owner)}（{prefix}…），"
                f"与所选平台 {PLATFORM_LABELS.get(platform, platform)} 不匹配，调用必然失败"
            )

    # OpenRouter 的 Key 有强前缀要求，识别不出前缀即可判定填错
    if platform == "openrouter" and not key.startswith("sk-or-v1-"):
        return "Key 格式异常：OpenRouter 的 Key 以 sk-or-v1- 开头，当前 Key 不符合该格式"

    if _looks_like_placeholder(key):
        if platform == "opencodezen":
            return (
                "Key 格式异常：当前是示例占位符。请到 https://opencode.ai/console → "
                "Account → Personal API keys 生成真实 Key（形如 oc_sk_…）"
            )
        return "Key 格式异常：当前填写的是示例占位符，请替换为该平台的真实 API Key"

    return None


def normalize_opencode_model(model_name: str) -> str:
    """去掉只用于 OpenCode 客户端配置的模型前缀。

    官方 v2 文档：直接调用 API 时模型 ID 不带 ``opencode/`` 前缀；Go 订阅在
    客户端里写作 ``opencode-go/<id>``，走 API 时同样要去掉。数据库里可能残留
    带前缀的旧配置，这里统一清洗，避免上报「模型不存在」。
    前缀按大小写不敏感匹配（用户可能手输 ``OpenCode/xxx``），但模型 ID 本身
    大小写敏感，返回时保留原样。
    """
    name = (model_name or "").strip()
    low = name.lower()
    for prefix in ("opencode-go/", "opencode/"):
        if low.startswith(prefix):
            return name[len(prefix):]
    return name


def _rejects_param(error_text: str, param: str) -> bool:
    """判断 provider 报错是否因为「不认识某个请求参数」。

    这类错误（免费模型、自建 one-api 网关常见）换个写法就能过，
    不应该当作模型不可用处理。
    """
    low = (error_text or "").lower()
    if param.lower() not in low:
        return False
    return any(k in low for k in (
        "unsupported", "not support", "does not support", "unknown", "unrecognized",
        "invalid", "unexpected", "extra fields", "not allowed", "400",
    ))


def _is_bad_request(error_text: str) -> bool:
    low = (error_text or "").lower()
    return "400" in low or "badrequest" in low or "bad request" in low


def _rejects_max_tokens(error_text: str) -> bool:
    """模型拒绝过大的 max_tokens（部分免费模型只允许 1k/2k 输出）。"""
    low = (error_text or "").lower()
    if "max_tokens" not in low and "max tokens" not in low:
        return False
    return any(k in low for k in ("too large", "exceed", "maximum", "greater than",
                                  "invalid", "less than or equal"))


# -- Model Gateway ------------------------------------------------------------
class LLMProvider(str, Enum):
    OPENAI = "openai"
    DEEPSEEK = "deepseek"
    OPENROUTER = "openrouter"
    OPENCODEZEN = "opencodezen"
    CUSTOM = "custom"


@dataclass
class LLMConfig:
    """Configuration for a single LLM model."""
    id: str = ""
    name: str = ""
    provider: LLMProvider = LLMProvider.OPENAI
    base_url: str = ""
    api_key: str = ""
    model_name: str = "gpt-4o-mini"
    max_tokens: int = 4096
    temperature: float = 0.7
    is_enabled: bool = True
    # 自定义请求头。OpenCode Go 订阅要求 x-opencode-session 与自定义 User-Agent，
    # 由「模型配置管理」里的 extra_headers 字段写入，透传给 LiteLLM。
    extra_headers: Dict[str, str] = field(default_factory=dict)


@dataclass
class LLMResponse:
    """Standardized LLM response."""
    content: str
    model: str
    tokens_used: int = 0
    latency_ms: float = 0.0
    success: bool = True
    error: str = ""
    # 结束原因（stop / length / content_filter …）。
    # ``length`` 是「输出被 max_tokens 截断」的唯一可靠信号——推理型模型会把
    # 思维链写进 content，一旦截断就永远拿不到 JSON，调用方必须能分辨。
    finish_reason: str = ""
    # 实际命中的配置 ID（故障转移后可能与首选不同），供调用方做「避开某模型重试」
    config_id: str = ""


class ModelGateway:
    """
    Unified AI model gateway.
    Routes requests to configured LLM providers via LiteLLM.
    Supports OpenAI, DeepSeek, OpenRouter, OpenCode Zen, and custom providers.
    """

    def __init__(self, encryption_secret: str):
        self._encryption_secret = encryption_secret
        self._configs: Dict[str, LLMConfig] = {}
        self._default_id: Optional[str] = None
        # 故障转移健康状态: {config_id: {"failures": n, "cooldown_until": ts, "reason": str}}
        self._health: Dict[str, Dict[str, Any]] = {}

    # -- Failover helpers ----------------------------------------------------
    def _cooldown_remaining(self, config_id: str) -> float:
        info = self._health.get(config_id)
        if not info:
            return 0.0
        remaining = info.get("cooldown_until", 0) - time.time()
        return remaining if remaining > 0 else 0.0

    @staticmethod
    def _cooldown_seconds(error_text: str) -> float:
        """根据错误类型决定冷却时长（秒）。"""
        low = (error_text or "").lower()
        if any(k in low for k in ("401", "api_key", "authentication", "expired", "invalid key",
                                  "key 归属异常", "key 格式异常")):
            return 1800.0          # 鉴权/配置错配不会自愈
        if any(k in low for k in ("429", "rate limit", "rate_limit", "quota")):
            return 600.0           # 限流：等配额重置
        if any(k in low for k in ("unavailable", "not found", "403", "region", "not supported")):
            return 1800.0          # 模型不可用/地区限制
        return 120.0

    @staticmethod
    def _quota_reset_cooldown(error_text: str) -> float:
        """配额类错误按「真实重置时刻」冷却，而不是固定 10 分钟。

        OpenRouter 的 429 会带 ``X-RateLimit-Reset``（当日 00:00 UTC）。若只冷却
        600 秒，之后每隔 10 分钟就会再撞一次同样的 429，多智能体工作流会被白白
        拖慢；这里直接用重置时间戳算出剩余秒数，最长封顶 24 小时。
        """
        try:
            from app.utils.errors import parse_rate_limit_reset
        except Exception:      # pragma: no cover - 理论不会发生
            return 0.0
        reset_at = parse_rate_limit_reset(error_text)
        if not reset_at:
            return 0.0
        remaining = (reset_at - datetime.now(timezone.utc)).total_seconds()
        return min(remaining, 86400.0) if remaining > 0 else 0.0

    def _mark_failure(self, config_id: str, error_text: str) -> None:
        info = self._health.setdefault(config_id, {"failures": 0, "cooldown_until": 0.0, "reason": ""})
        cooldown = max(
            self._cooldown_seconds(error_text),
            self._quota_reset_cooldown(error_text),
        )
        info["failures"] += 1
        info["reason"] = (error_text or "")[:200]
        info["cooldown_until"] = time.time() + cooldown
        logger.warning(
            f"Model '{config_id}' marked unhealthy "
            f"(failures={info['failures']}, cooldown={cooldown:.0f}s): "
            f"{info['reason'][:120]}"
        )

    def _mark_success(self, config_id: str) -> None:
        info = self._health.get(config_id)
        if info and info.get("failures"):
            logger.info(f"Model '{config_id}' recovered after {info['failures']} failure(s)")
        self._health.pop(config_id, None)

    def reset_health(self, config_id: str) -> None:
        """Clear cooldown state after a config is updated (e.g. model replaced)."""
        self._health.pop(config_id, None)

    def _candidate_configs(self, config_id: Optional[str]) -> List[LLMConfig]:
        """按优先级返回可用模型候选（跳过冷却中的模型）。"""
        if config_id:
            cfg = self._configs.get(config_id)
            if cfg and cfg.is_enabled:
                return [cfg]
            return []

        candidates: List[LLMConfig] = []
        default = self._configs.get(self._default_id) if self._default_id else None
        if default and default.is_enabled:
            candidates.append(default)
        for cfg in self._configs.values():
            if cfg.is_enabled and (default is None or cfg.id != default.id):
                candidates.append(cfg)
        return [c for c in candidates if self._cooldown_remaining(c.id) <= 0]

    def add_config(self, config: LLMConfig):
        """Register a model configuration."""
        self._configs[config.id] = config
        if not self._default_id:
            self._default_id = config.id

    def set_default(self, config_id: str):
        """Set default model."""
        if config_id in self._configs:
            self._default_id = config_id

    def get_config(self, config_id: Optional[str] = None) -> Optional[LLMConfig]:
        """Get model configuration by ID, or default."""
        cid = config_id or self._default_id
        return self._configs.get(cid) if cid else None

    def list_configs(self) -> List[LLMConfig]:
        return list(self._configs.values())

    def remove_config(self, config_id: str):
        self._configs.pop(config_id, None)
        if self._default_id == config_id:
            self._default_id = next(iter(self._configs), None)

    async def chat(
        self,
        messages: List[Dict[str, str]],
        config_id: Optional[str] = None,
        prefer_config_id: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stream: bool = False,
        response_format: Optional[Dict[str, Any]] = None,
        exclude_config_ids: Optional[List[str]] = None,
    ) -> LLMResponse:
        """Send chat completion request, with automatic failover across models.

        指定 config_id 时只尝试该模型；否则按「默认模型 → 其余启用模型」的顺序尝试，
        失败（限流/模型下线/鉴权失败/超时）的模型会被短暂冷却并自动切换到下一个，
        避免单模型不可用导致整个多智能体工作流失败。

        prefer_config_id：把该模型排在候选首位（例如优先用免费模型省额度），
        但它失败后仍会继续走故障转移链，不会因为免费额度耗尽而直接失败。

        response_format：结构化输出约束（如 ``{"type": "json_object"}``）。
        仅对支持的模型生效；不支持时 ``_chat_one`` 会自动去掉该参数重试一次，
        因此调用方可以放心传，不必为每家 provider 做能力探测。

        exclude_config_ids：本次跳过这些配置。用于「上一个模型把输出写成了
        思维链、注定给不出 JSON」时换模型重试；若排除后没有候选，则忽略排除
        （宁可重试同一模型，也不能因为排除把自己逼成「无模型可用」）。
        仅在未显式指定 ``config_id`` 时生效——显式指定意味着调用方就要这个模型。
        """
        if config_id:
            candidates = self._candidate_configs(config_id)
        else:
            candidates = self._candidate_configs(None)
            if prefer_config_id:
                pref = self._configs.get(prefer_config_id)
                if (pref and pref.is_enabled
                        and self._cooldown_remaining(pref.id) <= 0):
                    candidates = [pref] + [c for c in candidates if c.id != pref.id]
            if exclude_config_ids:
                kept = [c for c in candidates if c.id not in exclude_config_ids]
                if kept:
                    candidates = kept
        if not candidates:
            if config_id:
                return LLMResponse(
                    content="", model=config_id,
                    success=False, error=f"Model {config_id} not available (disabled or cooling down)",
                )
            return LLMResponse(
                content="", model="unknown",
                success=False, error="No usable model configured (all disabled or cooling down)",
            )

        last_error = ""
        last_model = candidates[0].model_name
        for index, config in enumerate(candidates):
            response = await self._chat_one(
                config, messages, temperature, max_tokens, stream,
                response_format=response_format,
            )
            if response.success:
                self._mark_success(config.id)
                if index > 0:
                    logger.info(f"LLM failover succeeded with model {config.model_name}")
                return response

            last_error = response.error
            last_model = config.model_name
            self._mark_failure(config.id, response.error)
            if index < len(candidates) - 1:
                logger.warning(
                    f"LLM failover: model {config.model_name} failed "
                    f"({(response.error or '')[:120]}), trying next candidate"
                )

        return LLMResponse(
            content="", model=last_model, tokens_used=0, latency_ms=0,
            success=False,
            error=f"All {len(candidates)} model(s) failed. Last error: {last_error}",
        )

    async def _chat_one(
        self,
        config: LLMConfig,
        messages: List[Dict[str, str]],
        temperature: Optional[float],
        max_tokens: Optional[int],
        stream: bool,
        response_format: Optional[Dict[str, Any]] = None,
    ) -> LLMResponse:
        """Send a single chat completion request with one model."""
        if not config.is_enabled:
            return LLMResponse(
                content="", model=config.model_name,
                success=False, error=f"Model {config.name} is disabled",
            )

        # Key 与平台不匹配时直接短路：这类配置永远不会成功，
        # 让故障转移立刻跳到下一个候选，不必等一次网络往返（最长 60s）。
        mismatch = detect_key_mismatch(config.provider.value, config.api_key)
        if mismatch:
            return LLMResponse(
                content="", model=config.model_name, success=False, error=mismatch,
            )

        start_time = time.time()

        try:
            import litellm

            # Concurrency: never assign litellm.api_key / litellm.api_base globally;
            # per-call kwargs below are the only credential channel.
            if config.provider == LLMProvider.OPENAI:
                model_str = config.model_name
            elif config.provider == LLMProvider.DEEPSEEK:
                model_str = f"deepseek/{config.model_name}"
            elif config.provider == LLMProvider.OPENROUTER:
                model_str = f"openrouter/{config.model_name}"
            elif config.provider == LLMProvider.OPENCODEZEN:
                # 官方 API 只接受不带 opencode/ 前缀的模型 ID
                model_str = f"openai/{normalize_opencode_model(config.model_name)}"
            elif config.provider == LLMProvider.CUSTOM:
                model_str = f"openai/{config.model_name}"
            else:
                model_str = config.model_name

            kwargs = {
                "model": model_str,
                "messages": messages,
                "temperature": temperature or config.temperature,
                "max_tokens": max_tokens or config.max_tokens,
                "api_key": config.api_key,
            }
            if config.base_url:
                kwargs["api_base"] = config.base_url
            # Go 订阅要求 x-opencode-session + 自定义 User-Agent；这些头由配置里的
            # extra_headers 透传（LiteLLM 原生支持 extra_headers 参数）。
            # 归一化在这里做：DB 的 JSON 列可能被写成数组/字符串，直接 dict() 会抛。
            safe_headers = normalize_extra_headers(config.extra_headers)
            if safe_headers:
                kwargs["extra_headers"] = safe_headers
            if response_format:
                kwargs["response_format"] = response_format

            # 两条「降配重试」路径：都只在请求参数层面出错时才走，
            # 属于可自愈的配置差异，不应升级成「模型不可用」：
            #  1) 部分免费/自建模型不支持 response_format，直接报 400；
            #  2) 部分模型限制输出长度，max_tokens 要得太大也报 400。
            # 用有界循环（最多 3 次请求）是为了让两种降配可以叠加生效，
            # 否则「同时不支持 json 模式 + 输出上限 1k」的模型会直接判死。
            attempts = 0
            while True:
                attempts += 1
                try:
                    response = await litellm.acompletion(**kwargs)
                    break
                except Exception as inner:
                    msg = str(inner)
                    if attempts >= 3:
                        raise
                    if "response_format" in kwargs and _rejects_param(msg, "response_format"):
                        logger.info(
                            f"Model {config.model_name} rejected response_format "
                            f"({msg[:120]}), retrying without it"
                        )
                        kwargs.pop("response_format", None)
                        continue
                    if _rejects_max_tokens(msg):
                        current = int(kwargs.get("max_tokens") or 1024)
                        # 逐次折半（最多降到 256），不猜具体上限：宁可多试一次，
                        # 也不要把一个本来能出 2k 的模型压到 1k 冤枉截断
                        smaller = max(256, current // 2)
                        if smaller < current:
                            logger.info(
                                f"Model {config.model_name} rejected max_tokens="
                                f"{current} ({msg[:120]}), retrying with {smaller}"
                            )
                            kwargs["max_tokens"] = smaller
                            continue
                    if "response_format" in kwargs and _is_bad_request(msg):
                        # 报错没点名参数、但是 400：我们额外加的参数只有
                        # response_format，先去掉它再试一次
                        logger.info(
                            f"Model {config.model_name} returned 400 with "
                            f"response_format ({msg[:120]}), retrying without it"
                        )
                        kwargs.pop("response_format", None)
                        continue
                    raise

            elapsed = (time.time() - start_time) * 1000
            choice = response.choices[0]
            content = choice.message.content or ""
            finish_reason = getattr(choice, "finish_reason", "") or ""
            tokens = response.usage.total_tokens if response.usage else 0

            if finish_reason == "length":
                logger.warning(
                    f"LLM output truncated by max_tokens: model={model_str}, "
                    f"max_tokens={kwargs.get('max_tokens')}, tokens={tokens} "
                    f"（推理型模型会把思维链写进 content，预算不足时拿不到最终答案）"
                )
            logger.info(
                f"LLM call: model={model_str}, tokens={tokens}, "
                f"finish_reason={finish_reason or 'unknown'}, latency={elapsed:.0f}ms"
            )
            return LLMResponse(
                content=content,
                model=config.model_name,
                tokens_used=tokens,
                latency_ms=elapsed,
                success=True,
                finish_reason=finish_reason,
                config_id=config.id,
            )

        except Exception as e:
            elapsed = (time.time() - start_time) * 1000
            logger.error(f"LLM call failed: model={config.model_name}, error={e}, latency={elapsed:.0f}ms")
            return LLMResponse(
                content="", model=config.model_name,
                tokens_used=0, latency_ms=elapsed,
                success=False, error=str(e),
            )

    async def probe(self, config: LLMConfig) -> Dict[str, Any]:
        """对单个配置做一次探测调用（不注册到网关），用于「测试连接」。

        与 chat() 的区别：
        - 忽略 is_enabled 停用标记，这样已停用的配置也能暴露出真实失败原因
          （例如 API Key 过期），而不是只返回「已停用」；
        - 不触发故障转移，也不会写入网关健康状态。
        """
        probe_config = LLMConfig(
            id=config.id, name=config.name, provider=config.provider,
            base_url=config.base_url, api_key=config.api_key,
            model_name=config.model_name, max_tokens=10, temperature=0.0,
            is_enabled=True,
            extra_headers=normalize_extra_headers(config.extra_headers),
        )
        response = await self._chat_one(
            probe_config,
            [{"role": "user", "content": "Hello, respond with 'OK' only."}],
            None, 10, False,
        )
        return {
            "success": response.success,
            "latency_ms": round(response.latency_ms, 1),
            "model": response.model,
            "error": response.error,
        }

    async def test_connection(self, config_id: str) -> Dict[str, Any]:
        """Test model connectivity."""
        config = self.get_config(config_id)
        if not config:
            logger.warning(f"test_connection: config not found id={config_id}")
            return {"success": False, "error": "Config not found"}

        logger.info(f"test_connection: model={config.model_name}, platform={config.provider.value}")
        start = time.time()
        try:
            response = await self.chat(
                messages=[{"role": "user", "content": "Hello, respond with 'OK' only."}],
                config_id=config_id,
                max_tokens=10,
            )
            latency = (time.time() - start) * 1000
            logger.info(
                f"test_connection result: model={config.model_name}, "
                f"success={response.success}, latency={latency:.0f}ms"
            )
            return {
                "success": response.success,
                "latency_ms": round(latency, 1),
                "model": config.model_name,
                "error": response.error if not response.success else None,
            }
        except Exception as e:
            latency = (time.time() - start) * 1000
            return {
                "success": False,
                "latency_ms": round(latency, 1),
                "model": config.model_name,
                "error": str(e),
            }


# -- Global Gateway Instance --------------------------------------------------
_model_gateway: Optional[ModelGateway] = None


def get_model_gateway(secret: Optional[str] = None) -> ModelGateway:
    global _model_gateway
    if _model_gateway is None:
        from app.config import get_settings
        settings = get_settings()
        _model_gateway = ModelGateway(
            encryption_secret=secret or settings.ENCRYPTION_KEY
        )
    return _model_gateway


def register_model_config(gateway: ModelGateway, cfg: Any, secret: str) -> bool:
    """把一条数据库模型配置注册到网关。返回是否注册成功。"""
    try:
        api_key = decrypt_api_key(cfg.api_key_encrypted, secret)
    except Exception as exc:
        logger.warning(
            f"Failed to decrypt API key for model '{cfg.name}' (id={cfg.id}): {exc}"
        )
        api_key = ""

    try:
        provider = LLMProvider(cfg.platform)
    except ValueError:
        provider = LLMProvider.CUSTOM

    gateway.add_config(LLMConfig(
        id=str(cfg.id),
        name=cfg.name,
        provider=provider,
        base_url=cfg.base_url or "",
        api_key=api_key,
        model_name=cfg.model_name,
        max_tokens=cfg.max_tokens or 4096,
        temperature=cfg.temperature if cfg.temperature is not None else 0.7,
        is_enabled=cfg.is_enabled,
        extra_headers=normalize_extra_headers(getattr(cfg, "extra_headers", None)),
    ))
    gateway.reset_health(str(cfg.id))
    return True


async def load_configs_from_db() -> int:
    """从数据库把启用的模型配置装载到全局网关。

    此前只有 ``POST /api/v1/models/{id}/test`` 会注册配置，导致分析引擎、Agent 与
    Celery 任务通过 ``get_model_gateway()`` 拿到的是空网关（0 个模型），AI 能力
    全部不可用。应用与 Worker 启动时调用本函数完成装载。

    这里使用 task_session（NullPool）访问数据库，因此同时适用于 API 进程
    （单一事件循环）与 Celery 任务（每次新建事件循环）两种场景。
    """
    from sqlalchemy import select
    from app.database import task_session
    from app.models import ModelConfig
    from app.config import get_settings

    settings = get_settings()
    gateway = get_model_gateway()
    loaded = 0
    default_id: Optional[str] = None

    async with task_session() as db:
        result = await db.execute(
            select(ModelConfig).where(
                ModelConfig.status == "active",
                ModelConfig.is_enabled == True,
            )
        )
        for cfg in result.scalars().all():
            # 单条坏配置（例如 extra_headers 被写成数组）不能拖垮整个装载：
            # 以前这里没有 try，一条脏数据会让网关 0 个模型，全平台 AI 静默失效。
            try:
                register_model_config(gateway, cfg, settings.ENCRYPTION_KEY)
            except Exception as exc:
                logger.error(
                    f"Skipping unloadable model config id={getattr(cfg, 'id', '?')} "
                    f"name={getattr(cfg, 'name', '?')}: {exc}",
                    exc_info=True,
                )
                continue
            loaded += 1
            if cfg.is_default:
                default_id = str(cfg.id)

    if default_id:
        gateway.set_default(default_id)
        logger.info(f"Default model set to {default_id}")
    logger.info(f"Loaded {loaded} model config(s) into gateway")
    return loaded
