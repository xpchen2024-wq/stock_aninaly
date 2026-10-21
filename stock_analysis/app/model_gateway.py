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


@dataclass
class LLMResponse:
    """Standardized LLM response."""
    content: str
    model: str
    tokens_used: int = 0
    latency_ms: float = 0.0
    success: bool = True
    error: str = ""


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
        if any(k in low for k in ("401", "api_key", "authentication", "expired", "invalid key")):
            return 1800.0          # 鉴权问题基本不会自愈
        if any(k in low for k in ("429", "rate limit", "rate_limit", "quota")):
            return 600.0           # 限流：等配额重置
        if any(k in low for k in ("unavailable", "not found", "403", "region", "not supported")):
            return 1800.0          # 模型不可用/地区限制
        return 120.0

    def _mark_failure(self, config_id: str, error_text: str) -> None:
        info = self._health.setdefault(config_id, {"failures": 0, "cooldown_until": 0.0, "reason": ""})
        info["failures"] += 1
        info["reason"] = (error_text or "")[:200]
        info["cooldown_until"] = time.time() + self._cooldown_seconds(error_text)
        logger.warning(
            f"Model '{config_id}' marked unhealthy "
            f"(failures={info['failures']}, cooldown={self._cooldown_seconds(error_text):.0f}s): "
            f"{info['reason'][:120]}"
        )

    def _mark_success(self, config_id: str) -> None:
        info = self._health.get(config_id)
        if info and info.get("failures"):
            logger.info(f"Model '{config_id}' recovered after {info['failures']} failure(s)")
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
    ) -> LLMResponse:
        """Send chat completion request, with automatic failover across models.

        指定 config_id 时只尝试该模型；否则按「默认模型 → 其余启用模型」的顺序尝试，
        失败（限流/模型下线/鉴权失败/超时）的模型会被短暂冷却并自动切换到下一个，
        避免单模型不可用导致整个多智能体工作流失败。

        prefer_config_id：把该模型排在候选首位（例如优先用免费模型省额度），
        但它失败后仍会继续走故障转移链，不会因为免费额度耗尽而直接失败。
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
            response = await self._chat_one(config, messages, temperature, max_tokens, stream)
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
    ) -> LLMResponse:
        """Send a single chat completion request with one model."""
        if not config.is_enabled:
            return LLMResponse(
                content="", model=config.model_name,
                success=False, error=f"Model {config.name} is disabled",
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
            elif config.provider in (LLMProvider.OPENCODEZEN, LLMProvider.CUSTOM):
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

            response = await litellm.acompletion(**kwargs)

            elapsed = (time.time() - start_time) * 1000
            content = response.choices[0].message.content or ""
            tokens = response.usage.total_tokens if response.usage else 0

            logger.info(f"LLM call: model={model_str}, tokens={tokens}, latency={elapsed:.0f}ms")
            return LLMResponse(
                content=content,
                model=config.model_name,
                tokens_used=tokens,
                latency_ms=elapsed,
                success=True,
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
    ))
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
            register_model_config(gateway, cfg, settings.ENCRYPTION_KEY)
            loaded += 1
            if cfg.is_default:
                default_id = str(cfg.id)

    if default_id:
        gateway.set_default(default_id)
        logger.info(f"Default model set to {default_id}")
    logger.info(f"Loaded {loaded} model config(s) into gateway")
    return loaded
