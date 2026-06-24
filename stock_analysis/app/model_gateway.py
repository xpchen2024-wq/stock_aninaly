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
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stream: bool = False,
    ) -> LLMResponse:
        """Send chat completion request to configured model."""
        config = self.get_config(config_id)
        if not config:
            return LLMResponse(
                content="", model="unknown",
                success=False, error="No model configured"
            )
        if not config.is_enabled:
            return LLMResponse(
                content="", model=config.model_name,
                success=False, error=f"Model {config.name} is disabled"
            )

        start_time = time.time()

        try:
            import litellm

            litellm.api_key = config.api_key
            if config.base_url:
                litellm.api_base = config.base_url

            model_str = f"{config.provider.value}/{config.model_name}"

            # Handle each provider
            if config.provider == LLMProvider.OPENAI:
                model_str = config.model_name
                litellm.api_key = config.api_key
                if config.base_url:
                    litellm.api_base = config.base_url
            elif config.provider == LLMProvider.DEEPSEEK:
                model_str = f"deepseek/{config.model_name}"
                litellm.api_key = config.api_key
                if config.base_url:
                    litellm.api_base = config.base_url
            elif config.provider == LLMProvider.OPENROUTER:
                model_str = f"openrouter/{config.model_name}"
                litellm.api_key = config.api_key
                if config.base_url:
                    litellm.api_base = config.base_url
            elif config.provider == LLMProvider.OPENCODEZEN:
                model_str = f"openai/{config.model_name}"
                litellm.api_key = config.api_key
                if config.base_url:
                    litellm.api_base = config.base_url
            elif config.provider == LLMProvider.CUSTOM:
                model_str = f"openai/{config.model_name}"
                litellm.api_key = config.api_key
                if config.base_url:
                    litellm.api_base = config.base_url

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
