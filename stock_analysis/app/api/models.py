# ============================================================================
# AI Stock Analysis Platform - AI Model Config API (AI-001 ~ AI-007)
# ============================================================================
from __future__ import annotations

import logging
from typing import Optional, List
from uuid import uuid4

import httpx
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update

from app.database import get_db
from app.config import get_settings
from app.models import ModelConfig, AgentModelBinding
from app.model_gateway import (
    ModelGateway, LLMConfig, LLMProvider,
    encrypt_api_key, decrypt_api_key, mask_api_key,
    get_model_gateway, detect_key_mismatch, normalize_opencode_model,
)
from app.utils.errors import explain_llm_error_with_hint

logger = logging.getLogger(__name__)
router = APIRouter()
settings = get_settings()


# -- Schemas --
class ModelConfigCreate(BaseModel):
    name: str
    platform: str  # openai / deepseek / openrouter / custom
    base_url: Optional[str] = None
    api_key: str
    model_name: str
    max_tokens: int = 4096
    temperature: float = 0.7
    is_default: bool = False
    is_enabled: bool = True
    extra_headers: Optional[dict] = None


class ModelConfigUpdate(BaseModel):
    name: Optional[str] = None
    platform: Optional[str] = None
    base_url: Optional[str] = None
    api_key: Optional[str] = None
    model_name: Optional[str] = None
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    is_default: Optional[bool] = None
    is_enabled: Optional[bool] = None
    extra_headers: Optional[dict] = None


class ModelConfigResponse(BaseModel):
    id: str
    name: str
    platform: str
    base_url: Optional[str]
    api_key_masked: str
    model_name: str
    max_tokens: int
    temperature: float
    is_default: bool
    is_enabled: bool
    status: str = "active"
    # Key 与平台不匹配时给出中文警告，界面直接展示（避免用户对着 401 反复改模型名）
    key_warning: Optional[str] = None
    # 自定义请求头（OpenCode Go 的 x-opencode-session 等），编辑弹窗需要回显
    extra_headers: Optional[dict] = None


class TestResult(BaseModel):
    success: bool
    latency_ms: float
    model: str
    error: Optional[str] = None
    raw_error: Optional[str] = None      # 原始英文错误，便于排查
    hint: Optional[str] = None            # 可执行建议


class BindingCreate(BaseModel):
    model_config_id: str
    agent_id: Optional[str] = None
    scene: Optional[str] = None


# -- Routes --
@router.get("", response_model=List[ModelConfigResponse])
async def list_models(
    include_deleted: bool = Query(False, description="是否包含已软删除的配置"),
    db: AsyncSession = Depends(get_db),
):
    """List model configurations (AI-001)."""
    logger.info(f"Listing model configurations (include_deleted={include_deleted})")
    query = select(ModelConfig)
    if not include_deleted:
        query = query.where(ModelConfig.status == "active")
    result = await db.execute(query.order_by(ModelConfig.created_at.desc()))
    configs = result.scalars().all()
    responses = []
    for c in configs:
        key_warning = None
        try:
            plain_key = decrypt_api_key(c.api_key_encrypted, settings.ENCRYPTION_KEY)
            masked = mask_api_key(plain_key)
            key_warning = detect_key_mismatch(c.platform, plain_key)
        except Exception:
            masked = "****（请重新配置 API Key）"
        responses.append(ModelConfigResponse(
            id=str(c.id), name=c.name, platform=c.platform,
            base_url=c.base_url, api_key_masked=masked,
            model_name=c.model_name, max_tokens=c.max_tokens,
            temperature=c.temperature, is_default=c.is_default, is_enabled=c.is_enabled,
            status=c.status, key_warning=key_warning,
            extra_headers=c.extra_headers,
        ))
    logger.info(f"Returned {len(responses)} model configs")
    return responses


@router.post("", response_model=ModelConfigResponse, status_code=201)
async def create_model(req: ModelConfigCreate, db: AsyncSession = Depends(get_db)):
    """Create new model configuration (AI-002, AI-004)."""
    logger.info(f"Creating model config: name={req.name}, platform={req.platform}, model={req.model_name}")
    key_warning = detect_key_mismatch(req.platform, req.api_key)
    if key_warning:
        logger.warning(f"Key/platform mismatch on create: name={req.name} → {key_warning}")
    encrypted_key = encrypt_api_key(req.api_key, settings.ENCRYPTION_KEY)

    # 官方 API 只接受不带 opencode/ 前缀的模型 ID，保存前统一规范化
    model_name = (normalize_opencode_model(req.model_name)
                  if req.platform == "opencodezen" else req.model_name)

    config = ModelConfig(
        id=str(uuid4()), name=req.name, platform=req.platform,
        base_url=req.base_url, api_key_encrypted=encrypted_key,
        model_name=model_name, max_tokens=req.max_tokens,
        temperature=req.temperature, is_default=req.is_default,
        is_enabled=req.is_enabled, extra_headers=req.extra_headers,
    )

    # If setting as default, unset others
    if req.is_default:
        await db.execute(update(ModelConfig).values(is_default=False))

    db.add(config)
    await db.flush()
    logger.info(f"Model config created: id={config.id}, name={config.name}")
    # 实时生效：同步注册到网关，避免必须重启进程才能用上新模型（AI-006）
    try:
        from app.model_gateway import register_model_config
        register_model_config(get_model_gateway(), config, settings.ENCRYPTION_KEY)
        if config.is_default:
            get_model_gateway().set_default(str(config.id))
    except Exception as exc:
        logger.warning(f"Failed to register new model config in gateway: {exc}")
    return ModelConfigResponse(
        id=str(config.id), name=config.name, platform=config.platform,
        base_url=config.base_url, api_key_masked=mask_api_key(req.api_key),
        model_name=config.model_name, max_tokens=config.max_tokens,
        temperature=config.temperature, is_default=config.is_default,
        is_enabled=config.is_enabled,
        key_warning=detect_key_mismatch(req.platform, req.api_key),
        extra_headers=config.extra_headers,
    )


# -- Agent / Scene → Model Bindings (must precede /{config_id} routes) --
@router.get("/bindings")
async def list_bindings(
    active_only: bool = True,
    db: AsyncSession = Depends(get_db),
):
    """List agent-scene to model bindings (AI-003).

    Optionally include inactive history via active_only=false.
    Returns model details for convenient frontend rendering.
    """
    stmt = select(AgentModelBinding, ModelConfig).outerjoin(
        ModelConfig, AgentModelBinding.model_config_id == ModelConfig.id
    )
    if active_only:
        stmt = stmt.where(AgentModelBinding.is_active == True)
    stmt = stmt.order_by(AgentModelBinding.created_at.desc())
    result = await db.execute(stmt)
    rows = result.all()
    return [
        {
            "id": str(b.id),
            "model_config_id": str(b.model_config_id),
            "agent_id": b.agent_id,
            "scene": b.scene,
            "is_active": b.is_active,
            "model_name": c.name if c else None,
            "model_platform": c.platform if c else None,
            "model_model": c.model_name if c else None,
        }
        for b, c in rows
    ]


@router.post("/bindings", status_code=201)
async def create_binding(req: BindingCreate, db: AsyncSession = Depends(get_db)):
    """Create agent-scene to model binding."""
    binding = AgentModelBinding(
        id=str(uuid4()), model_config_id=req.model_config_id,
        agent_id=req.agent_id, scene=req.scene, is_active=True,
    )
    db.add(binding)
    await db.flush()
    return {"id": str(binding.id), "status": "created"}


class BindingUpsert(BaseModel):
    """Upsert payload: provide agent_id OR scene + target model_config_id."""
    model_config_id: str
    agent_id: Optional[str] = None
    scene: Optional[str] = None


@router.put("/bindings")
async def upsert_binding(req: BindingUpsert, db: AsyncSession = Depends(get_db)):
    """Upsert (replace) the active binding for an agent_id / scene.

    Deactivates any existing active binding for the same agent_id or scene,
    then inserts a new active binding pointing to the given model.
    """
    logger.info(
        f"Upserting binding: agent_id={req.agent_id}, scene={req.scene}, "
        f"model_config_id={req.model_config_id}"
    )
    if not req.agent_id and not req.scene:
        raise HTTPException(status_code=400, detail="agent_id or scene is required")

    # Verify the target model exists and is active
    model_result = await db.execute(
        select(ModelConfig).where(ModelConfig.id == req.model_config_id)
    )
    model = model_result.scalar_one_or_none()
    if not model:
        raise HTTPException(status_code=404, detail="Model config not found")
    if model.status == "deleted":
        raise HTTPException(status_code=400, detail="Cannot bind to a deleted model")

    # Deactivate existing active bindings for the same agent_id / scene
    if req.agent_id:
        await db.execute(
            update(AgentModelBinding)
            .where(
                AgentModelBinding.agent_id == req.agent_id,
                AgentModelBinding.is_active == True,
            )
            .values(is_active=False)
        )
    if req.scene:
        await db.execute(
            update(AgentModelBinding)
            .where(
                AgentModelBinding.scene == req.scene,
                AgentModelBinding.is_active == True,
            )
            .values(is_active=False)
        )

    # Insert new active binding
    binding = AgentModelBinding(
        id=str(uuid4()),
        model_config_id=req.model_config_id,
        agent_id=req.agent_id,
        scene=req.scene,
        is_active=True,
    )
    db.add(binding)
    await db.flush()
    logger.info(f"Binding upserted: id={binding.id}")
    return {
        "id": str(binding.id),
        "status": "upserted",
        "model_config_id": str(binding.model_config_id),
        "agent_id": binding.agent_id,
        "scene": binding.scene,
        "model_name": model.name,
        "model_platform": model.platform,
        "model_model": model.model_name,
    }


@router.delete("/bindings/{binding_id}", status_code=204)
async def delete_binding(binding_id: str, db: AsyncSession = Depends(get_db)):
    """Delete a binding (hard delete)."""
    logger.info(f"Deleting binding: id={binding_id}")
    result = await db.execute(
        select(AgentModelBinding).where(AgentModelBinding.id == binding_id)
    )
    binding = result.scalar_one_or_none()
    if not binding:
        logger.warning(f"Binding not found: id={binding_id}")
        raise HTTPException(status_code=404, detail="Binding not found")
    await db.delete(binding)
    logger.info(f"Binding deleted: id={binding_id}")


# -- Per-config-id routes (MUST come after /bindings) --
@router.put("/{config_id}", response_model=ModelConfigResponse)
async def update_model(config_id: str, req: ModelConfigUpdate, db: AsyncSession = Depends(get_db)):
    """Update model configuration (AI-006: real-time effect)."""
    logger.info(f"Updating model config: id={config_id}")
    result = await db.execute(select(ModelConfig).where(ModelConfig.id == config_id))
    config = result.scalar_one_or_none()
    if not config:
        logger.warning(f"Model config not found: id={config_id}")
        raise HTTPException(status_code=404, detail="Model config not found")

    changes = []
    if req.name is not None:
        config.name = req.name; changes.append("name")
    if req.platform is not None:
        config.platform = req.platform; changes.append("platform")
    if req.base_url is not None:
        config.base_url = req.base_url; changes.append("base_url")
    if req.api_key is not None:
        config.api_key_encrypted = encrypt_api_key(req.api_key, settings.ENCRYPTION_KEY); changes.append("api_key")
    if req.model_name is not None:
        # 编辑时同样清掉 OpenCode 客户端专用的模型前缀
        config.model_name = (normalize_opencode_model(req.model_name)
                             if config.platform == "opencodezen" else req.model_name)
        changes.append("model_name")
    if req.max_tokens is not None:
        config.max_tokens = req.max_tokens; changes.append("max_tokens")
    if req.temperature is not None:
        config.temperature = req.temperature; changes.append("temperature")
    if req.is_enabled is not None:
        config.is_enabled = req.is_enabled; changes.append("is_enabled")
    if req.extra_headers is not None:
        config.extra_headers = req.extra_headers; changes.append("extra_headers")
    if req.is_default:
        await db.execute(update(ModelConfig).where(ModelConfig.id != config_id).values(is_default=False))
        config.is_default = True; changes.append("is_default")

    await db.flush()
    logger.info(f"Model config updated: id={config_id}, changes={changes}")
    # 实时生效：API Key / 模型名变更后同步刷新网关中的配置
    try:
        from app.model_gateway import register_model_config
        register_model_config(get_model_gateway(), config, settings.ENCRYPTION_KEY)
        if config.is_default:
            get_model_gateway().set_default(str(config.id))
    except Exception as exc:
        logger.warning(f"Failed to sync updated model config to gateway: {exc}")
    plain_key = ""
    try:
        plain_key = decrypt_api_key(config.api_key_encrypted, settings.ENCRYPTION_KEY)
        masked = mask_api_key(plain_key)
    except Exception:
        masked = "****"
    return ModelConfigResponse(
        id=str(config.id), name=config.name, platform=config.platform,
        base_url=config.base_url,
        api_key_masked=masked,
        model_name=config.model_name, max_tokens=config.max_tokens,
        temperature=config.temperature, is_default=config.is_default,
        is_enabled=config.is_enabled,
        key_warning=detect_key_mismatch(config.platform, plain_key),
        extra_headers=config.extra_headers,
    )


@router.delete("/{config_id}", status_code=204)
async def delete_model(config_id: str, db: AsyncSession = Depends(get_db)):
    """Soft-delete model configuration (set status='deleted')."""
    logger.info(f"Soft-deleting model config: id={config_id}")
    result = await db.execute(select(ModelConfig).where(ModelConfig.id == config_id))
    config = result.scalar_one_or_none()
    if not config:
        logger.warning(f"Model config not found for deletion: id={config_id}")
        raise HTTPException(status_code=404, detail="Model config not found")
    if config.status == "deleted":
        logger.info(f"Model already deleted: id={config_id}")
    config.status = "deleted"
    logger.info(f"Model soft-deleted: name={config.name}, platform={config.platform}, model={config.model_name}")
    await db.flush()
    try:
        get_model_gateway().remove_config(str(config.id))
    except Exception as exc:
        logger.warning(f"Failed to remove model config from gateway: {exc}")


@router.post("/{config_id}/restore", response_model=ModelConfigResponse)
async def restore_model(config_id: str, db: AsyncSession = Depends(get_db)):
    """Restore a soft-deleted model configuration."""
    logger.info(f"Restoring model config: id={config_id}")
    result = await db.execute(select(ModelConfig).where(ModelConfig.id == config_id))
    config = result.scalar_one_or_none()
    if not config:
        raise HTTPException(status_code=404, detail="Model config not found")
    if config.status == "active":
        return ModelConfigResponse(
            id=str(config.id), name=config.name, platform=config.platform,
            base_url=config.base_url, api_key_masked="****",
            model_name=config.model_name, max_tokens=config.max_tokens,
            temperature=config.temperature, is_default=config.is_default,
            is_enabled=config.is_enabled, status=config.status,
            extra_headers=config.extra_headers,
        )
    config.status = "active"
    await db.flush()
    try:
        masked = mask_api_key(decrypt_api_key(config.api_key_encrypted, settings.ENCRYPTION_KEY))
    except Exception:
        masked = "****（请重新配置 API Key）"
    # 同步注册回网关，使其立即可用
    try:
        from app.model_gateway import register_model_config
        register_model_config(get_model_gateway(), config, settings.ENCRYPTION_KEY)
    except Exception as exc:
        logger.warning(f"恢复后注册网关失败: {exc}")
    logger.info(f"Model config restored: id={config_id}, name={config.name}")
    return ModelConfigResponse(
        id=str(config.id), name=config.name, platform=config.platform,
        base_url=config.base_url, api_key_masked=masked,
        model_name=config.model_name, max_tokens=config.max_tokens,
        temperature=config.temperature, is_default=config.is_default,
        is_enabled=config.is_enabled, status=config.status,
        extra_headers=config.extra_headers,
    )


@router.post("/{config_id}/test", response_model=TestResult)
async def test_model(config_id: str, db: AsyncSession = Depends(get_db)):
    """Test model connectivity (AI-005)."""
    logger.info(f"Testing model connection: config_id={config_id}")
    result = await db.execute(select(ModelConfig).where(ModelConfig.id == config_id))
    config = result.scalar_one_or_none()
    if not config:
        logger.warning(f"Model config not found for test: id={config_id}")
        raise HTTPException(status_code=404, detail="Model config not found")

    gateway = get_model_gateway()
    try:
        api_key = decrypt_api_key(config.api_key_encrypted, settings.ENCRYPTION_KEY)
    except Exception:
        api_key = ""

    # Key 与平台不匹配时直接给出结论，不再发一次注定失败的请求。
    # 典型场景：把 OpenRouter 的 sk-or-v1-… 填进 OpenCode Zen 配置，
    # 上游只会回 401 Invalid credential，用户会误以为是模型名写错。
    mismatch = detect_key_mismatch(config.platform, api_key)
    if mismatch:
        logger.warning(f"Model test blocked by key/platform mismatch: config_id={config_id} → {mismatch}")
        return TestResult(
            success=False, latency_ms=0.0, model=config.model_name,
            error=mismatch,
            hint="到该平台控制台重新生成 Key 并填写到对应配置；不同平台的 Key 不能混用",
        )

    # 用 probe 而不是 test_connection：不写入网关、忽略停用标记，
    # 这样已停用/已失效的配置也能测出真实原因（例如 API Key 过期）
    test = await gateway.probe(LLMConfig(
        id=str(config.id), name=config.name,
        provider=LLMProvider(config.platform),
        base_url=config.base_url or "",
        api_key=api_key,
        model_name=config.model_name,
        max_tokens=config.max_tokens, temperature=config.temperature,
        is_enabled=config.is_enabled,
        # Go 订阅需要 x-opencode-session 等自定义头，测试连接时也必须带上
        extra_headers=dict(config.extra_headers or {}),
    ))
    logger.info(
        f"Model test result: config_id={config_id}, success={test.get('success')}, "
        f"latency={test.get('latency_ms', 'N/A')}ms"
    )

    # 把英文原始错误翻译成中文原因 + 修复建议（原始信息保留在 raw_error）
    raw_error = test.get("error")
    if raw_error:
        reason, hint = explain_llm_error_with_hint(raw_error)
        test["error"] = reason
        test["raw_error"] = str(raw_error)[:800]
        test["hint"] = hint
    return TestResult(**test)


@router.put("/{config_id}/default")
async def set_default_model(config_id: str, db: AsyncSession = Depends(get_db)):
    """Set model as default."""
    logger.info(f"Setting default model: config_id={config_id}")
    await db.execute(update(ModelConfig).values(is_default=False))
    await db.execute(update(ModelConfig).where(ModelConfig.id == config_id).values(is_default=True))
    try:
        get_model_gateway().set_default(str(config_id))
    except Exception as exc:
        logger.warning(f"Failed to set gateway default model: {exc}")
    logger.info(f"Default model set: config_id={config_id}")


# -- Available models from provider --
class AvailableModelsRequest(BaseModel):
    platform: str
    base_url: Optional[str] = None
    api_key: Optional[str] = None
    config_id: Optional[str] = None


@router.post("/available")
async def fetch_available_models(req: AvailableModelsRequest, db: AsyncSession = Depends(get_db)):
    """Fetch available model IDs from a provider's /v1/models endpoint.

    Proxies the request through the backend to avoid CORS / network issues.
    Falls back to known model lists when the provider does not respond.
    """
    logger.info(f"Fetching available models: platform={req.platform}, base_url={req.base_url}, config_id={req.config_id}")
    # --- Determine base URL（优先级：请求传入 > 已存配置 > 平台默认）---
    platform_defaults = {
        "openai": "https://api.openai.com/v1",
        "deepseek": "https://api.deepseek.com/v1",
        "openrouter": "https://openrouter.ai/api/v1",
        "opencodezen": "https://opencode.ai/inference/openai/v1",
    }
    base_url = (req.base_url or "").rstrip("/")
    api_key = (req.api_key or "").strip()

    # 编辑模式刷新时前端可能不传 base_url / key，应优先复用已存配置。
    # 旧实现先把平台默认值填进 base_url，使 stored.base_url 分支永远不执行，
    # 结果 Go 订阅配置会被拉到 pay-as-you-go 的模型目录（70 个 vs 26 个）。
    if req.config_id and db is not None:
        stored = (await db.execute(
            select(ModelConfig).where(ModelConfig.id == req.config_id)
        )).scalar_one_or_none()
        if stored and stored.platform == req.platform:
            if not base_url and stored.base_url:
                base_url = stored.base_url.rstrip("/")
            if not api_key:
                try:
                    api_key = decrypt_api_key(stored.api_key_encrypted, settings.ENCRYPTION_KEY)
                except Exception as exc:
                    logger.warning(f"Failed to decrypt stored API key for available models: {exc}")

    if not base_url:
        base_url = platform_defaults.get(req.platform, "")

    # --- Built-in fallback models (always available) ---
    fallback_models = {
        "openai": ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-4", "gpt-3.5-turbo"],
        "deepseek": ["deepseek-chat", "deepseek-reasoner"],
        "openrouter": [
            "nvidia/nemotron-3-ultra-550b-a55b:free",
            "deepseek/deepseek-v4.1-flash",
            "z-ai/glm-5.3-prime",
            "xingchen-agi/xing4.0-29b-a4b:free",
            "openai/gpt-4o", "openai/gpt-4o-mini",
            "anthropic/claude-sonnet-4", "google/gemini-2.5-pro",
            "deepseek/deepseek-chat",
        ],
        # 官方 v2 文档：免费档（-free / big-pickle）只允许在 OpenCode 客户端内
        # 使用，其他客户端即使带有效 Key 也会收到 403。外部集成的下拉列表里
        # 只给付费模型，避免用户选完才发现调用必然失败。
        # 下列 ID 均取自 https://opencode.ai/inference/v1/models（2026-11 核对）。
        "opencodezen": [
            "glm-5.3-flash", "glm-5.3", "glm-5.2",
            "deepseek-v4.1-flash", "deepseek-v4-pro",
            "minimax-m3", "kimi-k3", "kimi-k2.7-code",
            "qwen3.8-flash", "gpt-5.5",
        ],
    }

    # 平台配额/策略说明：随模型列表一起返回，界面直接展示，省得用户反复试错
    platform_notes = {
        "opencodezen": ("官方策略：免费档（-free、big-pickle）仅限 OpenCode 客户端内使用，"
                        "外部 API 调用一律返回 403。外部集成请选付费模型"
                        "（Console workspace 需有 credits）或订阅 OpenCode Go；"
                        "Key 在 opencode.ai/console → Personal API keys 生成，形如 oc_sk_…"),
        "openrouter": ("免费 :free 模型按账户共享 50 次/天，UTC 00:00（北京时间 08:00）重置；"
                       "账户充值满 10 美元可解锁 1000 次/天"),
    }
    note = platform_notes.get(req.platform)


    # --- Try live fetch from provider ---
    blocked_models: List[str] = []
    try:
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        # OpenCode Console 的模型目录在 /inference/v1/models（Go 订阅为
        # /inference/go/v1/models）；OpenAI 兼容路径下的 /models 是 404。
        if req.platform == "opencodezen":
            is_go = "/inference/go" in base_url
            models_url = ("https://opencode.ai/inference/go/v1/models" if is_go
                          else "https://opencode.ai/inference/v1/models")
        else:
            models_url = f"{base_url}/models"

        async with httpx.AsyncClient(timeout=httpx.Timeout(8.0)) as client:
            resp = await client.get(models_url, headers=headers)
            if resp.status_code == 200:
                data = resp.json()
                items = data.get("data", [])
                # Extract model IDs, filter out non-chat/system models
                model_ids = []
                for item in items:
                    mid = item.get("id", "")
                    if not mid or any(
                        skip in mid.lower()
                        for skip in [
                            "whisper", "tts", "dall-e", "embedding", "moderation",
                            "babbage", "davinci", "audio", "image",
                        ]
                    ):
                        continue
                    # 官方目录带 type 字段（chat / embedding / rerank…）；
                    # 关键字过滤只能挡住已见过的命名，按 type 判断才不漏。
                    if str(item.get("type") or "chat").lower() != "chat":
                        continue
                    # 免费档禁止外部调用（官方返回 403），单独返回供界面标注
                    if (req.platform == "opencodezen"
                            and (mid.endswith("-free") or mid == "big-pickle")):
                        blocked_models.append(mid)
                        continue
                    model_ids.append(mid)

                if model_ids:
                    logger.info(
                        f"Available models (live): platform={req.platform}, "
                        f"count={len(model_ids)}, blocked={len(blocked_models)}"
                    )
                    return {
                        "platform": req.platform,
                        "source": "live",
                        "models": model_ids,
                        "blocked_models": blocked_models,
                        "note": note,
                    }

    except Exception as e:
        logger.warning(f"Live model fetch failed for {req.platform}: {e}")

    # --- Fallback ---
    models = fallback_models.get(req.platform, fallback_models.get("openai", []))
    logger.info(
        f"Available models (builtin): platform={req.platform}, count={len(models)}"
    )
    return {
        "platform": req.platform,
        "source": "builtin",
        "models": models,
        "blocked_models": blocked_models,
        "note": note,
    }
