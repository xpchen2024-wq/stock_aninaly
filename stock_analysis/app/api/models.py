# ============================================================================
# AI Stock Analysis Platform - AI Model Config API (AI-001 ~ AI-007)
# ============================================================================
from __future__ import annotations

import logging
from typing import Optional, List
from uuid import uuid4

import httpx
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update

from app.database import get_db
from app.config import get_settings
from app.models import ModelConfig, AgentModelBinding
from app.model_gateway import (
    ModelGateway, LLMConfig, LLMProvider,
    encrypt_api_key, decrypt_api_key, mask_api_key,
    get_model_gateway,
)

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


class TestResult(BaseModel):
    success: bool
    latency_ms: float
    model: str
    error: Optional[str] = None


class BindingCreate(BaseModel):
    model_config_id: str
    agent_id: Optional[str] = None
    scene: Optional[str] = None


# -- Routes --
@router.get("", response_model=List[ModelConfigResponse])
async def list_models(db: AsyncSession = Depends(get_db)):
    """List all model configurations (AI-001)."""
    logger.info("Listing all model configurations")
    result = await db.execute(
        select(ModelConfig)
        .where(ModelConfig.status == "active")
        .order_by(ModelConfig.created_at.desc())
    )
    configs = result.scalars().all()
    responses = []
    for c in configs:
        try:
            masked = mask_api_key(decrypt_api_key(c.api_key_encrypted, settings.ENCRYPTION_KEY))
        except Exception:
            masked = "****（请重新配置 API Key）"
        responses.append(ModelConfigResponse(
            id=str(c.id), name=c.name, platform=c.platform,
            base_url=c.base_url, api_key_masked=masked,
            model_name=c.model_name, max_tokens=c.max_tokens,
            temperature=c.temperature, is_default=c.is_default, is_enabled=c.is_enabled,
        ))
    logger.info(f"Returned {len(responses)} model configs")
    return responses


@router.post("", response_model=ModelConfigResponse, status_code=201)
async def create_model(req: ModelConfigCreate, db: AsyncSession = Depends(get_db)):
    """Create new model configuration (AI-002, AI-004)."""
    logger.info(f"Creating model config: name={req.name}, platform={req.platform}, model={req.model_name}")
    encrypted_key = encrypt_api_key(req.api_key, settings.ENCRYPTION_KEY)

    config = ModelConfig(
        id=str(uuid4()), name=req.name, platform=req.platform,
        base_url=req.base_url, api_key_encrypted=encrypted_key,
        model_name=req.model_name, max_tokens=req.max_tokens,
        temperature=req.temperature, is_default=req.is_default,
        is_enabled=req.is_enabled, extra_headers=req.extra_headers,
    )

    # If setting as default, unset others
    if req.is_default:
        await db.execute(update(ModelConfig).values(is_default=False))

    db.add(config)
    await db.flush()
    logger.info(f"Model config created: id={config.id}, name={config.name}")
    return ModelConfigResponse(
        id=str(config.id), name=config.name, platform=config.platform,
        base_url=config.base_url, api_key_masked=mask_api_key(req.api_key),
        model_name=config.model_name, max_tokens=config.max_tokens,
        temperature=config.temperature, is_default=config.is_default,
        is_enabled=config.is_enabled,
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
        config.model_name = req.model_name; changes.append("model_name")
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
    try:
        masked = mask_api_key(decrypt_api_key(config.api_key_encrypted, settings.ENCRYPTION_KEY))
    except Exception:
        masked = "****"
    return ModelConfigResponse(
        id=str(config.id), name=config.name, platform=config.platform,
        base_url=config.base_url,
        api_key_masked=masked,
        model_name=config.model_name, max_tokens=config.max_tokens,
        temperature=config.temperature, is_default=config.is_default,
        is_enabled=config.is_enabled,
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
    gateway.add_config(LLMConfig(
        id=str(config.id), name=config.name,
        provider=LLMProvider(config.platform),
        base_url=config.base_url or "",
        api_key=api_key,
        model_name=config.model_name,
        max_tokens=config.max_tokens, temperature=config.temperature,
        is_enabled=config.is_enabled,
    ))

    test = await gateway.test_connection(str(config.id))
    logger.info(
        f"Model test result: config_id={config_id}, success={test.get('success')}, "
        f"latency={test.get('latency_ms', 'N/A')}ms"
    )
    return TestResult(**test)


@router.put("/{config_id}/default")
async def set_default_model(config_id: str, db: AsyncSession = Depends(get_db)):
    """Set model as default."""
    logger.info(f"Setting default model: config_id={config_id}")
    await db.execute(update(ModelConfig).values(is_default=False))
    await db.execute(update(ModelConfig).where(ModelConfig.id == config_id).values(is_default=True))
    logger.info(f"Default model set: config_id={config_id}")


# -- Available models from provider --
class AvailableModelsRequest(BaseModel):
    platform: str
    base_url: Optional[str] = None
    api_key: Optional[str] = None


@router.post("/available")
async def fetch_available_models(req: AvailableModelsRequest):
    """Fetch available model IDs from a provider's /v1/models endpoint.

    Proxies the request through the backend to avoid CORS / network issues.
    Falls back to known model lists when the provider does not respond.
    """
    logger.info(f"Fetching available models: platform={req.platform}, base_url={req.base_url}")
    # --- Determine base URL ---
    base_url = (req.base_url or "").rstrip("/")
    if not base_url:
        # Use well-known defaults
        platform_defaults = {
            "openai": "https://api.openai.com/v1",
            "deepseek": "https://api.deepseek.com/v1",
            "openrouter": "https://openrouter.ai/api/v1",
            "opencodezen": "https://opencode.ai/zen/v1",
        }
        base_url = platform_defaults.get(req.platform, "")

    # --- Built-in fallback models (always available) ---
    fallback_models = {
        "openai": ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-4", "gpt-3.5-turbo"],
        "deepseek": ["deepseek-chat", "deepseek-reasoner"],
        "openrouter": [
            "openai/gpt-4o", "openai/gpt-4o-mini", "openai/gpt-4-turbo",
            "anthropic/claude-sonnet-4", "anthropic/claude-3.5-sonnet",
            "google/gemini-2.5-pro", "google/gemini-2.5-flash",
            "meta-llama/llama-4-maverick", "deepseek/deepseek-chat",
        ],
        "opencodezen": [
            "opencode/gpt-5.5", "opencode/claude-4.5", "opencode/claude-sonnet-4",
            "opencode/gemini-2.5-pro", "opencode/gpt-4o",
        ],
    }

    api_key = (req.api_key or "").strip()

    # --- Try live fetch from provider ---
    try:
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        async with httpx.AsyncClient(timeout=httpx.Timeout(8.0)) as client:
            resp = await client.get(f"{base_url}/models", headers=headers)
            if resp.status_code == 200:
                data = resp.json()
                items = data.get("data", [])
                # Extract model IDs, filter out non-chat/system models
                model_ids = []
                for item in items:
                    mid = item.get("id", "")
                    if mid and not any(
                        skip in mid.lower()
                        for skip in [
                            "whisper", "tts", "dall-e", "embedding", "moderation",
                            "babbage", "davinci", "audio", "image",
                        ]
                    ):
                        model_ids.append(mid)

                if model_ids:
                    logger.info(
                        f"Available models (live): platform={req.platform}, count={len(model_ids)}"
                    )
                    return {
                        "platform": req.platform,
                        "source": "live",
                        "models": model_ids,
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
    }
