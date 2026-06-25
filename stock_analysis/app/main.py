# ============================================================================
# AI Stock Analysis Platform - FastAPI Application Entry
# ============================================================================
from __future__ import annotations

import logging
import sys
import time
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import get_settings
from app.database import init_db, close_db
from app.cache import close_redis

settings = get_settings()

# ── Setup logging ───────────────────────────────────────────────────────────
# Create logs directory if file logging is enabled
if settings.LOG_FILE:
    os.makedirs(os.path.dirname(settings.LOG_FILE) or ".", exist_ok=True)

# Resolve log level from settings
_log_level = getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO)

# Build handlers
_handlers: list = [logging.StreamHandler(sys.stdout)]
if settings.LOG_FILE:
    _handlers.append(logging.FileHandler(settings.LOG_FILE, encoding="utf-8"))

# Unified formatter
_formatter = logging.Formatter(
    fmt=settings.LOG_FORMAT,
    datefmt=settings.LOG_DATE_FORMAT,
)

# Configure root logger so all modules inherit
_root_logger = logging.getLogger()
_root_logger.setLevel(_log_level)
# Remove any pre-existing handlers to avoid duplicates
for h in _root_logger.handlers[:]:
    _root_logger.removeHandler(h)
for h in _handlers:
    h.setFormatter(_formatter)
    _root_logger.addHandler(h)

# Set uvicorn access log level
logging.getLogger("uvicorn.access").setLevel(_log_level)
logging.getLogger("uvicorn.error").setLevel(_log_level)

logger = logging.getLogger(__name__)
logger.info(f"Logging configured: level={settings.LOG_LEVEL}, file={settings.LOG_FILE or '(console only)'}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle management."""
    logger.info(f"Starting {settings.APP_NAME} v{settings.APP_VERSION}")
    await init_db()
    logger.info("Database initialized")
    yield
    logger.info("Shutting down...")
    await close_db()
    await close_redis()
    logger.info("Shutdown complete")


# Create app
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="AI-driven stock analysis platform with multi-agent collaboration",
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / Response Logging Middleware (ASGI) ────────────────────────────
_MAX_BODY_LOG = 2000  # 截断超过 2KB 的 body，避免控制台被刷屏


def _safe_decode(b: bytes) -> str:
    """Best-effort UTF-8 decode; fall back to hex for binary payloads."""
    if not b:
        return ""
    try:
        return b.decode("utf-8")
    except UnicodeDecodeError:
        return f"<binary {len(b)} bytes>: {b[:64].hex()}"


class RequestResponseLoggingMiddleware:
    """Pure ASGI middleware that logs request line, request body, response
    status, latency and response body. Works with both FastAPI handlers and
    mounted sub-apps (e.g. /docs, /ui)."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        start = time.monotonic()
        method = scope.get("method", "?")
        path = scope.get("path", "?")
        raw_qs = scope.get("query_string", b"").decode("latin-1")
        full_path = f"{path}?{raw_qs}" if raw_qs else path
        client_host = (scope.get("client") or ["unknown"])[0]

        # ── 收集请求体 ──────────────────────────────────────────────
        req_chunks: list[bytes] = []

        async def receive_wrapper():
            message = await receive()
            if message.get("type") == "http.request":
                chunk = message.get("body", b"") or b""
                if chunk:
                    req_chunks.append(chunk)
            return message

        # ── 收集响应体 ──────────────────────────────────────────────
        status_holder = {"code": 500}
        resp_chunks: list[bytes] = []

        async def send_wrapper(message):
            if message.get("type") == "http.response.start":
                status_holder["code"] = message.get("status", 500)
            elif message.get("type") == "http.response.body":
                chunk = message.get("body", b"") or b""
                if chunk:
                    resp_chunks.append(chunk)
            await send(message)

        # ── 跳过明显的静态/健康探活噪音（可按需调整）────────────────
        skip_paths = ("/health", "/docs", "/openapi.json", "/redoc")
        is_skippable = path in skip_paths

        if not is_skippable:
            logger.info(f"--> {method} {full_path}  | client={client_host}")
        else:
            logger.debug(f"--> {method} {full_path}  | client={client_host}")

        try:
            await self.app(scope, receive_wrapper, send_wrapper)
        except Exception as exc:
            elapsed_ms = (time.monotonic() - start) * 1000
            logger.error(
                f"<-- {method} {full_path}  | 500  | {elapsed_ms:.1f}ms  | "
                f"ERROR: {exc}",
                exc_info=True,
            )
            raise

        elapsed_ms = (time.monotonic() - start) * 1000
        status_code = status_holder["code"]
        req_body = b"".join(req_chunks)
        resp_body = b"".join(resp_chunks)

        # 截断过长的 body
        req_text = _safe_decode(req_body)
        if len(req_text) > _MAX_BODY_LOG:
            req_text = req_text[:_MAX_BODY_LOG] + f"... <truncated, total {len(req_body)} bytes>"

        resp_text = _safe_decode(resp_body)
        if len(resp_text) > _MAX_BODY_LOG:
            resp_text = resp_text[:_MAX_BODY_LOG] + f"... <truncated, total {len(resp_body)} bytes>"

        if is_skippable:
            # 静态/健康探活只打 DEBUG 摘要
            logger.debug(
                f"<-- {method} {full_path}  | {status_code}  | {elapsed_ms:.1f}ms"
            )
            return

        # 单行汇总
        logger.info(
            f"<-- {method} {full_path}  | {status_code}  | {elapsed_ms:.1f}ms"
        )
        # 单独打印请求体（多行 JSON 时保留缩进）
        if req_text:
            logger.info(f"    request_body  : {req_text}")
        # 单独打印返回体
        if resp_text:
            logger.info(f"    response_body : {resp_text}")


app.add_middleware(RequestResponseLoggingMiddleware)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(
        f"Unhandled exception on {request.method} {request.url.path}: {exc}",
        exc_info=True,
    )
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "detail": str(exc)},
    )


@app.get("/")
async def root():
    logger.debug("Root endpoint called")
    return {
        "name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "status": "running",
        "docs": "/docs",
        "ui": "/ui",
    }


# Serve frontend UI (static design prototype)
from fastapi.staticfiles import StaticFiles
from pathlib import Path

_ui_dir = Path(__file__).parent.parent.parent / "UI_design"
if _ui_dir.exists():
    app.mount("/ui", StaticFiles(directory=str(_ui_dir), html=True), name="ui")
    logger.info(f"Frontend UI mounted at /ui from {_ui_dir}")
else:
    logger.warning(f"UI design directory not found: {_ui_dir}")


@app.get("/health")
async def health_check():
    logger.debug("Health check called")
    return {"status": "healthy", "version": settings.APP_VERSION}


# Import and register routers
from app.api import (
    data_router, model_router, analysis_router,
    agent_router, watchlist_router, news_router,
    kol_router, auth_router,
)

app.include_router(auth_router, prefix="/api/v1/auth", tags=["Authentication"])
app.include_router(data_router, prefix="/api/v1/data", tags=["Data"])
app.include_router(model_router, prefix="/api/v1/models", tags=["AI Models"])
app.include_router(analysis_router, prefix="/api/v1/analysis", tags=["Analysis"])
app.include_router(agent_router, prefix="/api/v1/agent", tags=["Agent"])
app.include_router(watchlist_router, prefix="/api/v1/watchlist", tags=["Watchlist"])
app.include_router(news_router, prefix="/api/v1/news", tags=["News"])
app.include_router(kol_router, prefix="/api/v1/kol", tags=["KOL"])

logger.info(
    f"Routers registered: auth, data, models, analysis, agent, "
    f"watchlist, news, kol"
)
