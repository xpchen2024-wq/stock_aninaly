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
from starlette.middleware.base import BaseHTTPMiddleware

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


# ── Request Logging Middleware ───────────────────────────────────────────────
class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Log every HTTP request with method, path, status, and latency."""

    async def dispatch(self, request: Request, call_next):
        start_time = time.monotonic()
        method = request.method
        path = request.url.path
        query = request.url.query
        client = request.client.host if request.client else "unknown"

        # Log request start
        qs = f"?{query}" if query else ""
        logger.info(f"--> {method} {path}{qs}  | client={client}")

        try:
            response = await call_next(request)
            elapsed_ms = (time.monotonic() - start_time) * 1000
            status_code = response.status_code
            logger.info(
                f"<-- {method} {path}  | {status_code}  | {elapsed_ms:.1f}ms"
            )
            return response
        except Exception as exc:
            elapsed_ms = (time.monotonic() - start_time) * 1000
            logger.error(
                f"<-- {method} {path}  | 500  | {elapsed_ms:.1f}ms  | ERROR: {exc}"
            )
            raise


app.add_middleware(RequestLoggingMiddleware)


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
