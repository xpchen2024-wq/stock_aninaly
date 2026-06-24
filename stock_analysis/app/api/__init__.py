# ============================================================================
# AI Stock Analysis Platform - API Routes Package
# ============================================================================
from app.api.auth import router as auth_router
from app.api.data import router as data_router
from app.api.models import router as model_router
from app.api.analysis import router as analysis_router
from app.api.agent import router as agent_router
from app.api.watchlist import router as watchlist_router
from app.api.news import router as news_router
from app.api.kol import router as kol_router

__all__ = [
    "auth_router",
    "data_router",
    "model_router",
    "analysis_router",
    "agent_router",
    "watchlist_router",
    "news_router",
    "kol_router",
]
