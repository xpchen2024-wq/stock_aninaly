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
from app.api.evaluation import router as evaluation_router
from app.api.backtest import router as backtest_router
from app.api.capital import router as capital_router
from app.api.insight import router as insight_router
from app.api.forecast import router as forecast_router

__all__ = [
    "auth_router",
    "data_router",
    "model_router",
    "analysis_router",
    "agent_router",
    "watchlist_router",
    "news_router",
    "kol_router",
    "evaluation_router",
    "backtest_router",
    "capital_router",
    "insight_router",
    "forecast_router",
]
