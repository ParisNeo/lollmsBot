"""
FastAPI router for the LollmsBot Web UI.

Provides API routes for the web interface, including health checks,
settings management, and conversation endpoints.
"""

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

ui_router = APIRouter(
    prefix="/ui-api",
    tags=["ui"],
    responses={404: {"description": "Not found"}},
)


@ui_router.get("/health")
async def ui_health() -> dict:
    """Health check endpoint for the UI."""
    return {
        "status": "ok",
        "service": "lollmsbot-ui",
        "version": "0.1.0",
    }


@ui_router.get("/config")
async def ui_config(request: Request) -> dict:
    """Get current UI configuration (safe values only)."""
    # Return non-sensitive configuration for the frontend
    return {
        "max_history": 10,
        "features": {
            "tools": True,
            "settings": True,
            "streaming": True,
        },
    }
