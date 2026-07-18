"""
Health and info endpoints.

`/health` is a minimal liveness check suitable for uptime monitors and
Vercel's own health probing -- it does not touch Cerebras or any
external dependency, so it stays fast and reliable even if the LLM
provider is down. `/info` exposes a bit more operational detail
(whether Cerebras is configured, which model is active, history
window size) which is useful for debugging a deployment without
needing shell access to the serverless environment.
"""

from fastapi import APIRouter

from app import __version__
from app.config import get_settings

router = APIRouter(tags=["health"])


@router.get("/health")
async def health() -> dict:
    """Basic liveness check."""
    return {"status": "ok", "service": "vocalume-backend", "version": __version__}


@router.get("/info")
async def info() -> dict:
    """Operational info about the current deployment's configuration."""
    settings = get_settings()
    return {
        "service": "vocalume-backend",
        "version": __version__,
        "cerebras_configured": settings.cerebras_configured,
        "cerebras_model": settings.cerebras_model,
        "max_history_turns": settings.max_history_turns,
    }
