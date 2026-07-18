"""
FastAPI application factory for the VocaLume backend.

`create_app()` assembles the FastAPI instance: it configures CORS from
`Settings.allowed_origins`, mounts the health, chat, and session
routers, and registers a startup event that logs whether the Cerebras
API key is configured (a missing/invalid key is not fatal at startup
-- it only causes 500s from routes that actually need to call the
LLM -- but logging it loudly here makes misconfiguration obvious in
Vercel's function logs immediately after a deploy). The module-level
`app` object is what `api/index.py` re-exports for Vercel to serve.
"""

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app import __version__
from app.config import get_settings
from app.routers import chat, health, session

logger = logging.getLogger("vocalume")


def create_app() -> FastAPI:
    """Build and configure the VocaLume FastAPI application."""
    settings = get_settings()

    application = FastAPI(
        title="VocaLume API",
        version=__version__,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    application.add_middleware(
        CORSMiddleware,
        allow_origins=settings.allowed_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    application.include_router(health.router)
    application.include_router(chat.router)
    application.include_router(session.router)

    @application.on_event("startup")
    async def _log_startup_config() -> None:
        if settings.cerebras_configured:
            logger.info(
                "VocaLume backend starting with Cerebras model '%s'.",
                settings.cerebras_model,
            )
        else:
            logger.warning(
                "CEREBRAS_API_KEY is not configured (or does not start with "
                "'csk-'). Chat endpoints will fail until it is set."
            )

    return application


app = create_app()
