"""FastAPI application factory.

The Vercel entry point (``api/index.py``) imports ``app`` from this module.
"""
from __future__ import annotations

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app import __version__
from app.config import settings
from app.routers import chat, health, session

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
log = logging.getLogger("vocalume")


def create_app() -> FastAPI:
    app = FastAPI(
        title="VocaLume API",
        description=(
            "Real-time English speaking coach backend. LangChain + LangGraph "
            "orchestration over NVIDIA NIM LLM. Designed for Vercel serverless."
        ),
        version=__version__,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.allowed_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(health.router)
    app.include_router(chat.router)
    app.include_router(session.router)

    @app.on_event("startup")
    async def _on_startup() -> None:
        if not settings.nim_configured:
            log.warning(
                "NVIDIA_API_KEY not configured — LLM endpoints will return "
                "errors. Set it in .env (local) or Vercel env vars (prod)."
            )
        else:
            log.info("VocaLume API starting  ·  NIM model: %s", settings.nim_llm_model)

    return app


app = create_app()