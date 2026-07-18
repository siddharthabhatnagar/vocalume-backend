"""Riva ASR (Speech-to-Text) client.

⚠️ NOTE FOR VERCEL DEPLOYMENT
─────────────────────────────
NVIDIA Riva uses gRPC, which Vercel's serverless runtime does not support.
Self-host Riva on a GPU box and set ``RIVA_ASR_URL`` to point at it.
"""
from __future__ import annotations

import httpx

from app.config import settings


class RivaASRUnavailable(RuntimeError):
    pass


async def transcribe(audio_bytes: bytes, sample_rate: int = 16000, language: str = "en-US") -> str:
    if not settings.riva_asr_url:
        raise RivaASRUnavailable(
            "RIVA_ASR_URL is not set. Either deploy a Riva ASR proxy and set "
            "RIVA_ASR_URL, or run ASR on the Android client (recommended for Vercel)."
        )
    async with httpx.AsyncClient(timeout=20.0) as client:
        resp = await client.post(
            f"{settings.riva_asr_url.rstrip('/')}/asr",
            content=audio_bytes,
            headers={
                "Content-Type": "application/octet-stream",
                "X-Sample-Rate": str(sample_rate),
                "X-Language": language,
            },
        )
        resp.raise_for_status()
        return resp.json().get("transcript", "")