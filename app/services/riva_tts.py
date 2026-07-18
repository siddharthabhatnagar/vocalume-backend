"""Riva TTS (Text-to-Speech) client.

Same caveat as ``riva_asr.py``: Riva gRPC cannot run on Vercel. Talk to an
external Riva TTS proxy (self-hosted). If no URL is set, the Android client
should use Android's built-in TextToSpeech engine as a fallback.
"""
from __future__ import annotations

import httpx

from app.config import settings


class RivaTTSUnavailable(RuntimeError):
    pass


async def synthesize(text: str, voice: str = "English-US-Female-1", sample_rate: int = 22050) -> bytes:
    if not settings.riva_tts_url:
        raise RivaTTSUnavailable(
            "RIVA_TTS_URL is not set. Either deploy a Riva TTS proxy and set "
            "RIVA_TTS_URL, or use Android's on-device TextToSpeech engine."
        )
    async with httpx.AsyncClient(timeout=20.0) as client:
        resp = await client.post(
            f"{settings.riva_tts_url.rstrip('/')}/tts",
            json={"text": text, "voice": voice, "sample_rate": sample_rate},
        )
        resp.raise_for_status()
        return resp.content