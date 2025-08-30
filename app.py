from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from loguru import logger
import os
from dotenv import load_dotenv
import base64
import io
from typing import Tuple

from pydub import AudioSegment
from pydub import effects as audio_effects

# Import TTS providers
from tts_providers import (
    get_available_providers,
    get_provider_models,
    synthesize_speech,
)

# Load environment variables
load_dotenv()

app = FastAPI(
    title="TTS Router API",
    description="API to route text-to-speech requests to different providers",
    version="1.0.0",
)


class TTSRequest(BaseModel):
    text: str
    provider: str
    model: str = None


@app.get("/")
async def root():
    return {"message": "TTS Router API"}


@app.get("/providers")
async def providers():
    """List all available TTS providers"""
    return {"providers": get_available_providers()}


@app.get("/providers/{provider}/models")
async def models(provider: str):
    """List all available models for a specific provider"""
    try:
        return {"models": get_provider_models(provider)}
    except ValueError as e:
        logger.error(f"Error fetching models for provider {provider}: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/tts")
async def tts(request: TTSRequest):
    """Generate TTS audio from text using specified provider and model"""
    try:
        provider = request.provider
        model = request.model
        text = request.text

        logger.info(
            f"TTS request received - Provider: {provider}, Model: {model}, Text length: {len(text)}"
        )

        audio_data, extension = await synthesize_speech(text, provider, model)

        # Normalize audio volume to reduce bias between providers
        try:
            audio_data = _normalize_base64_audio(audio_data, extension)
            logger.info("Applied peak normalization to output audio")
        except Exception as norm_err:
            logger.warning(f"Audio normalization failed, returning original audio: {norm_err}")

        logger.info(
            f"TTS request completed successfully - Provider: {provider}, Model: {model}"
        )

        # The audio_data is now a base64 encoded string (already encoded by the provider)
        return {
            "status": "success",
            "provider": provider,
            "model": model,
            "audio_data": audio_data,  # Already base64 encoded
            "extension": extension,
        }
    except Exception as e:
        logger.error(f"Error generating TTS: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)


def _normalize_base64_audio(b64_audio: str, extension: str) -> str:
    """Quick peak normalization on base64 audio while preserving format.

    - Decodes base64 -> AudioSegment
    - Applies fast peak normalization with small headroom
    - Re-encodes to original format and returns base64
    """
    raw = base64.b64decode(b64_audio)
    buf = io.BytesIO(raw)

    # Map some extensions to ffmpeg format names if needed
    fmt = (extension or "mp3").lower()
    fmt_map = {"m4a": "mp4"}
    load_fmt = fmt_map.get(fmt, fmt)

    audio = AudioSegment.from_file(buf, format=load_fmt)

    # Fast peak normalization with 1 dB headroom to avoid clipping
    normalized = audio_effects.normalize(audio, headroom=1.0)

    out = io.BytesIO()
    export_fmt = fmt
    normalized.export(out, format=export_fmt)
    return base64.b64encode(out.getvalue()).decode("ascii")
