import os
import requests
import base64
import random
from loguru import logger
import numpy as np
import httpx
import io
import soundfile as sf
from typing import Dict, List, Tuple, Any

from .provider import TTSProvider
from .base import register_provider

ASYNC_VOICES = [
    "e0f39dc4-f691-4e78-bba5-5c636692cc04", # Nyomi
    "d2f7a640-be33-4a9b-a203-4dd6d5ade1e4", # Griffin
    "13616e5f-6fda-4247-b548-8821cb71fb54", # Alden
    "ea2ab73a-c93c-4cc1-9d7c-c767cc9b879a", # Sofia
    "f5b7eb43-2365-410a-95e0-beb92768809c", # Xavier
]


@register_provider("async")
class AsyncProvider(TTSProvider):
    _api_key = None
    _base_url = "https://api.async.ai/text_to_speech/streaming"
    _models = None

    @classmethod
    def _initialize_provider(cls):
        """Initialize the async provider"""
        cls._api_key = os.getenv("ASYNC_API_KEY")
        if not cls._api_key:
            logger.error("ASYNC API key not found in environment variables")
            raise ValueError("ASYNC_KEY environment variable is required")

        # Set up available models
        cls._models = [
            "asyncflow_v2.0"
        ]

    @classmethod
    def get_available_models(cls) -> List[Dict[str, Any]]:
        """Get a list of available models for async"""
        if not cls.is_available() or not cls._models:
            return ["async"]

        return cls._models

    @classmethod
    async def synthesize(cls, text: str, model_id: str = None) -> Tuple[str, str]:
        """Synthesize speech using async"""
        if not cls.is_available():
            raise ValueError("async provider is not available")

        selected_voice = random.choice(ASYNC_VOICES)
        payload = {
            "model_id": "asyncflow_v2.0",
            "transcript": text,
            "voice": { 
                "mode": "id", 
                "id": selected_voice 
            },
            "output_format": {
                "container": "raw",
                "encoding": "pcm_s16le",
                "sample_rate": 44100
            }
        }

        headers = {
            "X-Api-Key": cls._api_key,
        }
        audio_data = bytearray()
        try:
            with httpx.Client(http2=True, verify=False) as client:
                with client.stream(
                    "POST", 
                    cls._base_url, 
                    json=payload, 
                    headers=headers
                ) as response:
                    for chunk in response.iter_bytes():
                        audio_data.extend(chunk)

            audio_np = np.frombuffer(audio_data, dtype=np.int16)#.astype(np.float32) / 32768.0
            print(f"Audio data length: {len(audio_np)}")

            buffer = io.BytesIO()
            sf.write(
                buffer, 
                audio_np, 
                samplerate=44100, 
                format="WAV", 
                subtype="PCM_16"
            )
            buffer.seek(0)

            # Encode to base64
            audio_data = base64.b64encode(
                buffer.read()
            ).decode("ascii")

            return audio_data, "wav"
        except Exception as e:
            logger.error(f"Error in async synthesis: {str(e)}")
            raise Exception(f"async synthesis error: {str(e)}")
