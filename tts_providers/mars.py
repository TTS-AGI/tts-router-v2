import os
import io
import httpx
import base64
from loguru import logger
from typing import Dict, List, Tuple, Any
from pydub import AudioSegment

from .provider import TTSProvider
from .base import register_provider

from dotenv import load_dotenv
load_dotenv()


@register_provider("mars")
class MarsProvider(TTSProvider):
    _api_key = None
    _base_url = "https://mars-hf-leaderboard.camb.ai/predict"

    @classmethod
    def _initialize_provider(cls):
        """Initialize the MARS (Camb.ai) provider"""
        cls._api_key = os.getenv("MARS_API_KEY")
        if not cls._api_key:
            logger.error("MARS API key not found in environment variables")
            raise ValueError("MARS_API_KEY environment variable is required")

        logger.info("MARS (Camb.ai) provider initialized")

    @classmethod
    def get_available_models(cls) -> List[Dict[str, Any]]:
        """Get a list of available models for MARS"""
        if not cls.is_available():
            return []

        return [
            {
                "id": "mars",
                "name": "MARS",
                "description": "MARS text-to-speech by Camb.ai",
            },
        ]

    @classmethod
    async def synthesize(cls, text: str, model_id: str = None) -> Tuple[str, str]:
        """Synthesize speech using MARS (Camb.ai) API"""
        if not cls.is_available():
            raise ValueError("MARS provider is not available")

        headers = {
            "Content-Type": "application/json",
            "X-API-Key": cls._api_key,
        }

        payload = {
            "text": text,
            "language": "en-us",
            "only_predefined_voices": True,
            "stream": True,
        }

        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    cls._base_url,
                    headers=headers,
                    json=payload,
                    timeout=60.0,
                )

                if response.status_code != 200:
                    logger.error(
                        f"MARS API error: {response.status_code} - {response.text}"
                    )
                    raise Exception(
                        f"MARS API error: {response.status_code} - {response.text}"
                    )

                # Response is FLAC audio binary — convert to WAV
                flac_audio = AudioSegment.from_file(io.BytesIO(response.content), format="flac")
                wav_buffer = io.BytesIO()
                flac_audio.export(wav_buffer, format="wav")
                wav_bytes = wav_buffer.getvalue()

                audio_b64 = base64.b64encode(wav_bytes).decode("ascii")
                return audio_b64, "wav"

            except Exception as e:
                logger.error(f"Error in MARS synthesis: {str(e)}")
                raise Exception(f"MARS synthesis error: {str(e)}")
