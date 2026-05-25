import base64
import os
import random
from typing import Any, Dict, List, Tuple

import httpx
from dotenv import load_dotenv
from loguru import logger

from .base import register_provider
from .provider import TTSProvider

load_dotenv()

GRADIUM_VOICES = [
    {"id": "6MFfc37kq0sBjBjy", "name": "Sterling", "gender": "Male"},
    {"id": "_6Aslh2DxfmnRLmP", "name": "Russell", "gender": "Male"},
    {"id": "POBHtemksfWQbng0", "name": "Garrett", "gender": "Male"},
    {"id": "cLONiZ4hQ8VpQ4Sz", "name": "Skyler", "gender": "Female"},
    {"id": "vtG8ddh4IN32Otad", "name": "Quinn", "gender": "Female"},
    {"id": "7aEKz4P1ogZ0UsRP", "name": "Riley", "gender": "Female"},
]


@register_provider("gradium")
class GradiumProvider(TTSProvider):
    _api_key = None
    _base_url = "https://api.gradium.ai/api/post/speech/tts"
    _models = None
    _voices = GRADIUM_VOICES

    @classmethod
    def _initialize_provider(cls):
        """Initialize the Gradium provider."""
        cls._api_key = os.getenv("GRADIUM_API_KEY")
        if not cls._api_key:
            logger.error("Gradium API key not found in environment variables")
            raise ValueError("GRADIUM_API_KEY environment variable is required")

        cls._models = [
            {
                "id": "gradium",
                "name": "Gradium TTS",
                "description": "Gradium TTS model with randomized recommended voices",
            }
        ]

    @classmethod
    def get_available_models(cls) -> List[Dict[str, Any]]:
        """Get a list of available Gradium voice models."""
        if not cls.is_available() or not cls._models:
            return []

        return cls._models

    @classmethod
    async def synthesize(cls, text: str, model_id: str = None) -> Tuple[str, str]:
        """Synthesize speech using Gradium REST TTS."""
        if not cls.is_available():
            raise ValueError("Gradium provider is not available")

        if model_id and model_id != "gradium":
            logger.warning(f"Unknown Gradium model {model_id}, using default: gradium")

        voice = random.choice(cls._voices)
        voice_id = voice["id"]
        logger.info(f"Using Gradium voice: {voice['name']} ({voice_id})")

        headers = {
            "x-api-key": cls._api_key,
            "Content-Type": "application/json",
        }
        payload = {
            "text": text,
            "voice_id": voice_id,
            "output_format": "wav",
            "only_audio": True,
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
                        f"Gradium API error: {response.status_code} - {response.text}"
                    )
                    raise Exception(
                        f"Gradium API error: {response.status_code} - {response.text}"
                    )

                audio_data = base64.b64encode(response.content).decode("ascii")
                return audio_data, "wav"

            except Exception as e:
                logger.error(f"Error in Gradium synthesis: {str(e)}")
                raise Exception(f"Gradium synthesis error: {str(e)}")
