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

TYPECAST_VOICES = [
    {"id": "tc_65b34b05b3fb844f3d6b7aab", "name": "Simon", "accent": "US", "gender": "Male"},
    {"id": "tc_67b6985d4d5d632d97478263", "name": "Aaron", "accent": "US", "gender": "Male"},
    {"id": "tc_67bfc776d41bad708fdf4ef9", "name": "Rusty", "accent": "US", "gender": "Male"},
    {"id": "tc_67d238428572120c4aa644cc", "name": "Zoey", "accent": "US", "gender": "Female"},
    {"id": "tc_660e5c29eef728e75f95f538", "name": "Patricia", "accent": "US", "gender": "Female"},
    {"id": "tc_67a440ec1e05bd5665857efd", "name": "Margot", "accent": "US", "gender": "Female"},
    {"id": "tc_63aaebfaf95b9c23b311c88d", "name": "Graham", "accent": "UK", "gender": "Male"},
    {"id": "tc_645349827a050a4142d49edf", "name": "Maisie", "accent": "UK", "gender": "Female"},
]


@register_provider("typecast")
class TypecastProvider(TTSProvider):
    _api_key = None
    _base_url = "https://api.typecast.ai/v1/text-to-speech"
    _model_name = "ssfm-v30"
    _models = None
    _voices = TYPECAST_VOICES

    @classmethod
    def _initialize_provider(cls):
        """Initialize the Typecast provider."""
        cls._api_key = os.getenv("TYPECAST_API_KEY")
        if not cls._api_key:
            logger.error("Typecast API key not found in environment variables")
            raise ValueError("TYPECAST_API_KEY environment variable is required")

        cls._models = [
            {
                "id": "typecast",
                "name": "Typecast SSFM 3.0",
                "description": f"Typecast {cls._model_name} model with randomized recommended voices",
            }
        ]

    @classmethod
    def get_available_models(cls) -> List[Dict[str, Any]]:
        """Get a list of available Typecast voice models."""
        if not cls.is_available() or not cls._models:
            return []

        return cls._models

    @classmethod
    async def synthesize(cls, text: str, model_id: str = None) -> Tuple[str, str]:
        """Synthesize speech using Typecast TTS."""
        if not cls.is_available():
            raise ValueError("Typecast provider is not available")

        if model_id and model_id != "typecast":
            logger.warning(f"Unknown Typecast model {model_id}, using default: typecast")

        voice = random.choice(cls._voices)
        voice_id = voice["id"]
        logger.info(f"Using Typecast voice: {voice['name']} ({voice_id})")

        headers = {
            "X-API-KEY": cls._api_key,
            "Content-Type": "application/json",
        }
        payload = {
            "text": text,
            "model": cls._model_name,
            "voice_id": voice_id,
            "prompt": {
                "emotion_type": "preset",
                "emotion_preset": "normal",
            },
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
                        f"Typecast API error: {response.status_code} - {response.text}"
                    )
                    raise Exception(
                        f"Typecast API error: {response.status_code} - {response.text}"
                    )

                audio_data = base64.b64encode(response.content).decode("ascii")
                return audio_data, "wav"

            except Exception as e:
                logger.error(f"Error in Typecast synthesis: {str(e)}")
                raise Exception(f"Typecast synthesis error: {str(e)}")
