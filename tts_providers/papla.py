import os
import requests
import base64
from loguru import logger
from typing import Dict, List, Tuple, Any

from .provider import TTSProvider
from .base import register_provider

# Default voice ID for Papla
DEFAULT_VOICE_ID = "e54f37b3-818d-486b-9083-88f07f15d0e4"


@register_provider("papla")
class PaplaProvider(TTSProvider):
    _api_key = None
    _base_url = "https://api.papla.media/v1"
    _models = None
    _voices = [DEFAULT_VOICE_ID]

    @classmethod
    def _initialize_provider(cls):
        """Initialize the Papla provider"""
        cls._api_key = os.getenv("PAPLA_API_KEY")
        if not cls._api_key:
            logger.error("Papla API key not found in environment variables")
            raise ValueError("PAPLA_API_KEY environment variable is required")

        # Set up available models
        cls._models = [
            {
                "id": "papla_p1",
                "name": "Papla P1",
                "description": "Papla's primary text-to-speech model",
            }
        ]
        try:
            _voices = cls.get_available_voices()
            cls._voices = [voice["voice_id"] for voice in _voices]
        except Exception as e:
            logger.error(f"Papla API error: {str(e)}")
            raise ValueError(f"Papla API error: {str(e)}")

    @classmethod
    def get_available_voices(cls) -> List[Dict[str, Any]]:
        """Get a list of available voices for Papla"""
        response = requests.get(
            f"{cls._base_url}/voices", headers={"papla-api-key": cls._api_key}
        )
        return response.json()

    @classmethod
    def get_available_models(cls) -> List[Dict[str, Any]]:
        """Get a list of available models for Papla"""
        if not cls.is_available() or not cls._models:
            return []

        return cls._models

    @classmethod
    async def synthesize(cls, text: str, model_id: str = None) -> Tuple[str, str]:
        """Synthesize speech using Papla"""
        if not cls.is_available():
            raise ValueError("Papla provider is not available")

        # Default model if none specified
        if not model_id:
            model_id = "papla_p1"
            logger.info(f"No model specified for Papla, using default: {model_id}")

        try:
            headers = {"papla-api-key": cls._api_key}

            payload = {"text": text, "model_id": model_id}

            response = requests.post(
                f"{cls._base_url}/text-to-speech/{DEFAULT_VOICE_ID}",
                json=payload,
                headers=headers,
            )

            if response.status_code != 200:
                logger.error(
                    f"Papla API error: {response.status_code} - {response.text}"
                )
                raise Exception(
                    f"Papla API error: {response.status_code} - {response.text}"
                )

            # Base64 encode the audio data
            audio_data = base64.b64encode(response.content).decode("ascii")

            return audio_data, "mp3"

        except Exception as e:
            logger.error(f"Error in Papla synthesis: {str(e)}")
            raise Exception(f"Papla synthesis error: {str(e)}")
