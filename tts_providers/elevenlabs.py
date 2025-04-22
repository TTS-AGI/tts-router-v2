import os
import httpx
import random
from loguru import logger
from typing import Dict, List, Tuple, Any
import base64

from .provider import TTSProvider
from .base import register_provider

from dotenv import load_dotenv

load_dotenv()

# Default female American voice for ElevenLabs
DEFAULT_VOICE_ID = "21m00Tcm4TlvDq8ikWAM"  # Rachel voice


@register_provider("elevenlabs")
class ElevenLabsProvider(TTSProvider):
    _api_key = None
    _base_url = "https://api.elevenlabs.io/v1"
    _models = None
    _voices = ["21m00Tcm4TlvDq8ikWAM"]

    @classmethod
    def _initialize_provider(cls):
        """Initialize the ElevenLabs provider"""
        cls._api_key = os.getenv("ELEVENLABS_API_KEY")
        if not cls._api_key:
            logger.error("ElevenLabs API key not found in environment variables")
            raise ValueError("ELEVENLABS_API_KEY environment variable is required")

        # Fetch available models
        try:
            cls._fetch_models()
            _voices = cls._get_voices()["voices"]
            cls._voices = [voice["voice_id"] for voice in _voices]
        except Exception as e:
            logger.error(f"Failed to fetch ElevenLabs models: {str(e)}")
            raise

    @classmethod
    def _get_voices(cls):
        """Get a list of available voices for ElevenLabs"""
        headers = {"xi-api-key": cls._api_key}
        response = httpx.get(f"{cls._base_url}/voices", headers=headers)
        response.raise_for_status()
        return response.json()

    @classmethod
    def _fetch_models(cls):
        """Fetch available models from ElevenLabs API"""
        headers = {"xi-api-key": cls._api_key}

        with httpx.Client() as client:
            response = client.get(f"{cls._base_url}/models", headers=headers)
            response.raise_for_status()

            try:
                data = response.json()
                # Handle different response structures
                if isinstance(data, list):
                    cls._models = data
                else:
                    cls._models = data.get("models", [])
            except Exception as e:
                logger.error(f"Failed to parse ElevenLabs models response: {str(e)}")
                cls._models = []

            if not cls._models:
                logger.warning("No models found for ElevenLabs")

    @classmethod
    def get_available_models(cls) -> List[Dict[str, Any]]:
        """Get a list of available models for ElevenLabs"""
        if not cls.is_available() or not cls._models:
            return []

        return [
            {
                "id": model["model_id"],
                "name": model["name"],
                "description": model.get("description", ""),
            }
            for model in cls._models
        ]

    @classmethod
    async def synthesize(cls, text: str, model_id: str = None) -> Tuple[str, str]:
        """Synthesize speech using ElevenLabs"""
        if not cls.is_available():
            raise ValueError("ElevenLabs provider is not available")

        if not model_id and cls._models:
            # Use the first model as default
            model_id = cls._models[0]["model_id"]
            logger.info(f"No model specified for ElevenLabs, using default: {model_id}")

        # Check if model exists
        model_exists = any(model["model_id"] == model_id for model in cls._models)
        if not model_exists:
            available_models = ", ".join(model["model_id"] for model in cls._models)
            logger.error(
                f"Model {model_id} not found. Available models: {available_models}"
            )
            raise ValueError(f"Model {model_id} not found for ElevenLabs provider")

        # Use default voice (American female)
        voice_id = random.choice(cls._voices)

        headers = {
            "xi-api-key": cls._api_key,
            "Content-Type": "application/json",
        }

        data = {
            "text": text,
            "model_id": model_id,
            "voice_settings": {"stability": 0.5, "similarity_boost": 0.75},
        }

        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{cls._base_url}/text-to-speech/{voice_id}",
                headers=headers,
                json=data,
                timeout=30.0,  # Longer timeout for TTS generation
            )

            if response.status_code != 200:
                logger.error(
                    f"ElevenLabs API error: {response.status_code} - {response.text}"
                )
                raise Exception(
                    f"ElevenLabs API error: {response.status_code} - {response.text}"
                )

            # Base64 encode the audio data to handle binary data safely
            audio_data = base64.b64encode(response.content).decode("ascii")

            # Return base64 encoded audio data and MIME type
            return audio_data, "mp3"
