import os
import base64
import tempfile
import httpx
from loguru import logger
from typing import Dict, List, Tuple, Any
import asyncio
from .provider import TTSProvider
from .base import register_provider


@register_provider("lanternfish")
class LanternfishProvider(TTSProvider):
    _models = None

    @classmethod
    def _initialize_provider(cls):
        """Initialize the Lanternfish TTS provider"""
        try:
            # Set up available models
            cls._models = [
                {
                    "id": "lanternfish",
                    "name": "Lanternfish",
                    "description": "Lanternfish text-to-speech model",
                }
            ]
            logger.info("Successfully initialized Lanternfish TTS provider")
        except Exception as e:
            logger.error(f"Failed to initialize Lanternfish TTS provider: {str(e)}")
            raise ValueError(f"Lanternfish TTS initialization error: {str(e)}")

    @classmethod
    def get_available_models(cls) -> List[Dict[str, Any]]:
        """Get a list of available models for Lanternfish TTS"""
        if not cls.is_available() or not cls._models:
            return []

        return cls._models

    @classmethod
    async def synthesize(
        cls, text: str, model_id: str = None
    ) -> Tuple[str, str]:
        """Synthesize speech using Lanternfish TTS"""
        if not cls.is_available():
            raise ValueError("Lanternfish TTS provider is not available")

        # Default model is the only model
        if not model_id:
            model_id = "lanternfish-1"
            logger.info(f"No model specified for Lanternfish TTS, using default: {model_id}")

        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    os.getenv("LANTERNFISH_API_URL"),
                    json={
                        "reference_id": os.getenv("LANTERNFISH_REFERENCE_ID"),
                        "text": text,
                        "format": "mp3",
                    },
                    headers={
                        "api-key": os.getenv("LANTERNFISH_API_KEY"),
                        "model": os.getenv("LANTERNFISH_MODEL"),
                    },
                    timeout=30,
                )

            response.raise_for_status()

            audio_data = base64.b64encode(response.content).decode("ascii")
            return audio_data, "mp3"

        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error in Lanternfish TTS synthesis: {str(e)}, content: {e.response.text}")
            raise Exception(f"Lanternfish TTS synthesis error: HTTP error {e.response.status_code}")
        
        except Exception as e:
            logger.error(f"Error in Lanternfish TTS synthesis: {str(e)}")
            raise Exception(f"Lanternfish TTS synthesis error: {str(e)}")
