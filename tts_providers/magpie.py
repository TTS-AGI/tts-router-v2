import os
import base64
from loguru import logger
from typing import Dict, List, Tuple, Any
import requests
import tempfile
import numpy as np
import soundfile as sf

from .provider import TTSProvider
from .base import register_provider

from dotenv import load_dotenv

load_dotenv()


@register_provider("magpie")
class MagpieProvider(TTSProvider):
    _api_key = None
    _base_url = "https://nvidia-tts-arena-magpietts.hf.space"
    _models = None

    @classmethod
    def _initialize_provider(cls):
        """Initialize the Magpie provider"""
        try:
            # Get HF_TOKEN from environment
            cls._api_key = os.getenv("HF_TOKEN")
            if not cls._api_key:
                logger.warning(
                    "HF_TOKEN environment variable not set. Magpie provider may have limited access."
                )

            # Set up available models
            cls._models = [
                {
                    "id": "magpie",
                    "name": "Magpie",
                    "description": "Magpie text-to-speech model",
                }
            ]

            # Verify API is accessible
            response = requests.get(
                f"{cls._base_url}/",
                headers=(
                    {"Authorization": f"Bearer {cls._api_key}"} if cls._api_key else {}
                ),
            )
            if response.status_code != 200:
                logger.error(
                    f"Failed to connect to Magpie API: {response.status_code} - {response.text}"
                )
                raise ValueError(f"Magpie API connection error: {response.status_code}")

        except Exception as e:
            logger.error(f"Failed to initialize Magpie provider: {str(e)}")
            cls._models = None
            raise ValueError(f"Magpie initialization error: {str(e)}")

    @classmethod
    def get_available_models(cls) -> List[Dict[str, Any]]:
        """Get a list of available models for Magpie"""
        if not cls.is_available() or not cls._models:
            return []

        return cls._models

    @classmethod
    async def synthesize(cls, text: str, model_id: str = None) -> Tuple[str, str]:
        """Synthesize speech using Magpie"""
        if not cls.is_available():
            raise ValueError("Magpie provider is not available")

        # Default model is the only model
        if not model_id:
            model_id = "magpie"
            logger.info(f"No model specified for Magpie, using default: {model_id}")

        try:
            # Prepare headers with authorization if token is available
            headers = {"Content-Type": "application/json"}
            if cls._api_key:
                headers["Authorization"] = f"Bearer {cls._api_key}"

            # Call the Magpie API
            response = requests.post(
                f"{cls._base_url}/synthesize", headers=headers, json={"text": text}
            )

            if response.status_code != 200:
                logger.error(
                    f"Magpie API error: {response.status_code} - {response.text}"
                )
                raise Exception(
                    f"Magpie API error: {response.status_code} - {response.text}"
                )

            # Base64 encode the audio data
            audio_data = base64.b64encode(response.content).decode("ascii")

            return audio_data, "wav"

        except Exception as e:
            logger.error(f"Error in Magpie synthesis: {str(e)}")
            raise Exception(f"Magpie synthesis error: {str(e)}")
