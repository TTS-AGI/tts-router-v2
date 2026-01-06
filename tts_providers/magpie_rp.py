import os
import base64
from loguru import logger
from typing import Dict, List, Tuple, Any
import requests

from .provider import TTSProvider
from .base import register_provider

from dotenv import load_dotenv

load_dotenv()


@register_provider("magpie-rp")
class MagpieRPProvider(TTSProvider):
    _api_key = None
    _base_url = "https://nvidia-tts-arena-magpietts-server.hf.space"
    _models = None
    _voices = ["mia", "aria", "leo", "jason", "sofia"]

    @classmethod
    def _initialize_provider(cls):
        """Initialize the Magpie-RP provider"""
        try:
            # Get HF_TOKEN from environment
            cls._api_key = os.getenv("HF_TOKEN")
            if not cls._api_key:
                logger.warning(
                    "HF_TOKEN environment variable not set. Magpie-RP provider may have limited access."
                )

            # Set up available models with voice support
            cls._models = [
                {
                    "id": "magpietts_research",
                    "name": "Magpie RP Research",
                    "description": "Magpie text-to-speech model with role-play capabilities",
                }
            ]

            # Verify API is accessible
            response = requests.get(
                f"{cls._base_url}/",
                headers=(
                    {"Authorization": f"Bearer {cls._api_key}"} if cls._api_key else {}
                ),
                timeout=10,
            )
            if response.status_code != 200:
                logger.error(
                    f"Failed to connect to Magpie-RP API: {response.status_code} - {response.text}"
                )
                raise ValueError(f"Magpie-RP API connection error: {response.status_code}")

        except Exception as e:
            logger.error(f"Failed to initialize Magpie-RP provider: {str(e)}")
            cls._models = None
            raise ValueError(f"Magpie-RP initialization error: {str(e)}")

    @classmethod
    def get_available_models(cls) -> List[Dict[str, Any]]:
        """Get a list of available models for Magpie-RP"""
        if not cls.is_available() or not cls._models:
            return []

        return cls._models

    @classmethod
    def get_available_voices(cls) -> List[str]:
        """Get a list of available voices for Magpie-RP"""
        if not cls.is_available():
            return []

        return cls._voices

    @classmethod
    async def synthesize(
        cls,
        text: str,
        model_id: str = None,
        voice: str = None,
        context_type: str = "text",
    ) -> Tuple[str, str]:
        """Synthesize speech using Magpie-RP

        Args:
            text: The text to synthesize
            model_id: The model to use (default: "magpietts_research")
            voice: The voice to use (one of: mia, aria, leo, jason, sofia)
            context_type: Either "text" or "audio" (default: "text")

        Returns:
            Tuple of (base64_encoded_audio, format)
        """
        if not cls.is_available():
            raise ValueError("Magpie-RP provider is not available")

        # Default model
        if not model_id:
            model_id = "magpietts_research"
            logger.info(f"No model specified for Magpie-RP, using default: {model_id}")

        # Validate voice if provided
        if voice and voice not in cls._voices:
            logger.warning(
                f"Invalid voice '{voice}'. Available voices: {cls._voices}. Using random."
            )
            voice = None

        # Validate context_type
        if context_type not in ["text", "audio"]:
            logger.warning(
                f"Invalid context_type '{context_type}'. Using 'text' as default."
            )
            context_type = "text"

        try:
            # Prepare headers with authorization if token is available
            headers = {"Content-Type": "application/json"}
            if cls._api_key:
                headers["Authorization"] = f"Bearer {cls._api_key}"

            # Prepare request payload
            payload = {
                "text": text,
                "model": model_id,
                "context_type": context_type,
            }

            # Add voice if specified
            if voice:
                payload["voice"] = voice

            # Call the Magpie-RP API
            response = requests.post(
                f"{cls._base_url}/synthesize",
                headers=headers,
                json=payload,
                timeout=60,
            )

            if response.status_code != 200:
                logger.error(
                    f"Magpie-RP API error: {response.status_code} - {response.text}"
                )
                raise Exception(
                    f"Magpie-RP API error: {response.status_code} - {response.text}"
                )

            # Base64 encode the audio data
            audio_data = base64.b64encode(response.content).decode("ascii")

            return audio_data, "wav"

        except Exception as e:
            logger.error(f"Error in Magpie-RP synthesis: {str(e)}")
            raise Exception(f"Magpie-RP synthesis error: {str(e)}")
