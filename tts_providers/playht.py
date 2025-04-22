import os
import httpx
import json
from loguru import logger
from typing import Dict, List, Tuple, Any
import base64
import random

from .provider import TTSProvider
from .base import register_provider

from dotenv import load_dotenv

load_dotenv()

# Default female American voice for PlayHT
# DEFAULT_VOICE_ID = "s3://voice-cloning-zero-shot/d9ff78ba-d016-47f6-b0ef-dd630f59414e/female-cs/manifest.json"  # Jennifer voice

with open(os.path.join(os.path.dirname(__file__), "play_voices.txt"), "r") as f:
    PLAY_VOICES = f.read().splitlines()


@register_provider("playht")
class PlayHTProvider(TTSProvider):
    _api_key = None
    _user_id = None
    _base_url = "https://api.play.ht/api/v2"
    _models = None

    @classmethod
    def _initialize_provider(cls):
        """Initialize the PlayHT provider"""
        cls._api_key = os.getenv("PLAYHT_API_KEY")
        cls._user_id = os.getenv("PLAYHT_USER_ID")

        if not cls._api_key or not cls._user_id:
            logger.error("PlayHT API key or User ID not found in environment variables")
            raise ValueError(
                "PLAYHT_API_KEY and PLAYHT_USER_ID environment variables are required"
            )

        # Fetch available models
        try:
            cls._fetch_models()
        except Exception as e:
            logger.error(f"Failed to fetch PlayHT models: {str(e)}")
            raise

    @classmethod
    def _fetch_models(cls):
        """Fetch available models from PlayHT API"""
        headers = {
            "accept": "application/json",
            "AUTHORIZATION": cls._api_key,
            "X-USER-ID": cls._user_id,
        }

        # Set hardcoded models based on PlayHT documentation
        # As per https://docs.play.ht/reference/models
        cls._models = [
            {
                "id": "PlayDialog",
                "name": "PlayDialog",
                "description": "PlayHT's latest voice model built for fluid, emotive conversation",
            },
            {
                "id": "PlayHT2.0",
                "name": "PlayHT2.0 Turbo",
                "description": "PlayHT's legacy voice model",
            },
        ]

        try:
            # Still attempt to fetch voices to validate API credentials
            with httpx.Client() as client:
                response = client.get(f"{cls._base_url}/voices", headers=headers)
                response.raise_for_status()
                logger.info("Successfully validated PlayHT API credentials")
        except Exception as e:
            logger.warning(
                f"Could not fetch PlayHT voices: {str(e)}, but will continue with hardcoded models"
            )

    @classmethod
    def get_available_models(cls) -> List[Dict[str, Any]]:
        """Get a list of available models for PlayHT"""
        if not cls.is_available() or not cls._models:
            return []

        return cls._models

    @classmethod
    async def synthesize(cls, text: str, model_id: str = None) -> Tuple[str, str]:
        """Synthesize speech using PlayHT"""
        if not cls.is_available():
            raise ValueError("PlayHT provider is not available")

        if not model_id and cls._models:
            # Use PlayDialog as default
            model_id = "PlayDialog"
            logger.info(f"No model specified for PlayHT, using default: {model_id}")

        # Check if model exists
        model_exists = any(model["id"] == model_id for model in cls._models)
        if not model_exists:
            available_models = ", ".join(model["id"] for model in cls._models)
            logger.error(
                f"Model {model_id} not found. Available models: {available_models}"
            )
            raise ValueError(f"Model {model_id} not found for PlayHT provider")

        # Use default voice (American female)
        voice_id = random.choice(PLAY_VOICES)

        headers = {
            "accept": "*/*",
            "content-type": "application/json",
            "AUTHORIZATION": cls._api_key,
            "X-USER-ID": cls._user_id,
        }

        data = {
            "text": text,
            "voice": voice_id,
            "output_format": "mp3",
            "voice_engine": model_id,
        }

        async with httpx.AsyncClient() as client:
            # Use the streaming endpoint directly
            try:
                response = await client.post(
                    f"{cls._base_url}/tts/stream",
                    headers=headers,
                    json=data,
                    timeout=30.0,
                )

                if response.status_code != 200:
                    logger.error(
                        f"PlayHT API error during streaming: {response.status_code} - {response.text}"
                    )
                    raise Exception(
                        f"PlayHT API error: {response.status_code} - {response.text}"
                    )

                # Base64 encode the audio data
                audio_data = base64.b64encode(response.content).decode("ascii")

                return audio_data, "mp3"

            except Exception as e:
                logger.warning(f"PlayHT streaming API failed: {str(e)}")
                raise Exception(f"PlayHT streaming API error: {str(e)}")
