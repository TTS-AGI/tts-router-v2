import os
import httpx
import base64
from loguru import logger
from typing import Dict, List, Tuple, Any
from random import choice

from .provider import TTSProvider
from .base import register_provider

from dotenv import load_dotenv
load_dotenv()

VOCU_VOICES = [
    "52f3c95d-ea96-4e4a-8c79-5a1a0aaf5186",  # Ruby
    "4ba81871-0b4b-4bee-a483-49491f86240a",  # Piper
    "1aa3658c-ca34-4d50-822c-323a349fd498",  # Alistair
    "2b65195c-9221-40b8-badc-27f66222b1bb",  # David
    "b19e9f03-73cc-44f1-b990-5681c621894a",  # Scarlett
]


@register_provider("vocu")
class VocuProvider(TTSProvider):
    _api_key = None
    _base_url = None
    _models = None

    @classmethod
    def _initialize_provider(cls):
        """Initialize the Vocu provider"""
        cls._api_key = os.getenv("VOCU_API_KEY")
        cls._base_url = os.getenv("VOCU_BASE_URL", "https://v1.vocu.ai/api/tts/simple-generate")

        if not cls._api_key:
            logger.error("Vocu API key not found in environment variables")
            raise ValueError("VOCU_API_KEY environment variable is required")

        # Set up available models
        cls._models = [
            {
                "id": "vocu-balance",
                "name": "Vocu Balance",
                "description": "Balanced quality and speed text-to-speech synthesis",
            },
        ]

    @classmethod
    def get_available_models(cls) -> List[Dict[str, Any]]:
        """Get a list of available models for Vocu"""
        if not cls.is_available() or not cls._models:
            return []

        return cls._models

    @classmethod
    async def synthesize(cls, text: str, model_id: str = None) -> Tuple[str, str]:
        """Synthesize speech using Vocu"""
        if not cls.is_available():
            raise ValueError("Vocu provider is not available")

        # Default model if none specified
        if not model_id:
            model_id = "vocu-balance"
            logger.info(f"No model specified for Vocu, using default: {model_id}")

        # Only balance preset is available
        preset = "balance"

        # Randomly select a voice
        voice_id = choice(VOCU_VOICES)

        headers = {
            "Authorization": f"Bearer {cls._api_key}",
            "Content-Type": "application/json",
        }

        json_payload = {
            "voiceId": voice_id,
            "text": text,
            "preset": preset,
            "language": "en",
            "break_clone": True,
            "flash": False,
            "stream": False,
        }

        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    cls._base_url,
                    headers=headers,
                    json=json_payload,
                    timeout=30.0,
                )

                if response.status_code != 200:
                    logger.error(
                        f"Vocu API error: {response.status_code} - {response.text}"
                    )
                    raise Exception(
                        f"Vocu API error: {response.status_code} - {response.text}"
                    )

                # Parse the response
                response_data = response.json()

                if response_data.get("status") != 200 or "data" not in response_data:
                    logger.error(
                        f"Unexpected response format from Vocu: {response_data}"
                    )
                    raise Exception("Unexpected response format from Vocu API")

                # The audio is a URL that we need to download
                audio_url = response_data["data"]["audio"]
                
                logger.info(f"Downloading audio from Vocu: {audio_url}")
                
                # Download the audio file
                audio_response = await client.get(audio_url, timeout=30.0)
                
                if audio_response.status_code != 200:
                    logger.error(
                        f"Failed to download audio from Vocu: {audio_response.status_code}"
                    )
                    raise Exception(
                        f"Failed to download audio from Vocu: {audio_response.status_code}"
                    )
                
                # Get the audio bytes and encode to base64
                audio_bytes = audio_response.content
                audio_b64 = base64.b64encode(audio_bytes).decode('utf-8')
                
                return audio_b64, "mp3"

            except Exception as e:
                logger.error(f"Error in Vocu synthesis: {str(e)}")
                raise Exception(f"Vocu synthesis error: {str(e)}")

