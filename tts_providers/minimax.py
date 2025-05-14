import os
import json
import base64
import random
import httpx
from loguru import logger
from typing import Dict, List, Tuple, Any

from .provider import TTSProvider
from .base import register_provider

from dotenv import load_dotenv

load_dotenv()

MINIMAX_VOICES = [
    "English_Sweet_Female_4"
]


@register_provider("minimax")
class MinimaxProvider(TTSProvider):
    _api_key = None
    _group_id = None
    _base_url = "https://api.minimaxi.chat/v1/t2a_v2"
    _models = None

    @classmethod
    def _initialize_provider(cls):
        """Initialize the Minimax provider"""
        cls._api_key = os.getenv("MINIMAX_API_KEY")
        cls._group_id = os.getenv("MINIMAX_GROUP_ID")

        if not cls._api_key:
            logger.error("Minimax API key not found in environment variables")
            raise ValueError("MINIMAX_API_KEY environment variable is required")

        if not cls._group_id:
            logger.error("Minimax Group ID not found in environment variables")
            raise ValueError("MINIMAX_GROUP_ID environment variable is required")

        # Set up available models
        cls._models = [
            {
                "id": "speech-02-hd",
                "name": "Hailuo Speech 02 HD",
                "description": "Minimax's TTS model",
            },
            {
                "id": "speech-02-turbo",
                "name": "Hailuo Speech 02 Turbo",
                "description": "Minimax's TTS model",
            },
            {
                "id": "speech-01-hd",
                "name": "Hailuo Speech 01 HD",
                "description": "Minimax's TTS model",
            },
            {
                "id": "speech-01-turbo",
                "name": "Hailuo Speech 01 Turbo",
                "description": "Minimax's TTS model",
            },
        ]

    @classmethod
    def get_available_models(cls) -> List[Dict[str, Any]]:
        """Get a list of available models for Minimax"""
        if not cls.is_available() or not cls._models:
            return []

        return cls._models

    @classmethod
    async def synthesize(cls, text: str, model_id: str = None) -> Tuple[str, str]:
        """Synthesize speech using Minimax"""
        if not cls.is_available():
            raise ValueError("Minimax provider is not available")

        # Default model if none specified
        if not model_id:
            model_id = "speech-02-hd"
            logger.info(f"No model specified for Minimax, using default: {model_id}")

        # Select a random voice
        voice_id = random.choice(MINIMAX_VOICES)

        headers = {
            "Authorization": f"Bearer {cls._api_key}",
            "Content-Type": "application/json",
        }

        data = {
            "model": model_id,
            "text": text,
            "stream": False,
            "voice_setting": {"voice_id": voice_id, "speed": 1, "vol": 1, "pitch": 0},
            "audio_setting": {
                "sample_rate": 32000,
                "bitrate": 128000,
                "format": "mp3",
                "channel": 1,
            },
        }

        url = f"{cls._base_url}?GroupId={cls._group_id}"

        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    url,
                    headers=headers,
                    json=data,
                    timeout=30.0,
                )

                if response.status_code != 200:
                    logger.error(
                        f"Minimax API error: {response.status_code} - {response.text}"
                    )
                    raise Exception(
                        f"Minimax API error: {response.status_code} - {response.text}"
                    )

                # Parse the response
                response_data = response.json()

                if "data" not in response_data or "audio" not in response_data["data"]:
                    logger.error(
                        f"Unexpected response format from Minimax: {response_data}"
                    )
                    raise Exception("Unexpected response format from Minimax API")

                # Convert hex audio data to bytes and then base64
                audio_bytes = bytes.fromhex(response_data["data"]["audio"])
                audio_data = base64.b64encode(audio_bytes).decode("ascii")

                return audio_data, "mp3"

            except Exception as e:
                logger.error(f"Error in Minimax synthesis: {str(e)}")
                raise Exception(f"Minimax synthesis error: {str(e)}")
