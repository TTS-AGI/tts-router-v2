import os
import base64
import httpx
import random
from loguru import logger
from typing import Dict, List, Tuple, Any
from .provider import TTSProvider
from .base import register_provider


@register_provider("maya1")
class Maya1Provider(TTSProvider):
    _models = None
    _api_key = None
    _base_url = "https://v3.mayaresearch.ai/v1/tts"

    # Available Maya-1 voices
    _human_voices = [
        "Noah",
        "Ava",
        "Chloe",
        "Liam",
    ]

    _creative_voices = [
        "AnimatedCartoon",
        "Anime",
        "Flirty",
        "Seductively",
        "AIMachineVoice",
        "Cyborg",
        "AlienSciFi",
        "Pirate",
        "Gangster",
    ]

    _all_voices = _human_voices + _creative_voices

    @classmethod
    def _initialize_provider(cls):
        """Initialize the Maya Research Maya-1 TTS provider"""
        try:
            cls._api_key = os.getenv("MAYA1_API_KEY")
            if not cls._api_key:
                logger.error("Maya-1 API key not found in environment variables")
                raise ValueError("MAYA1_API_KEY environment variable is required")

            # Set up available models based on voices
            cls._models = []

            # Add human voices
            for voice in cls._human_voices:
                cls._models.append({
                    "id": voice,
                    "name": voice,
                    "description": f"Maya-1 Human Voice: {voice}",
                })

            # Add creative voices
            for voice in cls._creative_voices:
                cls._models.append({
                    "id": voice,
                    "name": voice,
                    "description": f"Maya-1 Creative Voice: {voice}",
                })

            logger.info("Successfully initialized Maya Research Maya-1 TTS provider")
        except Exception as e:
            logger.error(f"Failed to initialize Maya Research Maya-1 TTS provider: {str(e)}")
            raise ValueError(f"Maya Research Maya-1 TTS initialization error: {str(e)}")

    @classmethod
    def get_available_models(cls) -> List[Dict[str, Any]]:
        """Get a list of available models for Maya Research Maya-1 TTS"""
        if not cls.is_available() or not cls._models:
            return []

        return cls._models

    @classmethod
    async def synthesize(
        cls, text: str, model_id: str = None
    ) -> Tuple[str, str]:
        """Synthesize speech using Maya Research Maya-1 TTS"""
        if not cls.is_available():
            raise ValueError("Maya Research Maya-1 TTS provider is not available")

        try:
            endpoint = f"{cls._base_url}/generate"

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    endpoint,
                    json={
                        "text": text,
                        "voice_id": random.choice(cls._all_voices),
                        "stream": False,
                    },
                    headers={
                        "Content-Type": "application/json",
                        "X-API-Key": cls._api_key,
                    },
                    timeout=60,  # Longer timeout for TTS generation
                )

            response.raise_for_status()

            # Return base64 encoded audio data and extension
            audio_data = base64.b64encode(response.content).decode("ascii")
            return audio_data, "wav"

        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error in Maya Research Maya-1 TTS synthesis: {str(e)}, content: {e.response.text}")
            raise Exception(f"Maya Research Maya-1 TTS synthesis error: HTTP error {e.response.status_code}")

        except Exception as e:
            logger.error(f"Error in Maya Research Maya-1 TTS synthesis: {str(e)}")
            raise Exception(f"Maya Research Maya-1 TTS synthesis error: {str(e)}")
