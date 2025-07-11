import os
import base64
import httpx
from loguru import logger
from typing import Dict, List, Tuple, Any
from .provider import TTSProvider
from .base import register_provider


@register_provider("inworld")
class InworldProvider(TTSProvider):
    _models = [
        {
            "id": "mark",
            "name": "Mark",
            "voiceId": "Mark",
            "gender": "male",
        },
        {
            "id": "ashley",
            "name": "Ashley",
            "voiceId": "Ashley",
            "gender": "female",
        },
        {
            "id": "alex",
            "name": "Alex",
            "voiceId": "Alex",
            "gender": "male",
        },
        {
            "id": "theodore",
            "name": "Theodore",
            "voiceId": "Theodore",
            "gender": "male",
        },
        {
            "id": "deborah",
            "name": "Deborah",
            "voiceId": "Deborah",
            "gender": "female",
        },
        {
            "id": "sarah",
            "name": "Sarah",
            "voiceId": "Sarah",
            "gender": "female",
        },
        {
            "id": "edward",
            "name": "Edward",
            "voiceId": "Edward",
            "gender": "male",
        },
        {
            "id": "olivia",
            "name": "Olivia",
            "voiceId": "Olivia",
            "gender": "female",
        },
        {
            "id": "hades",
            "name": "Hades",
            "voiceId": "Hades",
            "gender": "male",
        },
        {
            "id": "elizabeth",
            "name": "Elizabeth",
            "voiceId": "Elizabeth",
            "gender": "female",
        },
    ]
    _api_url = "https://api.inworld.ai/tts/v1/voice"
    _api_key = os.getenv("INWORLD_API_KEY")
    _model_id = "inworld-tts-1"

    @classmethod
    def _initialize_provider(cls):
        """Initialize the Inworld TTS provider"""
        if not cls._api_key:
            logger.warning("INWORLD_API_KEY not set - Inworld provider will not be available")
            cls._available = False
        else:
            logger.info("Successfully initialized Inworld TTS provider")

    @classmethod
    def get_available_models(cls) -> List[Dict[str, Any]]:
        """Get a list of available models for Inworld TTS"""
        if not cls.is_available():
            return []
        return cls._models

    @classmethod
    async def synthesize(
        cls, text: str, model_id: str = None
    ) -> Tuple[str, str]:
        """Synthesize speech using Inworld TTS"""
        if not cls.is_available():
            raise ValueError("Inworld TTS provider is not available")

        # Autocycle/randomly select a voice for each generation, ignore model_id
        import random
        voice = random.choice(cls._models)
        voice_id = voice["voiceId"]

        payload = {
            "text": text,
            "voiceId": voice_id,
            "modelId": cls._model_id,
        }
        headers = {
            "Authorization": f"Basic {cls._api_key}",
            "Content-Type": "application/json",
        }

        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    cls._api_url,
                    json=payload,
                    headers=headers,
                    timeout=30,
                )
            response.raise_for_status()
            
            # The response should contain audio data
            audio_data = base64.b64encode(response.content).decode("ascii")
            return audio_data, "wav"

        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error in Inworld TTS synthesis: {str(e)}, content: {e.response.text}")
            raise Exception(f"Inworld TTS synthesis error: HTTP error {e.response.status_code}")
        except Exception as e:
            logger.error(f"Error in Inworld TTS synthesis: {str(e)}")
            raise Exception(f"Inworld TTS synthesis error: {str(e)}") 