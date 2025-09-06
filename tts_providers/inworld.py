import os
import base64
import httpx
from loguru import logger
from typing import Dict, List, Tuple, Any
from .provider import TTSProvider
from .base import register_provider


@register_provider("inworld")
class InworldProvider(TTSProvider):
    # Inworld voices (speakers) exposed in the arena UI
    _models = [
        {"id": "alex", "name": "Alex", "voiceId": "Alex", "gender": "male"},
        {"id": "olivia", "name": "Olivia", "voiceId": "Olivia", "gender": "female"},
        {"id": "mark", "name": "Mark", "voiceId": "Mark", "gender": "male"},
        {"id": "ashley", "name": "Ashley", "voiceId": "Ashley", "gender": "female"},
        {"id": "deborah", "name": "Deborah", "voiceId": "Deborah", "gender": "female"},
        {"id": "ronald", "name": "Ronald", "voiceId": "Ronald", "gender": "male"},
        {"id": "dennis", "name": "Dennis", "voiceId": "Dennis", "gender": "male"},
        {"id": "theodore", "name": "Theodore", "voiceId": "Theodore", "gender": "male"},
        {"id": "wendy", "name": "Wendy", "voiceId": "Wendy", "gender": "female"},
        {"id": "craig", "name": "Craig", "voiceId": "Craig", "gender": "male"},
    ]
    _api_url = "https://api.inworld.ai/tts/v1/voice"
    _api_key = os.getenv("INWORLD_API_KEY")
    # Default engine model; can be overridden per-request via model_id
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

        # Determine voice and engine model for this request
        import random

        # Map lowercase id -> voice entry for quick lookup
        voice_map = {v["id"].lower(): v for v in cls._models}

        # Default to random voice
        selected_voice = random.choice(cls._models)
        selected_voice_id = selected_voice["voiceId"]

        # Engine model selection: allow either default or MAX when explicitly requested
        engine_model_id = cls._model_id

        if model_id:
            mid = str(model_id).strip()

            # If the provided model_id is an engine selector, use it
            if mid in {"inworld-tts-1", "inworld-tts-1-max"}:
                engine_model_id = mid
            # Otherwise, if it matches a known voice id, select that voice explicitly
            elif mid.lower() in voice_map:
                selected_voice_id = voice_map[mid.lower()]["voiceId"]

        payload = {
            "text": text,
            "voiceId": selected_voice_id,
            "modelId": engine_model_id,
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
            
            # Parse JSON response to get audioContent
            response_data = response.json()
            if "audioContent" not in response_data:
                raise Exception("No audioContent in response")
            
            # The audioContent is already base64 encoded, so just return it
            audio_data = response_data["audioContent"]
            return audio_data, "wav"

        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error in Inworld TTS synthesis: {str(e)}, content: {e.response.text}")
            raise Exception(f"Inworld TTS synthesis error: HTTP error {e.response.status_code}")
        except Exception as e:
            logger.error(f"Error in Inworld TTS synthesis: {str(e)}")
            raise Exception(f"Inworld TTS synthesis error: {str(e)}") 
