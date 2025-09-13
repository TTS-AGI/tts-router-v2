import os
import base64
import httpx
import random
from loguru import logger
from typing import Dict, List, Tuple, Any
from .provider import TTSProvider
from .base import register_provider


@register_provider("veena")
class VeenaProvider(TTSProvider):
    _models = None
    _api_key = None
    _base_url = "https://flash.mayaresearch.ai"

    # Available Veena speakers based on Maya Research documentation
    _speakers = [
        "keerti_joy",
        "varun_chat",
        "soumya_calm",
        "mohini_whispers",
        "charu_soft",
    ]

    @classmethod
    def _initialize_provider(cls):
        """Initialize the Maya Research Veena TTS provider"""
        try:
            cls._api_key = os.getenv("VEENA_API_KEY")
            if not cls._api_key:
                logger.error("Veena API key not found in environment variables")
                raise ValueError("VEENA_API_KEY environment variable is required")

            # Set up available models based on speakers
            cls._models = []
            for speaker in cls._speakers:
                cls._models.append({
                    "id": speaker,
                    "name": speaker.replace("_", " ").title(),
                    "description": f"Maya Research Veena voice: {speaker}",
                })
            
            logger.info("Successfully initialized Maya Research Veena TTS provider")
        except Exception as e:
            logger.error(f"Failed to initialize Maya Research Veena TTS provider: {str(e)}")
            raise ValueError(f"Maya Research Veena TTS initialization error: {str(e)}")

    @classmethod
    def get_available_models(cls) -> List[Dict[str, Any]]:
        """Get a list of available models for Maya Research Veena TTS"""
        if not cls.is_available() or not cls._models:
            return []

        return cls._models

    @classmethod
    async def synthesize(
        cls, text: str, model_id: str = None
    ) -> Tuple[str, str]:
        """Synthesize speech using Maya Research Veena TTS"""
        if not cls.is_available():
            raise ValueError("Maya Research Veena TTS provider is not available")

        try:
            endpoint = f"{cls._base_url}/generate"

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    endpoint,
                    json={
                        "text": text,
                        "speaker_id": random.choice(cls._speakers) ,
                        "streaming": False,
                        "normalize": True,
                        "skip_text_validation": True,
                    },
                    headers={
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {cls._api_key}",
                    },
                    timeout=60,  # Longer timeout for TTS generation
                )

            response.raise_for_status()

            # Return base64 encoded audio data and extension
            audio_data = base64.b64encode(response.content).decode("ascii")
            return audio_data, "wav"

        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error in Maya Research Veena TTS synthesis: {str(e)}, content: {e.response.text}")
            raise Exception(f"Maya Research Veena TTS synthesis error: HTTP error {e.response.status_code}")
        
        except Exception as e:
            logger.error(f"Error in Maya Research Veena TTS synthesis: {str(e)}")
            raise Exception(f"Maya Research Veena TTS synthesis error: {str(e)}")