import os
import base64
import random
import httpx
from loguru import logger
from typing import Dict, List, Tuple, Any
from .provider import TTSProvider
from .base import register_provider


@register_provider("wordcab")
class WordcabProvider(TTSProvider):
    _models = None

    # Available Wordcab voices
    _voices = [
        "zhanna_call_sample_28s",
        "sheena_youtube_sample_18s", 
        "kesley_zoom_sample_36s",
        "derick_clip_001_15s_001",
        "chris_clip_001_15s_001",
    ]

    @classmethod
    def _initialize_provider(cls):
        """Initialize the Wordcab TTS provider"""
        try:
            # Set up available models based on voices
            cls._models = []
            for voice in cls._voices:
                cls._models.append({
                    "id": voice,
                    "name": voice.replace("_", " ").title(),
                    "description": f"Wordcab voice: {voice}",
                })
            logger.info("Successfully initialized Wordcab TTS provider")
        except Exception as e:
            logger.error(f"Failed to initialize Wordcab TTS provider: {str(e)}")
            raise ValueError(f"Wordcab TTS initialization error: {str(e)}")

    @classmethod
    def get_available_models(cls) -> List[Dict[str, Any]]:
        """Get a list of available models for Wordcab TTS"""
        if not cls.is_available() or not cls._models:
            return []

        return cls._models

    @classmethod
    async def synthesize(
        cls, text: str, model_id: str = None
    ) -> Tuple[str, str]:
        """Synthesize speech using Wordcab TTS"""
        if not cls.is_available():
            raise ValueError("Wordcab TTS provider is not available")

        # Choose random voice if no model specified, otherwise accept any model
        if not model_id:
            model_id = random.choice(cls._voices)
            logger.info(f"No model specified for Wordcab TTS, using random voice: {model_id}")
        else:
            logger.info(f"Using specified model for Wordcab TTS: {model_id}")

        try:
            # Get API URL, default to the reference URL if not set
            api_url = os.getenv("WORDCAB_API_URL")
            endpoint = f"{api_url}/v1/audio/speech"

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    endpoint,
                    json={
                        "input": text,
                        "voice": model_id,
                    },
                    headers={
                        "Content-Type": "application/json"
                    },
                    timeout=30,
                )

            response.raise_for_status()

            # Return base64 encoded audio data and extension
            audio_data = base64.b64encode(response.content).decode("ascii")
            return audio_data, "wav"

        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error in Wordcab TTS synthesis: {str(e)}, content: {e.response.text}")
            raise Exception(f"Wordcab TTS synthesis error: HTTP error {e.response.status_code}")
        
        except Exception as e:
            logger.error(f"Error in Wordcab TTS synthesis: {str(e)}")
            raise Exception(f"Wordcab TTS synthesis error: {str(e)}")
