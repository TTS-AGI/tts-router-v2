import os
import httpx
import base64
import tempfile
from loguru import logger
from typing import Dict, List, Tuple, Any

from .provider import TTSProvider
from .base import register_provider

from dotenv import load_dotenv

load_dotenv()


@register_provider("cosyvoice")
class CosyVoiceProvider(TTSProvider):
    _hf_token = None
    _base_url = "https://tts-agi-cosyvoice2-0-5b.hf.space"
    _models = None

    @classmethod
    def _initialize_provider(cls):
        """Initialize the CosyVoice provider"""
        cls._hf_token = os.getenv("HF_TOKEN")
        if not cls._hf_token:
            logger.error("Hugging Face token not found in environment variables")
            raise ValueError("HF_TOKEN environment variable is required")

        # Set up available models
        cls._models = [
            {
                "id": "cosyvoice-2.0.5b",
                "name": "CosyVoice 2.0.5b",
                "description": "CosyVoice text-to-speech model",
            }
        ]

    @classmethod
    def get_available_models(cls) -> List[Dict[str, Any]]:
        """Get a list of available models for CosyVoice"""
        if not cls.is_available() or not cls._models:
            return []

        return cls._models

    @classmethod
    async def synthesize(
        cls, text: str, model_id: str = None, seed: int = 42
    ) -> Tuple[str, str]:
        """Synthesize speech using CosyVoice"""
        if not cls.is_available():
            raise ValueError("CosyVoice provider is not available")

        # Default model is the only model
        if not model_id:
            model_id = "cosyvoice-2.0.5b"
            logger.info(f"No model specified for CosyVoice, using default: {model_id}")

        headers = {
            "Authorization": f"Bearer {cls._hf_token}",
            "Content-Type": "application/json",
        }

        payload = {"text": text, "seed": seed}

        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{cls._base_url}/generate",
                    headers=headers,
                    json=payload,
                    timeout=60.0,  # Longer timeout for TTS generation
                )

                if response.status_code != 200:
                    logger.error(
                        f"CosyVoice API error: {response.status_code} - {response.text}"
                    )
                    raise Exception(
                        f"CosyVoice API error: {response.status_code} - {response.text}"
                    )

                # Base64 encode the audio data to handle binary data safely
                audio_data = base64.b64encode(response.content).decode("ascii")

                return audio_data, "wav"

        except Exception as e:
            logger.error(f"Error in CosyVoice synthesis: {str(e)}")
            raise Exception(f"CosyVoice synthesis error: {str(e)}")
