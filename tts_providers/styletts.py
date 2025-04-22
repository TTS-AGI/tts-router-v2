import os
import base64
from loguru import logger
from typing import Dict, List, Tuple, Any
from gradio_client import Client

from .provider import TTSProvider
from .base import register_provider


@register_provider("styletts")
class StyleTTSProvider(TTSProvider):
    _client = None
    _models = None

    @classmethod
    def _initialize_provider(cls):
        """Initialize the StyleTTS provider"""
        try:
            cls._client = Client(
                "TTS-AGI/styletts2-api", hf_token=os.getenv("HF_TOKEN")
            )

            # Set up available models
            cls._models = [
                {
                    "id": "styletts2",
                    "name": "StyleTTS 2",
                    "description": "StyleTTS 2 text-to-speech model",
                }
            ]
        except Exception as e:
            logger.error(f"Failed to initialize StyleTTS client: {str(e)}")
            raise ValueError(f"StyleTTS initialization error: {str(e)}")

    @classmethod
    def get_available_models(cls) -> List[Dict[str, Any]]:
        """Get a list of available models for StyleTTS"""
        if not cls.is_available() or not cls._models:
            return []

        return cls._models

    @classmethod
    async def synthesize(
        cls, text: str, model_id: str = None, steps: int = 3
    ) -> Tuple[str, str]:
        """Synthesize speech using StyleTTS"""
        if not cls.is_available():
            raise ValueError("StyleTTS provider is not available")

        # Default model is the only model
        if not model_id:
            model_id = "styletts2"
            logger.info(f"No model specified for StyleTTS, using default: {model_id}")

        try:
            # Call the StyleTTS API
            result = cls._client.predict(
                text=text, steps=steps, api_name="/ljsynthesize"
            )

            # Read the audio file and encode it as base64
            with open(result, "rb") as audio_file:
                audio_data = base64.b64encode(audio_file.read()).decode("ascii")

            return audio_data, "wav"

        except Exception as e:
            logger.error(f"Error in StyleTTS synthesis: {str(e)}")
            raise Exception(f"StyleTTS synthesis error: {str(e)}")
