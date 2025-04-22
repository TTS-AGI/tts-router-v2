import os
import requests
import base64
import random
from loguru import logger
from typing import Dict, List, Tuple, Any

from .provider import TTSProvider
from .base import register_provider

HUME_VOICES = [
    # "Classical Film Actor",
    "Vince Douglas",
    "Mysterious Woman",
    "Male English Actor",
    "Inspiring Woman",
    "Campfire Narrator",
    "TikTok Fashion Influencer",
    # "Colton Rivers",
    "Inspiring Man",
    # "Terrence Bentley",
    # "Ava Song",
    # "Alice Bennett",
    # "Sitcom Girl",
    # "Unserious Movie Trailer Narrator",
    # "Big Dicky",
    # "English Children's Book Narrator",
    "New York Comedian Guy",
    "Excitable British Naturalist",
    "Colorful Fashion Influencer",
]


@register_provider("hume")
class HumeProvider(TTSProvider):
    _api_key = None
    _base_url = "https://api.hume.ai/v0/tts/file"
    _models = None

    @classmethod
    def _initialize_provider(cls):
        """Initialize the Hume provider"""
        cls._api_key = os.getenv("HUME_API_KEY")
        if not cls._api_key:
            logger.error("Hume API key not found in environment variables")
            raise ValueError("HUME_KEY environment variable is required")

        # Set up available models
        cls._models = [
            # {
            #     "id": "male_english_actor",
            #     "name": "Male English Actor",
            #     "description": "Male English actor voice from Hume AI"
            # }
            "octave"
        ]

    @classmethod
    def get_available_models(cls) -> List[Dict[str, Any]]:
        """Get a list of available models for Hume"""
        if not cls.is_available() or not cls._models:
            return ["octave"]

        return cls._models

    @classmethod
    async def synthesize(cls, text: str, model_id: str = None) -> Tuple[str, str]:
        """Synthesize speech using Hume"""
        if not cls.is_available():
            raise ValueError("Hume provider is not available")

        # Default model is the only model
        # if not model_id:
        #     model_id = "male_english_actor"
        #     logger.info(f"No model specified for Hume, using default: {model_id}")

        try:
            response = requests.post(
                cls._base_url,
                headers={
                    "X-Hume-Api-Key": cls._api_key,
                    "Content-Type": "application/json",
                },
                json={
                    "utterances": [
                        {
                            "text": text,
                            "voice": {
                                # "name": "Male English Actor",
                                "name": random.choice(HUME_VOICES),
                                "provider": "HUME_AI",
                            },
                        }
                    ],
                    "format": {"type": "mp3"},
                    "num_generations": 1,
                },
            )

            if response.status_code != 200:
                logger.error(
                    f"Hume API error: {response.status_code} - {response.text}"
                )
                raise Exception(
                    f"Hume API error: {response.status_code} - {response.text}"
                )

            # Base64 encode the audio data
            audio_data = base64.b64encode(response.content).decode("ascii")

            return audio_data, "mp3"

        except Exception as e:
            logger.error(f"Error in Hume synthesis: {str(e)}")
            raise Exception(f"Hume synthesis error: {str(e)}")
