import base64
import os
import random
import tempfile
from typing import Any, Dict, List, Tuple

from audiosample import AudioSample
from deepdub import DeepdubClient
from dotenv import load_dotenv
from loguru import logger

from .base import register_provider
from .provider import TTSProvider

load_dotenv()

DEEPDUB_VOICES = [
    {
        "voice_name": "Samuel Gray",
        "voice_id": "26c5f982-e80b-4252-b4c2-bd7e118fcd72_prompt-reading-neutral",
    },
    {
        "voice_name": "Roy Rivera",
        "voice_id": "776aa833-fc77-4eac-9203-13da9021030c_prompt-reading-neutral",
    },
    {
        "voice_name": "Terry Wood",
        "voice_id": "8a12db1b-6c6d-4474-b8cf-3c0c2d3b105e_reading-neutral",
    },
    {
        "voice_name": "Heather Long",
        "voice_id": "337e4733-acc7-4bb8-aef8-6e9404c8b874",
    },
    {
        "voice_name": "Anne Phillips",
        "voice_id": "33f02485-049f-4436-b6b7-0aaa7c7ff5d5_reading-neutral",
    },
    {
        "voice_name": "Denise Cox",
        "voice_id": "b532c72a-662a-41b7-8470-68c34181b734_reading-neutral",
    },
]


@register_provider("deepdub")
class DeepdubProvider(TTSProvider):
    _api_key = None
    _client = None
    _models = None
    _voices = DEEPDUB_VOICES

    @classmethod
    def _initialize_provider(cls):
        """Initialize the Deepdub provider."""
        cls._api_key = os.getenv("DEEPDUB_API_KEY")
        if not cls._api_key:
            logger.error("Deepdub API key not found in environment variables")
            raise ValueError("DEEPDUB_API_KEY environment variable is required")

        cls._client = DeepdubClient(api_key=cls._api_key)
        cls._models = [
            {
                "id": "dd-etts-3.2",
                "name": "Deepdub ETTS 3.2",
                "description": "Deepdub expressive text-to-speech model",
            }
        ]

    @classmethod
    def get_available_models(cls) -> List[Dict[str, Any]]:
        """Get a list of available models for Deepdub."""
        if not cls.is_available() or not cls._models:
            return []

        return cls._models

    @classmethod
    async def synthesize(cls, text: str, model_id: str = None) -> Tuple[str, str]:
        """Synthesize speech using Deepdub."""
        if not cls.is_available():
            raise ValueError("Deepdub provider is not available")

        if not model_id:
            model_id = "dd-etts-3.2"
            logger.info(f"No model specified for Deepdub, using default: {model_id}")

        if model_id not in {model["id"] for model in cls._models}:
            available_models = ", ".join(model["id"] for model in cls._models)
            logger.error(
                f"Model {model_id} not found. Available models: {available_models}"
            )
            raise ValueError(f"Model {model_id} not found for Deepdub provider")

        voice = random.choice(cls._voices)
        voice_prompt_id = voice["voice_id"]
        logger.info(
            f"Using Deepdub voice prompt: {voice['voice_name']} ({voice_prompt_id})"
        )

        try:
            collection = AudioSample()
            async with cls._client.async_connect() as connection:
                async for chunk in connection.async_tts(
                    text=text,
                    voice_prompt_id=voice_prompt_id,
                    model=model_id,
                    locale="en-US",
                    variance=0.75,
                    tempo=None,
                    temperature=1.0,
                    prompt_boost=True,
                    accent_base_locale="en-US",
                    accent_locale="en-US",
                    accent_ratio=0.75,
                ):
                    collection += AudioSample(chunk)

            with tempfile.NamedTemporaryFile(suffix=".wav") as output_file:
                collection.write(output_file.name)
                output_file.seek(0)
                audio_data = output_file.read()

            return base64.b64encode(audio_data).decode("ascii"), "wav"

        except Exception as e:
            logger.error(f"Error in Deepdub synthesis: {str(e)}")
            raise Exception(f"Deepdub synthesis error: {str(e)}")
