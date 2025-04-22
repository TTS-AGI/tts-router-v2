import os
import base64
import tempfile
import httpx
from loguru import logger
from typing import Dict, List, Tuple, Any
import asyncio
from gradio_client import Client, handle_file
import random
from .provider import TTSProvider
from .base import register_provider


@register_provider("spark")
class SparkProvider(TTSProvider):
    _models = None
    _client = None

    @classmethod
    def _initialize_provider(cls):
        """Initialize the Spark TTS provider"""
        try:
            # Set up available models
            cls._models = [
                {
                    "id": "spark-tts",
                    "name": "Spark TTS",
                    "description": "Spark text-to-speech model with voice cloning capabilities",
                }
            ]
            # Initialize gradio client
            cls._client = Client("Mobvoi/Offical-Spark-TTS")
            logger.info("Successfully initialized Spark TTS provider")
        except Exception as e:
            logger.error(f"Failed to initialize Spark TTS provider: {str(e)}")
            raise ValueError(f"Spark TTS initialization error: {str(e)}")

    @classmethod
    def get_available_models(cls) -> List[Dict[str, Any]]:
        """Get a list of available models for Spark TTS"""
        if not cls.is_available() or not cls._models:
            return []

        return cls._models

    @classmethod
    async def synthesize(
        cls, text: str, model_id: str = None, reference_audio: str = None
    ) -> Tuple[str, str]:
        """Synthesize speech using Spark TTS with voice cloning"""
        if not cls.is_available():
            raise ValueError("Spark TTS provider is not available")

        # Default model is the only model
        if not model_id:
            model_id = "spark-tts"
            logger.info(f"No model specified for Spark TTS, using default: {model_id}")

        # Use a default reference audio if none provided
        reference_audio_url = (
            reference_audio
            or random.choice(
                [
                    "https://files.mrfake.name/api/file/files/nanospeech-voices/celeste.wav",
                    "https://files.mrfake.name/api/file/files/nanospeech-voices/nash.wav",
                    "https://files.mrfake.name/api/file/files/nanospeech-voices/orion.wav",
                    "https://files.mrfake.name/api/file/files/nanospeech-voices/rhea.wav",
                ]
            )
        )

        try:
            # We need to run the gradio client in a thread pool since it's synchronous
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                lambda: cls._client.predict(
                    text=text,
                    prompt_text="",
                    prompt_wav_upload=handle_file(reference_audio_url),
                    prompt_wav_record=None,
                    api_name="/voice_clone"
                )
            )
            
            logger.info(f"Spark TTS synthesis result: {result}")
            
            if not result:
                raise Exception("No result returned from Spark TTS API")
                
            # The result should be the path to the generated audio file
            audio_path = result
            
            with open(audio_path, "rb") as f:
                audio_data = base64.b64encode(f.read()).decode("ascii")
                
            return audio_data, "wav"

        except Exception as e:
            logger.error(f"Error in Spark TTS synthesis: {str(e)}")
            raise Exception(f"Spark TTS synthesis error: {str(e)}")
