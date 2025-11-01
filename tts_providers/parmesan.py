import os
import httpx
import base64
import io
import wave
from loguru import logger
from typing import Dict, List, Tuple, Any
import numpy as np

from .provider import TTSProvider
from .base import register_provider

from dotenv import load_dotenv

load_dotenv()

PARMESAN_VOICES = [
    "grant"
]


@register_provider("parmesan")
class ParmesanProvider(TTSProvider):
    _api_key = None
    _base_url = None
    _models = None

    @classmethod
    def _initialize_provider(cls):
        """Initialize the Parmesan provider"""
        cls._api_key = os.getenv("PARMESAN_API_KEY")
        cls._base_url = os.getenv("PARMESAN_BASE_URL", "https://api.phonic.co/v1/tts")

        if not cls._api_key:
            logger.error("Parmesan API key not found in environment variables")
            raise ValueError("PARMESAN_API_KEY environment variable is required")

        # Set up available models
        cls._models = [
            {
                "id": "parmesan-base",
                "name": "Parmesan Base Model",
                "description": "High-quality text-to-speech synthesis",
            },
        ]

    @classmethod
    def get_available_models(cls) -> List[Dict[str, Any]]:
        """Get a list of available models for Parmesan"""
        if not cls.is_available() or not cls._models:
            return []

        return cls._models

    @classmethod
    async def synthesize(cls, text: str, model_id: str = None) -> Tuple[str, str]:
        """Synthesize speech using Parmesan"""
        if not cls.is_available():
            raise ValueError("Parmesan provider is not available")

        # Default model if none specified
        if not model_id:
            model_id = "parmesan-base"
            logger.info(f"No model specified for Parmesan, using default: {model_id}")

        # Use the first voice (can be randomized like other providers if needed)
        voice_id = PARMESAN_VOICES[0]

        headers = {
            "Authorization": f"Bearer {cls._api_key}",
        }

        json_payload = {
            "text": text,
            "voice_id": voice_id,
            "output_format": "pcm_44100"
        }

        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    cls._base_url,
                    headers=headers,
                    json=json_payload,
                    timeout=30.0,
                )

                if response.status_code != 200:
                    logger.error(
                        f"Parmesan API error: {response.status_code} - {response.text}"
                    )
                    raise Exception(
                        f"Parmesan API error: {response.status_code} - {response.text}"
                    )

                # Parse the response
                response_data = response.json()

                if "audio" not in response_data:
                    logger.error(
                        f"Unexpected response format from Parmesan: {response_data}"
                    )
                    raise Exception("Unexpected response format from Parmesan API")

                # The audio is base64 encoded PCM data
                audio_b64 = response_data["audio"]
                
                # Decode base64 to bytes
                audio_bytes = base64.b64decode(audio_b64)
                
                # Convert bytes to numpy array
                audio_np = np.frombuffer(audio_bytes, dtype=np.int16)
                
                # Convert PCM to WAV format
                wav_buffer = io.BytesIO()
                with wave.open(wav_buffer, 'wb') as wav_file:
                    wav_file.setnchannels(1)  # Mono
                    wav_file.setsampwidth(2)  # 16-bit
                    wav_file.setframerate(44100)  # 44.1kHz
                    wav_file.writeframes(audio_np.tobytes())
                
                # Get WAV data and encode to base64
                wav_data = wav_buffer.getvalue()
                wav_b64 = base64.b64encode(wav_data).decode('utf-8')
                
                return wav_b64, "wav"

            except Exception as e:
                logger.error(f"Error in Parmesan synthesis: {str(e)}")
                raise Exception(f"Parmesan synthesis error: {str(e)}")
