import os
import io
import httpx
import random
import base64
from loguru import logger
from typing import Dict, List, Tuple, Any
from pydub import AudioSegment

from .provider import TTSProvider
from .base import register_provider

from dotenv import load_dotenv
load_dotenv()

TONTAUBE_VOICES = [
    ("malcom", "Malcom"),
    ("harvey", "Harvey"),
    ("barry", "Barry"),
    ("miles", "Miles"),
    ("jonny", "Jonny"),
    ("evie", "Evie"),
    ("sahra", "Sahra"),
]


@register_provider("tontaube")
class TontaubeProvider(TTSProvider):
    _api_key = None
    _base_url = "https://api.tontaube.ai/tts/arena"
    _voices = TONTAUBE_VOICES

    @classmethod
    def _initialize_provider(cls):
        """Initialize the Tontaube provider"""
        cls._api_key = os.getenv("TONTAUBE_API_KEY")
        if not cls._api_key:
            logger.error("Tontaube API key not found in environment variables")
            raise ValueError("TONTAUBE_API_KEY environment variable is required")

        logger.info(f"Tontaube provider initialized with {len(cls._voices)} voices")

    @classmethod
    def get_available_models(cls) -> List[Dict[str, Any]]:
        """Get a list of available models (voices) for Tontaube"""
        if not cls.is_available():
            return []

        return [
            {
                "id": voice_id,
                "name": voice_name,
                "description": f"Tontaube voice: {voice_name}",
            }
            for voice_id, voice_name in cls._voices
        ]

    @classmethod
    async def synthesize(cls, text: str, model_id: str = None) -> Tuple[str, str]:
        """Synthesize speech using Tontaube API"""
        if not cls.is_available():
            raise ValueError("Tontaube provider is not available")

        # Select voice
        valid_ids = [v[0] for v in cls._voices]
        if model_id and model_id in valid_ids:
            voice_id = model_id
        else:
            voice_id = random.choice(valid_ids)

        headers = {
            "Content-Type": "application/json",
            "X-Api-Key": cls._api_key,
        }

        payload = {
            "text": text,
            "voice": voice_id,
        }

        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    cls._base_url,
                    headers=headers,
                    json=payload,
                    timeout=60.0,
                )

                if response.status_code != 200:
                    logger.error(
                        f"Tontaube API error: {response.status_code} - {response.text}"
                    )
                    raise Exception(
                        f"Tontaube API error: {response.status_code} - {response.text}"
                    )

                # Response is Opus in MP4 container — convert to WAV
                audio = AudioSegment.from_file(io.BytesIO(response.content), format="mp4")
                wav_buffer = io.BytesIO()
                audio.export(wav_buffer, format="wav")
                audio_b64 = base64.b64encode(wav_buffer.getvalue()).decode("ascii")
                return audio_b64, "wav"

            except Exception as e:
                logger.error(f"Error in Tontaube synthesis: {str(e)}")
                raise Exception(f"Tontaube synthesis error: {str(e)}")
