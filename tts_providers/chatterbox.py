import os
import base64
import httpx
import random
from loguru import logger
from typing import Dict, List, Tuple, Any
from .provider import TTSProvider
from .base import register_provider


@register_provider("chatterbox")
class ChatterboxProvider(TTSProvider):
    _voices = [
        {
            "id": "voice1-male",
            "name": "Voice 1 (Male)",
            "voice_uuid": "4e228dba",
            "gender": "male",
        },
        {
            "id": "voice2-male",
            "name": "Voice 2 (Male)",
            "voice_uuid": "01bcc102",
            "gender": "male",
        },
        {
            "id": "voice3-female",
            "name": "Voice 3 (Female)",
            "voice_uuid": "ecbe5d97",
            "gender": "female",
        },
        {
            "id": "voice4-female",
            "name": "Voice 4 (Female)",
            "voice_uuid": "ae8223ca",
            "gender": "female",
        },
    ]
    _api_url = "https://p.cluster.resemble.ai/synthesize"
    _api_key = os.getenv("CHATTERBOX_API_KEY")

    @classmethod
    def _initialize_provider(cls):
        logger.info("ChatterboxProvider initialized.")

    @classmethod
    def get_available_models(cls) -> List[Dict[str, Any]]:
        # For compatibility, return voices as "models"
        return cls._voices

    @classmethod
    async def synthesize(
        cls, text: str, model_id: str = None
    ) -> Tuple[str, str]:
        # Autocycle/randomly select a voice for each generation, ignore model_id
        voice = random.choice(cls._voices)
        voice_uuid = voice["voice_uuid"]
        # Wrap text in SSML if not already
        if not text.strip().startswith("<speak"):
            ssml = f'<speak exaggeration="0.6">{text}</speak>'
        else:
            ssml = text
        payload = {
            "voice_uuid": voice_uuid,
            "data": ssml,
            "output_format": "wav",
        }
        headers = {
            "Authorization": cls._api_key,
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
            
            # Parse JSON response to get audio_content
            response_data = response.json()
            if "audio_content" not in response_data:
                raise Exception("No audio_content in response")
            
            # The audio_content is already base64 encoded, so just return it
            audio_data = response_data["audio_content"]
            return audio_data, "wav"
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error in Chatterbox TTS synthesis: {str(e)}, content: {e.response.text}")
            raise Exception(f"Chatterbox TTS synthesis error: HTTP error {e.response.status_code}")
        except Exception as e:
            logger.error(f"Error in Chatterbox TTS synthesis: {str(e)}")
            raise Exception(f"Chatterbox TTS synthesis error: {str(e)}")