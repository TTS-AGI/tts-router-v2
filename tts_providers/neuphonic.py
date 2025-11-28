import os
import httpx
import random
import json
import base64
from loguru import logger
from typing import Dict, List, Tuple, Any

from .provider import TTSProvider
from .base import register_provider

from dotenv import load_dotenv

load_dotenv()

# Neuphonic voices
NEUPHONIC_VOICES = [
    ("fc854436-2dac-4d21-aa69-ae17b54e98eb", "Emily - US Female"),
    ("caba5581-7452-4523-8421-97793e90807a", "Dave - British Male"),
    ("f8698a9e-947a-43cd-a897-57edd4070a78", "Albert - British Male"),
    ("fa634f3b-4ccf-41ae-8162-202570cebef4", "Callum - Scottish Male"),
    ("8e9c4bc8-3979-48ab-8626-df53befc2090", "Holly - US Female"),
    ("24a451b8-30b6-4bbe-9646-63e6900e10de", "Jack - Australian Male"),
    ("59f9cb97-c00f-44c5-ad52-f43b52ae8378", "Paul - US Male"),
]


@register_provider("neuphonic")
class NeuPhonicProvider(TTSProvider):
    _api_key = None
    _base_url = "https://api.neuphonic.com/sse/speak/en"
    _voices = NEUPHONIC_VOICES

    @classmethod
    def _initialize_provider(cls):
        """Initialize the Neuphonic provider"""
        cls._api_key = os.getenv("NEUPHONIC_API_KEY")
        if not cls._api_key:
            logger.error("Neuphonic API key not found in environment variables")
            raise ValueError("NEUPHONIC_API_KEY environment variable is required")

        logger.info(f"Neuphonic provider initialized with {len(cls._voices)} voices")

    @classmethod
    def get_available_models(cls) -> List[Dict[str, Any]]:
        """Get a list of available models (voices) for Neuphonic"""
        if not cls.is_available():
            return []

        return [
            {
                "id": voice_id,
                "name": voice_name,
                "description": f"Neuphonic voice: {voice_name}",
            }
            for voice_id, voice_name in cls._voices
        ]

    @classmethod
    async def synthesize(cls, text: str, model_id: str = None) -> Tuple[str, str]:
        """Synthesize speech using Neuphonic SSE API"""
        if not cls.is_available():
            raise ValueError("Neuphonic provider is not available")

        # Select voice - use provided model_id or pick random
        if model_id:
            voice_id = model_id
            # Validate voice exists
            valid_ids = [v[0] for v in cls._voices]
            if voice_id not in valid_ids:
                logger.warning(f"Voice {voice_id} not found, using random voice")
                voice_id = random.choice(valid_ids)
        else:
            voice_id = random.choice([v[0] for v in cls._voices])
            logger.info(f"No voice specified for Neuphonic, using random: {voice_id}")

        headers = {
            "Content-Type": "application/json",
            "X-API-KEY": cls._api_key,
            "Accept": "text/event-stream",
        }

        data = {
            "text": text,
            "voice_id": voice_id,
        }

        audio_chunks = []

        async with httpx.AsyncClient() as client:
            async with client.stream(
                "POST",
                cls._base_url,
                headers=headers,
                json=data,
                timeout=60.0,
            ) as response:
                async for line in response.aiter_lines():
                    if not line:
                        continue

                    if line.startswith("event: error"):
                        logger.error(f"Neuphonic SSE error event received")
                        raise Exception("Neuphonic API returned an error event")

                    if line.startswith("data: "):
                        json_str = line[6:]  # Skip 'data: ' prefix
                        try:
                            json_data = json.loads(json_str)
                        except json.JSONDecodeError:
                            continue

                        if json_data.get("status_code") == 400:
                            error_msg = json_data.get("errors", "Unknown error")
                            logger.error(f"Neuphonic API error: {error_msg}")
                            raise Exception(f"Neuphonic API error: {error_msg}")

                        if json_data.get("status_code") == 200:
                            audio_base64 = json_data.get("data", {}).get("audio")
                            if audio_base64:
                                audio_bytes = base64.b64decode(audio_base64)
                                audio_chunks.append(audio_bytes)

        if not audio_chunks:
            raise Exception("No audio data received from Neuphonic")

        # Combine all audio chunks
        combined_audio = b"".join(audio_chunks)

        # Return base64 encoded audio and extension
        # Neuphonic returns raw PCM 16-bit mono at 22050Hz, we'll return as wav
        # But the raw audio is already PCM, so we need to wrap it in WAV header
        wav_audio = cls._wrap_pcm_as_wav(combined_audio, sample_rate=22050)
        audio_b64 = base64.b64encode(wav_audio).decode("ascii")

        return audio_b64, "wav"

    @classmethod
    def _wrap_pcm_as_wav(cls, pcm_data: bytes, sample_rate: int = 22050) -> bytes:
        """Wrap raw PCM data in a WAV header"""
        import struct

        num_channels = 1
        sample_width = 2  # 16-bit
        byte_rate = sample_rate * num_channels * sample_width
        block_align = num_channels * sample_width
        data_size = len(pcm_data)
        file_size = 36 + data_size

        # Build WAV header
        header = b"RIFF"
        header += struct.pack("<I", file_size)
        header += b"WAVE"
        header += b"fmt "
        header += struct.pack("<I", 16)  # Subchunk1Size (16 for PCM)
        header += struct.pack("<H", 1)  # AudioFormat (1 for PCM)
        header += struct.pack("<H", num_channels)
        header += struct.pack("<I", sample_rate)
        header += struct.pack("<I", byte_rate)
        header += struct.pack("<H", block_align)
        header += struct.pack("<H", sample_width * 8)  # BitsPerSample
        header += b"data"
        header += struct.pack("<I", data_size)

        return header + pcm_data

