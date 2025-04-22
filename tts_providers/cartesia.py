import os
import httpx
import random
from loguru import logger
from typing import Dict, List, Tuple, Any
import base64

from .provider import TTSProvider
from .base import register_provider

from dotenv import load_dotenv

load_dotenv()


@register_provider("cartesia")
class CartesiaProvider(TTSProvider):
    _api_key = None
    _base_url = "https://api.cartesia.ai"
    _api_version = "2025-04-16"
    _models = None
    _voices = None

    @classmethod
    def _initialize_provider(cls):
        """Initialize the Cartesia provider"""
        cls._api_key = os.getenv("CARTESIA_API_KEY")
        if not cls._api_key:
            logger.error("Cartesia API key not found in environment variables")
            raise ValueError("CARTESIA_API_KEY environment variable is required")

        # Fetch available models and voices
        try:
            cls._fetch_models()
            cls._fetch_voices()
        except Exception as e:
            logger.error(f"Failed to initialize Cartesia provider: {str(e)}")
            raise

    @classmethod
    def _fetch_models(cls):
        """Set up available models for Cartesia"""
        # Currently Cartesia has a single model
        cls._models = [
            {
                "id": "sonic-2",
                "name": "Sonic 2",
                "description": "Cartesia's primary TTS model",
            }
        ]

    @classmethod
    def _fetch_voices(cls):
        """Fetch available voices from Cartesia API"""
        headers = {
            "Authorization": f"Bearer {cls._api_key}",
            "Cartesia-Version": cls._api_version,
        }

        try:
            with httpx.Client() as client:
                logger.info("Fetching Cartesia voices...")
                response = client.get(
                    f"{cls._base_url}/voices/", headers=headers, params={"limit": 100}
                )
                response.raise_for_status()

                data = response.json()
                cls._voices = data.get("data", [])
                logger.info(f"Fetched initial batch of {len(cls._voices)} voices")

                # Track voice IDs to avoid duplicates
                voice_ids = {
                    voice.get("id") for voice in cls._voices if voice.get("id")
                }

                # Handle pagination if needed
                page_count = 1
                while data.get("has_more", False) and cls._voices:
                    # Use the ID of the last voice in the current response for pagination
                    last_voice_id = cls._voices[-1].get("id")
                    if not last_voice_id:
                        logger.warning(
                            "Cannot continue pagination: last voice has no ID"
                        )
                        break

                    logger.info(
                        f"Fetching additional voices page {page_count+1} with starting_after: {last_voice_id}"
                    )
                    response = client.get(
                        f"{cls._base_url}/voices/",
                        headers=headers,
                        params={"limit": 100, "starting_after": last_voice_id},
                    )
                    response.raise_for_status()
                    data = response.json()
                    new_voices = data.get("data", [])

                    # Filter out duplicate voices
                    unique_new_voices = []
                    for voice in new_voices:
                        voice_id = voice.get("id")
                        if voice_id and voice_id not in voice_ids:
                            unique_new_voices.append(voice)
                            voice_ids.add(voice_id)

                    if len(unique_new_voices) != len(new_voices):
                        logger.warning(
                            f"Found {len(new_voices) - len(unique_new_voices)} duplicate voices on page {page_count+1}"
                        )

                    cls._voices.extend(unique_new_voices)
                    logger.info(
                        f"Fetched {len(unique_new_voices)} additional unique voices from page {page_count+1}"
                    )
                    page_count += 1

                if not cls._voices:
                    logger.warning("No voices found for Cartesia")
                else:
                    logger.info(
                        f"Successfully fetched a total of {len(cls._voices)} unique Cartesia voices from {page_count} page(s)"
                    )
        except Exception as e:
            logger.error(f"Failed to fetch Cartesia voices: {str(e)}")
            cls._voices = []
            raise

    @classmethod
    def get_available_models(cls) -> List[Dict[str, Any]]:
        """Get a list of available models for Cartesia"""
        if not cls.is_available() or not cls._models:
            return []

        return cls._models

    @classmethod
    async def synthesize(cls, text: str, model_id: str = None) -> Tuple[str, str]:
        """Synthesize speech using Cartesia"""
        if not cls.is_available():
            raise ValueError("Cartesia provider is not available")

        if not model_id:
            # Use the default model
            model_id = "sonic-2"
            logger.info(f"No model specified for Cartesia, using default: {model_id}")

        # Select a random English voice if available
        english_voices = [v for v in cls._voices if v.get("language") == "en"]
        if not english_voices:
            english_voices = cls._voices  # Fallback to all voices if no English ones

        if not english_voices:
            raise ValueError("No voices available for Cartesia")

        voice = random.choice(english_voices)
        voice_id = voice.get("id")

        headers = {
            "Authorization": f"Bearer {cls._api_key}",
            "Cartesia-Version": cls._api_version,
            "Content-Type": "application/json",
        }

        data = {
            "model_id": model_id,
            "transcript": text,
            "voice": {"mode": "id", "id": voice_id},
            "output_format": {
                "container": "mp3",
                "bit_rate": 128000,
                "sample_rate": 44100,
            },
            "language": "en",
        }

        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    f"{cls._base_url}/tts/bytes",
                    headers=headers,
                    json=data,
                    timeout=30.0,
                )

                if response.status_code != 200:
                    logger.error(
                        f"Cartesia API error: {response.status_code} - {response.text}"
                    )
                    raise Exception(
                        f"Cartesia API error: {response.status_code} - {response.text}"
                    )

                # Base64 encode the audio data
                audio_data = base64.b64encode(response.content).decode("ascii")

                return audio_data, "mp3"

            except Exception as e:
                logger.error(f"Error in Cartesia synthesis: {str(e)}")
                raise Exception(f"Cartesia synthesis error: {str(e)}")
