import os
import requests
import json
import random
import base64
from loguru import logger
from typing import Dict, List, Tuple, Any

from .provider import TTSProvider
from .base import register_provider

from dotenv import load_dotenv

load_dotenv()


@register_provider("nls")
class NLSProvider(TTSProvider):
    _token = None
    _base_url = os.getenv("NLS_BASE_URL", "https://nls-gateway-singapore.aliyuncs.com")
    _models = None
    _spk_lists = {}

    @classmethod
    def _initialize_provider(cls):
        """Initialize the NLS provider"""
        cls._token = os.getenv("NLS_TOKEN")
        if not cls._token:
            logger.error("NLS token not found in environment variables")
            raise ValueError("NLS_TOKEN environment variable is required")

        # Set up available models and fetch speaker list
        cls._models = [
            {
                "id": "tts-arena",
                "name": "NLS Pre V1",
                "description": "NLS pre-release TTS model",
            },
            {
                "id": "tts-arena-1",
                "name": "Luck Dolphin",
                "description": "NLS Luck Dolphin pre-release TTS model",
            },
            {
                "id": "tts-arena-2",
                "name": "Luck Dolphin Turbo",
                "description": "NLS Luck Dolphin Turbo pre-release TTS model",
            },
        ]
        
        # Fetch available speakers
        try:
            for model in cls._models:
                cls._fetch_speakers(model["id"])
        except Exception as e:
            logger.error(f"Failed to fetch NLS speakers: {str(e)}")

    @classmethod
    def _make_http_header(cls):
        """Create HTTP headers for NLS API requests"""
        return {
            "Content-Type": "application/json",
            "X-NLS-Token": cls._token,
        }

    @classmethod
    def _fetch_speakers(cls, appkey: str):
        """Fetch available speakers from NLS API"""
        list_url = cls._base_url + '/rest/v1/general/TtsArenaGet'
        
        try:
            response = requests.post(
                list_url, 
                headers=cls._make_http_header(), 
                params={'appkey': appkey, 'any_response': 'true'}
            )
            
            if response.status_code == 200:
                list_result = json.loads(response.json()['data'])
                cls._spk_lists[appkey] = list_result.get('spk_list', [])
                logger.info(f"Fetched {len(cls._spk_lists[appkey])} NLS speakers for {appkey}")
            else:
                logger.error(f"Failed to fetch NLS speakers for {appkey}: {response.status_code}")
                cls._spk_lists[appkey] = []
        except Exception as e:
            logger.error(f"Error fetching NLS speakers for {appkey}: {str(e)}")
            cls._spk_lists[appkey] = []

    @classmethod
    def get_available_models(cls) -> List[Dict[str, Any]]:
        """Get a list of available models for NLS"""
        if not cls.is_available() or not cls._models:
            return []

        return cls._models

    @classmethod
    async def synthesize(cls, text: str, model_id: str = None) -> Tuple[str, str]:
        """Synthesize speech using NLS"""
        if not cls.is_available():
            raise ValueError("NLS provider is not available")

        if not model_id:
            model_id = "tts-arena"
            logger.info(f"No model specified for NLS, using default: {model_id}")

        valid_model_ids = {model["id"] for model in cls._models}
        if model_id == "nls-1":
            model_id = "tts-arena"
        if model_id not in valid_model_ids:
            available_models = ", ".join(valid_model_ids)
            logger.error(f"NLS model {model_id} not found. Available models: {available_models}")
            raise ValueError(f"Model {model_id} not found for NLS provider")

        # Select a random speaker if available
        if not cls._spk_lists.get(model_id):
            logger.warning("No speakers available, attempting to fetch speakers")
            cls._fetch_speakers(model_id)
        
        if not cls._spk_lists.get(model_id):
            raise ValueError("No speakers available for NLS synthesis")

        spk_id = random.choice(cls._spk_lists[model_id])
        logger.info(f"Using NLS model {model_id} speaker: {spk_id}")

        synthesis_url = cls._base_url + '/rest/v1/general/TtsArenaInfer'
        data = {
            'tts_text': text,
            'spk_id': spk_id
        }

        try:
            response = requests.post(
                synthesis_url, 
                headers=cls._make_http_header(), 
                data=json.dumps(data),
                params={'appkey': model_id, 'any_response': 'true'},
                timeout=30.0
            )

            if response.status_code != 200:
                logger.error(f"NLS API error: {response.status_code} - {response.text}")
                raise Exception(f"NLS API error: {response.status_code} - {response.text}")

            # Parse the response
            result = json.loads(response.json()['data'])
            
            # The result contains a URL to the audio file
            if 'url' not in result:
                logger.error(f"NLS API response missing audio URL: {result}")
                raise Exception("NLS API response missing audio URL")

            audio_url = result['url']
            logger.info(f"Downloading audio from: {audio_url}")

            # Download the audio file from the URL
            audio_response = requests.get(audio_url, timeout=30.0)
            if audio_response.status_code != 200:
                logger.error(f"Failed to download audio: {audio_response.status_code}")
                raise Exception(f"Failed to download audio from NLS: {audio_response.status_code}")

            # Base64 encode the audio data
            audio_data = base64.b64encode(audio_response.content).decode("ascii")

            return audio_data, "wav"

        except Exception as e:
            logger.error(f"Error in NLS synthesis: {str(e)}")
            raise Exception(f"NLS synthesis error: {str(e)}") 
