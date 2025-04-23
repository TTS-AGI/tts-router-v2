import os
import requests
import json
import tempfile
import time
import base64
from loguru import logger
from typing import Dict, List, Tuple, Any

from .provider import TTSProvider
from .base import register_provider

from dotenv import load_dotenv

load_dotenv()


@register_provider("megatts3")
class MegaTTS3Provider(TTSProvider):
    _hf_token = None
    _base_url = "https://bytedance-megatts3.hf.space/gradio_api"
    _models = None

    @classmethod
    def _initialize_provider(cls):
        """Initialize the MegaTTS3 provider"""
        cls._hf_token = os.getenv("HF_TOKEN")
        if not cls._hf_token:
            logger.error("Hugging Face token not found in environment variables")
            raise ValueError("HF_TOKEN environment variable is required")

        # Set up available models
        cls._models = [
            {
                "id": "megatts3",
                "name": "MegaTTS3",
                "description": "ByteDance's MegaTTS3 voice cloning model",
            }
        ]
        logger.info("Successfully initialized MegaTTS3 provider")

    @classmethod
    def get_available_models(cls) -> List[Dict[str, Any]]:
        """Get a list of available models for MegaTTS3"""
        if not cls.is_available() or not cls._models:
            return []

        return cls._models

    @classmethod
    async def synthesize(
        cls, text: str, model_id: str = None, reference_audio: str = None
    ) -> Tuple[str, str]:
        """Synthesize speech using MegaTTS3 with voice cloning"""
        if not cls.is_available():
            raise ValueError("MegaTTS3 provider is not available")

        # Default model is the only model
        if not model_id:
            model_id = "megatts3"
            logger.info(f"No model specified for MegaTTS3, using default: {model_id}")

        # Use default reference files if none provided
        inp_audio_url = (
            reference_audio
            or "https://huggingface.co/spaces/ByteDance/MegaTTS3/resolve/main/official_test_case/bbc_news.wav"
        )
        inp_npy_url = "https://huggingface.co/spaces/ByteDance/MegaTTS3/resolve/main/official_test_case/bbc_news.npy"

        # Default parameters
        infer_timestep = 32
        p_w = 1.4
        t_w = 3

        try:
            # Prepare headers with authentication
            headers = {
                "Authorization": f"Bearer {cls._hf_token}",
                "Content-Type": "application/json",
            }

            # Prepare the request payload
            payload = {
                "data": [
                    {"path": inp_audio_url, "meta": {"_type": "gradio.FileData"}},
                    {"path": inp_npy_url, "meta": {"_type": "gradio.FileData"}},
                    text,
                    infer_timestep,
                    p_w,
                    t_w,
                ]
            }

            logger.info("Sending MegaTTS3 synthesis request...")
            # Initiate the API call
            response = requests.post(
                f"{cls._base_url}/call/predict", headers=headers, json=payload
            )

            # Process the response to get the event ID
            if response.status_code != 200:
                logger.error(
                    f"MegaTTS3 API error: {response.status_code} - {response.text}"
                )
                raise Exception(
                    f"MegaTTS3 API error: {response.status_code} - {response.text}"
                )

            try:
                event_data = response.json()
                logger.debug(f"MegaTTS3 initial response: {event_data}")
            except json.JSONDecodeError as e:
                logger.error(f"Failed to decode JSON from response: {response.text}")
                raise Exception(f"Invalid JSON response: {e}")

            if "event_id" not in event_data:
                raise Exception(f"No event_id in response: {event_data}")

            event_id = event_data["event_id"]
            logger.info(f"Got MegaTTS3 event ID: {event_id}")

            # Poll for the result
            max_attempts = 30
            attempt = 0

            while attempt < max_attempts:
                attempt += 1
                logger.debug(f"Polling attempt {attempt}/{max_attempts}...")

                # Get the result using event_id
                result_response = requests.get(
                    f"{cls._base_url}/call/predict/{event_id}",
                    headers={"Authorization": f"Bearer {cls._hf_token}"},
                )

                if result_response.status_code != 200:
                    logger.warning(
                        f"Polling error: {result_response.status_code} - {result_response.text}"
                    )
                    time.sleep(1)
                    continue

                # Check if the response contains valid JSON by processing line-by-line for SSE
                for line in result_response.text.splitlines():
                    if line.startswith("data:"):
                        try:
                            # Attempt to parse the JSON payload after "data: "
                            json_str = line[len("data:") :].strip()
                            if json_str:  # Ensure it's not just "data:"
                                current_data = json.loads(json_str)
                                if (
                                    current_data
                                    and len(current_data) > 0
                                    and "url" in current_data[0]
                                ):
                                    audio_url = current_data[0]["url"]

                                    # Download the audio file
                                    audio_response = requests.get(audio_url)
                                    if audio_response.status_code != 200:
                                        raise Exception(
                                            f"Failed to download audio: {audio_response.status_code}"
                                        )

                                    # Base64 encode the audio data
                                    audio_data = base64.b64encode(
                                        audio_response.content
                                    ).decode("ascii")

                                    return audio_data, "wav"
                        except json.JSONDecodeError:
                            # Skip invalid JSON lines
                            continue

                time.sleep(2)

            raise Exception("Max polling attempts reached, no result obtained")

        except Exception as e:
            logger.error(f"Error in MegaTTS3 synthesis: {str(e)}")
            raise Exception(f"MegaTTS3 synthesis error: {str(e)}")
