from loguru import logger
import importlib
import inspect
from typing import Dict, List, Tuple, Any
from .audio_processor import AudioProcessor

# Registry to store provider implementations
_PROVIDERS = {}


def register_provider(name: str):
    """Decorator to register a TTS provider class"""

    def decorator(cls):
        _PROVIDERS[name.lower()] = cls
        return cls

    return decorator


def get_available_providers() -> List[str]:
    """Return a list of all registered providers that initialized successfully"""
    return [name for name, provider in _PROVIDERS.items() if provider.is_available()]


def get_provider_models(provider_name: str) -> List[Dict[str, Any]]:
    """Return a list of available models for a specific provider"""
    provider_name = provider_name.lower()
    if provider_name not in _PROVIDERS:
        raise ValueError(f"Provider '{provider_name}' not found or not available")

    provider = _PROVIDERS[provider_name]
    return provider.get_available_models()


async def synthesize_speech(
    text: str, provider_name: str, model_id: str = None
) -> Tuple[str, str]:
    """Synthesize speech using the specified provider and model, with audio anonymization"""
    provider_name = provider_name.lower()
    if provider_name not in _PROVIDERS:
        raise ValueError(f"Provider '{provider_name}' not found or not available")

    provider = _PROVIDERS[provider_name]
    
    # Get raw audio from provider
    raw_audio_data, original_extension = await provider.synthesize(text, model_id)
    
    # Process audio for anonymization (convert to MP3, remove metadata, resample to 44kHz)
    logger.info(f"Processing audio from {provider_name} for anonymization")
    processed_audio, processed_extension = AudioProcessor.process_base64_audio(
        raw_audio_data, original_extension
    )
    
    return processed_audio, processed_extension


# Try to load all provider modules
try:
    from . import elevenlabs
except Exception as e:
    logger.error(f"Failed to load ElevenLabs provider: {str(e)}")

try:
    from . import playht
except Exception as e:
    logger.error(f"Failed to load PlayHT provider: {str(e)}")

try:
    from . import cosyvoice
except Exception as e:
    logger.error(f"Failed to load CosyVoice provider: {str(e)}")

try:
    from . import styletts
except Exception as e:
    logger.error(f"Failed to load StyleTTS provider: {str(e)}")

try:
    from . import hume
except Exception as e:
    logger.error(f"Failed to load Hume provider: {str(e)}")

try:
    from . import papla
except Exception as e:
    logger.error(f"Failed to load Papla provider: {str(e)}")

try:
    from . import kokoro
except Exception as e:
    logger.error(f"Failed to load Kokoro provider: {str(e)}")

try:
    from . import cartesia
except Exception as e:
    logger.error(f"Failed to load Cartesia provider: {str(e)}")

try:
    from . import spark
except Exception as e:
    logger.error(f"Failed to load Spark-TTS provider: {str(e)}")

try:
    from . import megatts3
except Exception as e:
    logger.error(f"Failed to load MegaTTS3 provider: {str(e)}")

try:
    from . import minimax
except Exception as e:
    logger.error(f"Failed to load Minimax provider: {str(e)}")

try:
    from . import lanternfish
except Exception as e:
    logger.error(f"Failed to load Lanternfish provider: {str(e)}")

try:
    from . import asyncai
except Exception as e:
    logger.error(f"Failed to load AsyncAI provider: {str(e)}")

# Initialize providers
for name, provider_class in list(_PROVIDERS.items()):
    try:
        provider_class.initialize()
        logger.info(f"Successfully initialized provider: {name}")
    except Exception as e:
        logger.error(f"Failed to initialize provider {name}: {str(e)}")
        # Do not remove the provider from registry, just mark it as unavailable
        # This allows us to report the provider as unavailable rather than missing
