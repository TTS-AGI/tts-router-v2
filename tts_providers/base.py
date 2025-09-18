from loguru import logger
from typing import Dict, List, Tuple, Any

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
    """Synthesize speech using the specified provider and model"""
    provider_name = provider_name.lower()
    if provider_name not in _PROVIDERS:
        raise ValueError(f"Provider '{provider_name}' not found or not available")

    provider = _PROVIDERS[provider_name]

    # Get raw audio from provider
    raw_audio_data, original_extension = await provider.synthesize(text, model_id)

    # No post-processing — just raw output
    return raw_audio_data, original_extension


# Try to load all provider modules
def _try_import(module_name: str, pretty_name: str):
    try:
        __import__(f"tts_providers.{module_name}")
    except Exception as e:
        logger.error(f"Failed to load {pretty_name} provider: {str(e)}")


_provider_modules = [
    ("elevenlabs", "ElevenLabs"),
    ("playht", "PlayHT"),
    ("cosyvoice", "CosyVoice"),
    ("styletts", "StyleTTS"),
    ("hume", "Hume"),
    ("papla", "Papla"),
    ("kokoro", "Kokoro"),
    ("magpie", "Magpie"),
    ("cartesia", "Cartesia"),
    ("spark", "Spark-TTS"),
    ("megatts3", "MegaTTS3"),
    ("minimax", "Minimax"),
    ("lanternfish", "Lanternfish"),
    ("asyncai", "AsyncAI"),
    ("nls", "NLS"),
    ("chatterbox", "Chatterbox"),
    ("inworld", "Inworld"),
    ("wordcab", "Wordcab"),
    ("veena", "Maya Research Veena"),
]

for module_name, pretty_name in _provider_modules:
    _try_import(module_name, pretty_name)

# Initialize providers
for name, provider_class in list(_PROVIDERS.items()):
    try:
        provider_class.initialize()
        logger.info(f"Successfully initialized provider: {name}")
    except Exception as e:
        logger.error(f"Failed to initialize provider {name}: {str(e)}")
        # Keep the provider in registry but mark it as unavailable
