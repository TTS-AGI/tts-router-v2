from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Any
from loguru import logger


class TTSProvider(ABC):
    """Base class for all TTS providers"""

    _initialized = False
    _available = False

    @classmethod
    def initialize(cls):
        """Initialize the provider. Should be called before using the provider."""
        if not cls._initialized:
            cls._initialize_provider()
            cls._initialized = True
            cls._available = True

    @classmethod
    @abstractmethod
    def _initialize_provider(cls):
        """Provider-specific initialization logic"""
        pass

    @classmethod
    def is_available(cls) -> bool:
        """Check if the provider is available (initialized successfully)"""
        return cls._available

    @classmethod
    @abstractmethod
    def get_available_models(cls) -> List[Dict[str, Any]]:
        """Get a list of available models for this provider"""
        pass

    @classmethod
    @abstractmethod
    async def synthesize(cls, text: str, model_id: str = None) -> Tuple[bytes, str]:
        """
        Synthesize speech using the specified model

        Args:
            text: The text to synthesize
            model_id: The ID of the model to use. If None, use the default model.

        Returns:
            A tuple of (audio_data, extension)
        """
        pass
