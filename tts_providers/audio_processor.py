import ffmpeg
import tempfile
import os
import base64
from loguru import logger
from typing import Tuple


class AudioProcessor:
    """Audio processor for anonymizing TTS output"""
    
    @staticmethod
    def process_audio(audio_data: bytes, input_format: str = None) -> Tuple[str, str]:
        """
        Process audio to remove identifying characteristics:
        - Convert to MP3 format
        - Remove all metadata and headers
        - Resample to 44kHz
        - Apply consistent encoding settings
        
        Args:
            audio_data: Raw audio bytes
            input_format: Optional input format hint
            
        Returns:
            Tuple of (base64_encoded_audio, extension)
        """
        try:
            # Create temporary files for processing
            with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{input_format or "audio"}') as input_file:
                input_file.write(audio_data)
                input_file.flush()
                input_path = input_file.name
            
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as output_file:
                output_path = output_file.name
            
            try:
                # Process audio with ffmpeg
                stream = ffmpeg.input(input_path)
                
                # Apply processing pipeline:
                # - Convert to MP3 with consistent settings
                # - Resample to 44kHz
                # - Remove all metadata
                # - Use consistent bitrate and encoding
                stream = ffmpeg.output(
                    stream,
                    output_path,
                    acodec='mp3',           # Force MP3 codec
                    ar=44100,               # Resample to 44kHz
                    ab='128k',              # Consistent bitrate
                    ac=1,                   # Convert to mono to reduce identifying characteristics
                    map_metadata=-1,        # Remove all metadata
                    fflags='+bitexact',     # Ensure reproducible output
                    avoid_negative_ts='make_zero',  # Normalize timestamps
                    f='mp3'                 # Force MP3 format
                )
                
                # Run the conversion
                ffmpeg.run(stream, overwrite_output=True, quiet=True)
                
                # Read the processed audio
                with open(output_path, 'rb') as f:
                    processed_audio = f.read()
                
                # Base64 encode for transport
                encoded_audio = base64.b64encode(processed_audio).decode('ascii')
                
                logger.info(f"Audio processed successfully: {len(audio_data)} -> {len(processed_audio)} bytes")
                
                return encoded_audio, "mp3"
                
            finally:
                # Clean up temporary files
                try:
                    os.unlink(input_path)
                    os.unlink(output_path)
                except OSError:
                    pass
                    
        except Exception as e:
            logger.error(f"Audio processing failed: {str(e)}")
            # Fallback: return original audio as base64 if processing fails
            fallback_audio = base64.b64encode(audio_data).decode('ascii')
            return fallback_audio, input_format or "mp3"
    
    @staticmethod
    def process_base64_audio(base64_audio: str, input_format: str = None) -> Tuple[str, str]:
        """
        Process base64-encoded audio
        
        Args:
            base64_audio: Base64 encoded audio data
            input_format: Optional input format hint
            
        Returns:
            Tuple of (base64_encoded_processed_audio, extension)
        """
        try:
            # Decode base64 audio
            audio_data = base64.b64decode(base64_audio)
            return AudioProcessor.process_audio(audio_data, input_format)
        except Exception as e:
            logger.error(f"Failed to process base64 audio: {str(e)}")
            # Return original if processing fails
            return base64_audio, input_format or "mp3" 