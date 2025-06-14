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
                # Create intermediate raw audio file to completely strip all metadata
                with tempfile.NamedTemporaryFile(delete=False, suffix='.raw') as raw_file:
                    raw_path = raw_file.name
                
                try:
                    # Step 1: Convert to raw PCM to strip all metadata and container info
                    stream1 = ffmpeg.input(input_path)
                    stream1 = ffmpeg.output(
                        stream1,
                        raw_path,
                        acodec='pcm_s16le',     # Raw PCM 16-bit little endian
                        ar=44100,               # Resample to 44kHz
                        ac=1,                   # Convert to mono
                        f='s16le'               # Raw format
                    )
                    ffmpeg.run(stream1, overwrite_output=True, quiet=True)
                    
                    # Step 2: Convert raw PCM back to MP3 with no metadata
                    stream2 = ffmpeg.input(raw_path, f='s16le', ar=44100, ac=1)
                    stream2 = ffmpeg.output(
                        stream2,
                        output_path,
                        acodec='libmp3lame',    # Use libmp3lame for better control
                        ar=44100,               # Keep 44kHz
                        ab='128k',              # Consistent bitrate
                        ac=1,                   # Keep mono
                        map_metadata=-1,        # Remove all metadata
                        id3v2_version=0,        # Disable ID3v2 tags completely
                        write_id3v1=0,          # Disable ID3v1 tags
                        write_apetag=0,         # Disable APE tags
                        fflags='+bitexact',     # Ensure reproducible output
                        f='mp3'                 # Force MP3 format
                    )
                    
                    # Run the conversion
                    ffmpeg.run(stream2, overwrite_output=True, quiet=True)
                    
                finally:
                    # Clean up intermediate raw file
                    try:
                        os.unlink(raw_path)
                    except OSError:
                        pass
                
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