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
                # Complete audio reconstruction approach:
                # 1. Extract to raw PCM (no container, no metadata)
                # 2. Reconstruct as clean MP3
                
                with tempfile.NamedTemporaryFile(delete=False, suffix='.raw') as raw_file:
                    raw_path = raw_file.name
                
                try:
                    # Step 1: Extract to raw PCM samples (completely strips all metadata/container)
                    stream1 = ffmpeg.input(input_path)
                    stream1 = ffmpeg.output(
                        stream1,
                        raw_path,
                        acodec='pcm_s16le',     # Raw 16-bit PCM
                        ar=44100,               # Resample to 44kHz
                        ac=1,                   # Convert to mono
                        f='s16le'               # Raw format - no container
                    )
                    ffmpeg.run(stream1, overwrite_output=True, quiet=True)
                    
                    # Step 2: Reconstruct MP3 from raw samples with clean encoding
                    stream2 = ffmpeg.input(raw_path, f='s16le', ar=44100, ac=1)
                    stream2 = ffmpeg.output(
                        stream2,
                        output_path,
                        acodec='libmp3lame',    # MP3 encoder
                        ar=44100,               # 44kHz
                        ab='128k',              # Fixed bitrate
                        ac=1,                   # Mono
                        map_metadata=-1,        # No metadata
                        id3v2_version=0,        # No ID3v2
                        write_id3v1=0,          # No ID3v1  
                        write_apetag=0,         # No APE tags
                        write_xing=0,           # No Xing header
                        fflags='+bitexact',     # Reproducible
                        f='mp3'
                    )
                    
                    # Run the reconstruction
                    ffmpeg.run(stream2, overwrite_output=True, quiet=True)
                    
                finally:
                    # Clean up raw file
                    try:
                        os.unlink(raw_path)
                    except OSError:
                        pass
                
                # Read the processed audio
                with open(output_path, 'rb') as f:
                    processed_audio = f.read()
                
                # Final binary-level cleaning to remove any remaining signatures
                processed_audio = AudioProcessor._deep_clean_binary(processed_audio)
                
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
    def _deep_clean_binary(audio_data: bytes) -> bytes:
        """
        Perform deep binary-level cleaning of audio data to remove any identifying information.
        This approach analyzes the MP3 structure and removes/neutralizes non-audio data.
        
        Args:
            audio_data: Raw MP3 audio bytes
            
        Returns:
            Cleaned audio bytes with identifying information removed
        """
        if len(audio_data) < 10:
            return audio_data
            
        cleaned_data = bytearray(audio_data)
        modifications_made = 0
        
        # MP3 frame sync pattern: 0xFFE (11 bits)
        # We'll preserve only valid MP3 frames and clean everything else
        
        i = 0
        while i < len(cleaned_data) - 4:
            # Look for MP3 frame sync (0xFF followed by 0xE0-0xFF)
            if cleaned_data[i] == 0xFF and (cleaned_data[i + 1] & 0xE0) == 0xE0:
                # This looks like an MP3 frame header
                # Calculate frame length and skip over the frame
                try:
                    # Parse MP3 header to get frame length
                    header = (cleaned_data[i] << 24) | (cleaned_data[i + 1] << 16) | \
                            (cleaned_data[i + 2] << 8) | cleaned_data[i + 3]
                    
                    # Extract bitrate and sample rate info
                    version = (header >> 19) & 0x3
                    layer = (header >> 17) & 0x3
                    bitrate_index = (header >> 12) & 0xF
                    sample_rate_index = (header >> 10) & 0x3
                    
                    # Skip if invalid indices
                    if bitrate_index == 0 or bitrate_index == 15 or sample_rate_index == 3:
                        i += 1
                        continue
                    
                    # Calculate frame size (simplified for Layer III)
                    if layer == 1:  # Layer III
                        frame_size = 144 * 128 // 44100  # Approximate for our fixed settings
                        if frame_size > 0 and i + frame_size < len(cleaned_data):
                            i += frame_size
                            continue
                            
                except (IndexError, ZeroDivisionError):
                    pass
                    
                # If we can't parse the frame, move to next byte
                i += 1
            else:
                # This byte is not part of an MP3 frame
                # Check if it's part of metadata/text that should be cleaned
                
                # Look for text patterns (ASCII printable characters in sequences)
                if 32 <= cleaned_data[i] <= 126:  # Printable ASCII
                    # Check if this starts a text sequence
                    text_length = 0
                    j = i
                    while j < len(cleaned_data) and j < i + 100:  # Max 100 chars
                        if 32 <= cleaned_data[j] <= 126 or cleaned_data[j] in [0, 9, 10, 13]:
                            text_length += 1
                            j += 1
                        else:
                            break
                    
                    # If we found a text sequence of reasonable length, neutralize it
                    if text_length >= 4:  # Minimum 4 characters to be considered text
                        # Replace with null bytes to maintain file structure
                        for k in range(i, min(i + text_length, len(cleaned_data))):
                            if cleaned_data[k] != 0:  # Don't modify existing nulls
                                cleaned_data[k] = 0
                                modifications_made += 1
                        i += text_length
                        continue
                
                # Look for common binary signatures and neutralize them
                # Check for sequences that look like metadata headers
                if i < len(cleaned_data) - 8:
                    # Look for 4-byte sequences that might be tags
                    four_bytes = bytes(cleaned_data[i:i+4])
                    if (four_bytes.isalpha() or  # All letters
                        (four_bytes[0:3].isalpha() and four_bytes[3].isdigit()) or  # 3 letters + digit
                        four_bytes in [b'ID3\x03', b'ID3\x04', b'TAG+', b'APEV']):  # Known headers
                        
                        # Neutralize this potential metadata header
                        for k in range(i, min(i + 4, len(cleaned_data))):
                            cleaned_data[k] = 0
                            modifications_made += 1
                        i += 4
                        continue
                
                i += 1
        
        if modifications_made > 0:
            logger.info(f"Deep binary cleaning: neutralized {modifications_made} bytes of potential identifying data")
        
        return bytes(cleaned_data)
    
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