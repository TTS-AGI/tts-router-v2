import base64
import io
from loguru import logger
from typing import Tuple
from pydub import AudioSegment
from pydub.utils import make_chunks


class AudioProcessor:
    """Audio processor for anonymizing TTS output using pydub"""
    
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
            input_format: Optional input format hint (e.g., 'wav', 'mp3', 'ogg')
            
        Returns:
            Tuple of (base64_encoded_audio, extension)
        """
        try:
            # Load audio from bytes
            audio_io = io.BytesIO(audio_data)
            
            # Try to determine format if not provided
            if input_format is None:
                input_format = AudioProcessor._detect_format(audio_data)
            
            # Load audio with pydub
            try:
                if input_format:
                    audio = AudioSegment.from_file(audio_io, format=input_format)
                else:
                    # Let pydub auto-detect format
                    audio = AudioSegment.from_file(audio_io)
            except Exception as e:
                logger.warning(f"Failed to load with format '{input_format}', trying auto-detection: {e}")
                audio_io.seek(0)
                audio = AudioSegment.from_file(audio_io)
            
            # Complete audio reconstruction approach:
            # 1. Convert to raw samples (strips container and metadata)
            # 2. Reconstruct with clean settings
            
            # Step 1: Normalize audio properties
            # Convert to mono (removes stereo positioning info)
            if audio.channels > 1:
                audio = audio.set_channels(1)
            
            # Resample to 44.1kHz (standard rate, removes original sample rate fingerprint)
            if audio.frame_rate != 44100:
                audio = audio.set_frame_rate(44100)
            
            # Normalize sample width to 16-bit (removes bit depth fingerprint)
            if audio.sample_width != 2:
                audio = audio.set_sample_width(2)
            
            # Step 2: Extract raw audio data and reconstruct
            # Get raw audio samples as bytes
            raw_audio_data = audio.raw_data
            
            # Reconstruct AudioSegment from raw data (completely clean)
            clean_audio = AudioSegment(
                raw_audio_data,
                frame_rate=44100,
                sample_width=2,
                channels=1
            )
            
            # Step 3: Apply audio processing to further anonymize
            # Slight normalization to remove volume fingerprints
            clean_audio = clean_audio.normalize()
            
            # Optional: Apply very subtle high-pass filter to remove DC offset
            # This helps remove some encoding artifacts
            clean_audio = clean_audio.high_pass_filter(20)
            
            # Step 4: Export to MP3 with consistent settings
            output_io = io.BytesIO()
            clean_audio.export(
                output_io,
                format="mp3",
                bitrate="128k",
                parameters=[
                    "-ar", "44100",           # Sample rate
                    "-ac", "1",               # Mono
                    "-map_metadata", "-1",    # Remove all metadata
                    "-id3v2_version", "0",    # No ID3v2 tags
                    "-write_id3v1", "0",      # No ID3v1 tags
                    "-write_apetag", "0",     # No APE tags
                    "-write_xing", "0",       # No Xing header
                ]
            )
            
            processed_audio = output_io.getvalue()
            
            # Final binary-level cleaning to remove any remaining signatures
            processed_audio = AudioProcessor._deep_clean_binary(processed_audio)
            
            # Base64 encode for transport
            encoded_audio = base64.b64encode(processed_audio).decode('ascii')
            
            logger.info(f"Audio processed successfully: {len(audio_data)} -> {len(processed_audio)} bytes")
            
            return encoded_audio, "mp3"
            
        except Exception as e:
            logger.error(f"Audio processing failed: {str(e)}")
            # Fallback: return original audio as base64 if processing fails
            fallback_audio = base64.b64encode(audio_data).decode('ascii')
            return fallback_audio, input_format or "mp3"
    
    @staticmethod
    def _detect_format(audio_data: bytes) -> str:
        """
        Detect audio format from binary data
        
        Args:
            audio_data: Raw audio bytes
            
        Returns:
            Detected format string or None
        """
        if len(audio_data) < 12:
            return None
            
        # Check for common audio format signatures
        if audio_data.startswith(b'RIFF') and b'WAVE' in audio_data[:12]:
            return 'wav'
        elif audio_data.startswith(b'ID3') or audio_data.startswith(b'\xff\xfb'):
            return 'mp3'
        elif audio_data.startswith(b'OggS'):
            return 'ogg'
        elif audio_data.startswith(b'fLaC'):
            return 'flac'
        elif audio_data.startswith(b'FORM') and b'AIFF' in audio_data[:12]:
            return 'aiff'
        elif audio_data[4:8] == b'ftyp':
            return 'm4a'
        
        return None
    
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
    
    @staticmethod
    def chunk_audio(audio_data: bytes, chunk_duration_ms: int = 30000, input_format: str = None) -> list:
        """
        Split audio into chunks for processing large files
        
        Args:
            audio_data: Raw audio bytes
            chunk_duration_ms: Duration of each chunk in milliseconds
            input_format: Optional input format hint
            
        Returns:
            List of audio chunks as bytes
        """
        try:
            audio_io = io.BytesIO(audio_data)
            
            if input_format:
                audio = AudioSegment.from_file(audio_io, format=input_format)
            else:
                audio = AudioSegment.from_file(audio_io)
            
            chunks = make_chunks(audio, chunk_duration_ms)
            
            chunk_bytes = []
            for chunk in chunks:
                chunk_io = io.BytesIO()
                chunk.export(chunk_io, format="wav")
                chunk_bytes.append(chunk_io.getvalue())
            
            return chunk_bytes
            
        except Exception as e:
            logger.error(f"Failed to chunk audio: {str(e)}")
            return [audio_data]  # Return original as single chunk if chunking fails
