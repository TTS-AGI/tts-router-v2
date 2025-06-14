# Audio Anonymization Features

This TTS router includes comprehensive audio anonymization to make it harder to identify distinguishing characteristics in the generated audio.

## Features Implemented

### 1. Format Standardization
- **Convert all audio to MP3**: Regardless of the original format from TTS providers
- **Consistent sample rate**: All audio resampled to 44kHz
- **Mono conversion**: Convert to single channel to reduce identifying characteristics
- **Consistent bitrate**: All audio encoded at 128kbps

### 2. Complete Audio Reconstruction
- **Raw PCM extraction**: Convert to raw PCM samples, completely stripping all container metadata and headers
- **Clean MP3 reconstruction**: Re-encode from raw samples with minimal, standardized headers
- **Two-stage processing**: Streamlined pipeline for maximum compatibility and effectiveness

### 3. Deep Binary-Level Cleaning
- **MP3 structure analysis**: Analyzes MP3 file structure to identify audio frames vs metadata
- **Automatic text detection**: Detects and neutralizes suspicious text sequences (4+ consecutive printable characters)
- **Metadata header removal**: Identifies and removes binary signatures that could indicate metadata headers
- **Comprehensive sanitization**: Neutralizes any non-audio data while preserving MP3 frame integrity

### 4. Generalized Approach
Instead of relying on specific signature lists, the system uses intelligent detection:
- **Text pattern analysis**: Automatically identifies and removes text-like sequences
- **Binary signature detection**: Recognizes metadata headers by structure rather than specific strings
- **Container-agnostic processing**: Works regardless of the original audio format or embedded metadata
- **Future-proof**: Effective against new or unknown identifying markers

## Implementation Details

### AudioProcessor Class
Located in `tts_providers/audio_processor.py`, this class handles all audio anonymization:

- `process_audio()` - Main processing function for raw audio bytes
- `process_base64_audio()` - Wrapper for base64-encoded audio (used by TTS providers)
- `_deep_clean_binary()` - Advanced binary-level cleaning that analyzes MP3 structure and removes identifying data

### Integration
The audio processor is automatically applied to all TTS provider outputs through the `synthesize_speech()` function in `tts_providers/base.py`.

## Benefits for Anonymization

1. **Consistent output format**: All audio has identical technical characteristics
2. **No provider fingerprinting**: Removes TTS provider-specific metadata
3. **No encoder fingerprinting**: Removes encoder-specific signatures
4. **Standardized quality**: Consistent bitrate and sample rate across all outputs
5. **Minimal identifying information**: Only contains the actual audio data

## Dependencies

- `ffmpeg-python`: Python bindings for FFmpeg audio processing
- System FFmpeg installation with libmp3lame support

## Usage

The audio anonymization is applied automatically to all TTS requests. No additional configuration is required - all audio returned by the `/tts` endpoint will be processed through the anonymization pipeline. 