# Audio Anonymization Features

This TTS router includes comprehensive audio anonymization to make it harder to identify distinguishing characteristics in the generated audio.

## Features Implemented

### 1. Format Standardization
- **Convert all audio to MP3**: Regardless of the original format from TTS providers
- **Consistent sample rate**: All audio resampled to 44kHz
- **Mono conversion**: Convert to single channel to reduce identifying characteristics
- **Consistent bitrate**: All audio encoded at 128kbps

### 2. Metadata Removal
- **Complete metadata stripping**: Remove all ID3v1, ID3v2, and APE tags
- **Encoder signature removal**: Strip encoder identifiers like "LAME", "Lavf"
- **TTS signature removal**: Remove AI-generated content markers and TTS-specific metadata
- **Header normalization**: Remove Xing/Info headers that may contain identifying information

### 3. Processing Pipeline
The audio processing uses a two-step approach:
1. **Raw PCM conversion**: Convert to raw PCM format to completely strip container metadata
2. **Clean MP3 encoding**: Re-encode to MP3 with strict metadata controls

### 4. Signatures Removed
The system automatically removes these identifying signatures:
- `Lavf` - libavformat signatures
- `LAME` - LAME encoder signatures  
- `TSSE` - Software encoder tags
- `TXXX` - User-defined text metadata
- `aigc` - AI-generated content markers
- `HUABABSpeech` - Specific TTS provider signatures
- `ContentProducer` - Content producer identification
- `ProduceID` - Producer ID tags
- `Xing`/`Info` - MP3 header information
- `VBRI` - Variable bitrate information

## Implementation Details

### AudioProcessor Class
Located in `tts_providers/audio_processor.py`, this class handles all audio anonymization:

- `process_audio()` - Main processing function for raw audio bytes
- `process_base64_audio()` - Wrapper for base64-encoded audio (used by TTS providers)
- `_remove_encoder_signatures()` - Post-processing to remove remaining signatures

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