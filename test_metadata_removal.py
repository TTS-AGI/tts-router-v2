#!/usr/bin/env python3
"""
Test script to verify metadata removal from audio files
"""
import base64
import tempfile
import os
from tts_providers.audio_processor import AudioProcessor

def test_metadata_removal():
    """Test that metadata is completely removed from audio"""
    
    # Create a simple test audio file with metadata (using ffmpeg)
    import ffmpeg
    
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as test_file:
        test_path = test_file.name
    
    try:
        # Generate a simple test tone with metadata
        stream = ffmpeg.input('anullsrc=channel_layout=mono:sample_rate=44100', f='lavfi', t=1)
        stream = ffmpeg.output(
            stream,
            test_path,
            acodec='libmp3lame',
            **{
                'metadata:g:0': 'title=Test Audio',
                'metadata:g:1': 'artist=Test Artist', 
                'metadata:g:2': 'album=Test Album'
            },
            f='mp3'
        )
        ffmpeg.run(stream, overwrite_output=True, quiet=True)
        
        # Read the test file
        with open(test_path, 'rb') as f:
            original_audio = f.read()
        
        print(f"Original audio size: {len(original_audio)} bytes")
        
        # Check for metadata in original
        original_hex = original_audio[:200].hex()
        print(f"Original audio header (hex): {original_hex}")
        
        # Look for ID3 tags
        if b'ID3' in original_audio[:100]:
            print("✓ Original audio contains ID3 metadata (as expected)")
        else:
            print("✗ Original audio doesn't contain ID3 metadata")
        
        # Process through our audio processor
        processed_b64, ext = AudioProcessor.process_audio(original_audio, 'mp3')
        
        # Decode processed audio
        processed_audio = base64.b64decode(processed_b64)
        print(f"Processed audio size: {len(processed_audio)} bytes")
        
        # Check processed audio for metadata
        processed_hex = processed_audio[:200].hex()
        print(f"Processed audio header (hex): {processed_hex}")
        
        # Look for ID3 tags in processed audio
        if b'ID3' in processed_audio[:100]:
            print("✗ FAIL: Processed audio still contains ID3 metadata!")
            return False
        else:
            print("✓ SUCCESS: Processed audio has no ID3 metadata")
        
        # Look for other metadata indicators
        metadata_indicators = [b'TSSE', b'TXXX', b'aigc', b'Lavf', b'HUABABSpeech']
        found_metadata = []
        
        for indicator in metadata_indicators:
            if indicator in processed_audio:
                found_metadata.append(indicator.decode('utf-8', errors='ignore'))
        
        if found_metadata:
            print(f"✗ FAIL: Found metadata indicators: {found_metadata}")
            return False
        else:
            print("✓ SUCCESS: No metadata indicators found")
        
        return True
        
    finally:
        try:
            os.unlink(test_path)
        except OSError:
            pass

if __name__ == "__main__":
    print("Testing metadata removal...")
    success = test_metadata_removal()
    if success:
        print("\n🎉 All tests passed! Metadata removal is working correctly.")
    else:
        print("\n❌ Tests failed! Metadata is still present in processed audio.") 