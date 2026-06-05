> [!IMPORTANT]  
> This repository is now archived, it was used in [TTS Arena v2](https://github.com/TTS-AGI/TTS-Arena/tree/v2). Going forward, TTS providers are added as plugins directly to the [TTS Arena repository](https://github.com/TTS-AGI/TTS-Arena). To add your model, please review documentation in that repository.

# TTS Router V2

The router used in the upcoming new release of the TTS Arena.

## Features

- Support for multiple TTS providers (ElevenLabs, PlayHT, Papla, CosyVoice, Hume, Deepdub, Gradium, Typecast, Kokoro, Maya Research Veena)
- RESTful API with FastAPI
- Provider and model selection
- Default voice selection

## Setup

1. Clone the repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Create a `.env` file with your API keys:
   ```
   ELEVENLABS_API_KEY=your_elevenlabs_api_key
   PLAYHT_API_KEY=your_playht_api_key
   PLAYHT_USER_ID=your_playht_user_id
   PAPLA_API_KEY=your_papla_api_key
   CARTESIA_API_KEY=your_cartesia_api_key
   HF_TOKEN=your_huggingface_token # For Kokoro, CosyVoice spaces
   HUME_API_KEY=your_hume_api_key
   DEEPDUB_API_KEY=your_deepdub_api_key
   GRADIUM_API_KEY=your_gradium_api_key
   TYPECAST_API_KEY=your_typecast_api_key
   NLS_TOKEN=your_nls_token
   NLS_BASE_URL=https://nls-gateway-singapore.aliyuncs.com
   VEENA_API_KEY=your_veena_api_key # For Maya Research Veena TTS
   VOCU_API_KEY=your_vocu_api_key # For Vocu TTS
   PARMESAN_API_KEY=your_parmesan_api_key # For Parmesan TTS
   ZEROGPU_TOKENS=hf_xxx,hf_xxx
   ```
   (The `ZEROGPU_TOKENS` variable should be a comma-separated list of Hugging Face tokens for multiple Hugging Face accounts to avoid ZeroGPU rate limiting)

## Running the API

```
uvicorn app:app --host 0.0.0.0 --port 8000
```

Or use the Python script:

```
python app.py
```

## API Endpoints

### GET /providers

Lists all available TTS providers.

**Response:**
```json
{
  "providers": ["elevenlabs", "playht"]
}
```

### GET /providers/{provider}/models

Lists all available models for a specific provider.

**Response:**
```json
{
  "models": [
    {
      "id": "eleven_multilingual_v2",
      "name": "Multilingual v2",
      "description": "Multilingual speech generation model"
    },
    ...
  ]
}
```

### POST /tts

Generates TTS audio from text using the specified provider and model.

**Request:**
```json
{
  "text": "Hello, this is a test.",
  "provider": "elevenlabs",
  "model": "eleven_multilingual_v2"
}
```

**Response:**
```json
{
  "status": "success",
  "provider": "elevenlabs",
  "model": "eleven_multilingual_v2",
  "audio_data": "base64_encoded_audio_data",
  "extension": "mp3"
}
```

## Extending with New Providers

To add a new TTS provider:

1. Create a new file in the `tts_providers` directory, e.g., `tts_providers/new_provider.py`
2. Implement the `TTSProvider` interface
3. Register the provider using the `@register_provider` decorator
4. Add the import to `tts_providers/base.py`

## Error Handling

The API includes robust error handling:
- Provider-specific errors are logged and don't crash the server
- If a provider fails to initialize, it's marked as unavailable but doesn't affect other providers

## License

Unless otherwise specified, the code is licensed under the MIT license.

Note that the code in the `spaces` directory may be subject to different licenses. For example, the CosyVoice space is licensed under the Apache License 2.0.
