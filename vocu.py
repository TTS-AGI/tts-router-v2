import requests
import json
import random

VOICES = [
    "52f3c95d-ea96-4e4a-8c79-5a1a0aaf5186",  # Ruby
    "4ba81871-0b4b-4bee-a483-49491f86240a",  # Piper
    "1aa3658c-ca34-4d50-822c-323a349fd498",  # Alistair
    "2b65195c-9221-40b8-badc-27f66222b1bb",  # David
    "b19e9f03-73cc-44f1-b990-5681c621894a",  # Scarlett
]

url = "https://v1.vocu.ai/api/tts/simple-generate"

payload = json.dumps({
   "voiceId": random.choice(VOICES),
   "text": "In the quiet hours before dawn, when the world holds its breath between darkness and light, there exists a moment of perfect stillness. The stars fade like forgotten dreams, and the horizon begins to blush with the promise of a new day. It is in these fleeting seconds that we remember what it means to be alive, to witness the eternal dance of shadow and illumination, to feel the weight of time passing through our fingers like grains of sand on an endless shore.",
   "preset": "balance",
   "language": "en",
   "break_clone": True,
   "flash": False,
   "stream": False,
})
headers = {
   'Authorization': 'Bearer sk-b216be682fbc0a717619a0b735a1123d',
   'Content-Type': 'application/json'
}

response = requests.request("POST", url, headers=headers, data=payload)

print(response.text)

# Use data.audio for generated audio
"""
{
    "status": 200,
    "message": "OK",
    "data": {
        "id": "6e2818f1-0817-4425-896b-13fed645a2ce",
        "audio": "https://storage.vocu.ai/generate/f6d422f8-0d1c-4a26-8ee3-a255eb25ebeb/12a2dcfd-9aa8-42b6-bd47-5f6fa3a235cc.mp3",
        "credit_used": 44
    }
}
"""