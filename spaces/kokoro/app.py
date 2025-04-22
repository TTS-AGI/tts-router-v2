from flask import Flask, request, send_file, jsonify
from kokoro import KPipeline
import torch
import soundfile as sf
import numpy as np
import tempfile
import os, random

# voices = ['af_alloy', 'af_aoede', 'af_bella', 'af_heart', 'af_jessica', 'af_kore', 'af_nicole', 'af_nova', 'af_river', 'af_sarah', 'af_sky', 'am_adam', 'am_echo', 'am_eric', 'am_fenrir', 'am_liam', 'am_michael', 'am_onyx', 'am_puck', 'am_santa', 'bf_alice', 'bf_emma', 'bf_isabella', 'bf_lily', 'bm_daniel', 'bm_fable', 'bm_george', 'bm_lewis']
voices = [
    "bf_isabella",
    "am_fenrir",
    "bm_fable",
    "af_bella",
    "af_heart",
    "am_michael",
    "bm_george",
    "af_aoede",
    "bf_emma",
]  # Voices included at author's request

app = Flask(__name__)

pipeline = KPipeline(lang_code="a")


@app.route("/synthesize", methods=["POST"])
def synthesize():
    data = request.get_json()
    if not data or "text" not in data:
        return jsonify({"error": 'Missing "text" field in JSON body'}), 400

    text = data["text"]
    try:
        generator = pipeline(text, voice=random.choice(voices))
        audio_chunks = [audio for _, _, audio in generator]
        audio_concat = np.concatenate(audio_chunks)

        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        sf.write(temp_file.name, audio_concat, 24000)

        return send_file(
            temp_file.name,
            mimetype="audio/wav",
            as_attachment=True,
            download_name="output.wav",
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/", methods=["GET"])
def root():
    return (
        "Kokoro TTS API is running. POST to /synthesize with {'text': 'your sentence'}"
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860)
