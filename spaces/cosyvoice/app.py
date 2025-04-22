# app.py
import os
import torch
import sys
import numpy as np
import torchaudio
import librosa
from flask import Flask, request, jsonify, send_file, render_template
from funasr import AutoModel
from huggingface_hub import snapshot_download
import io
import soundfile as sf

# Initial setup and CUDA check
os.system("nvidia-smi")
print(torch.backends.cudnn.version())


# Modify dynamic modules for huggingface hub compatibility
def modify_dynamic_modules_file(file_path):
    if os.path.exists(file_path):
        with open(file_path, "r") as file:
            lines = file.readlines()
        with open(file_path, "w") as file:
            for line in lines:
                if "from huggingface_hub import cached_download" in line:
                    file.write(
                        "from huggingface_hub import hf_hub_download, model_info\n"
                    )
                else:
                    file.write(line)


# Try to modify both possible locations
dynamic_modules_files = [
    "/home/user/.pyenv/versions/3.10.16/lib/python3.10/site-packages/diffusers/utils/dynamic_modules_utils.py",
    "/usr/local/lib/python3.10/site-packages/diffusers/utils/dynamic_modules_utils.py",
]
for file_path in dynamic_modules_files:
    modify_dynamic_modules_file(file_path)

# Setup paths and download models
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(f"{ROOT_DIR}/third_party/Matcha-TTS")

# Download necessary models
snapshot_download(
    "FunAudioLLM/CosyVoice2-0.5B", local_dir="pretrained_models/CosyVoice2-0.5B"
)
snapshot_download(
    "FunAudioLLM/CosyVoice-ttsfrd", local_dir="pretrained_models/CosyVoice-ttsfrd"
)
os.system(
    "cd pretrained_models/CosyVoice-ttsfrd/ && pip install ttsfrd_dependency-0.1-py3-none-any.whl && pip install ttsfrd-0.4.2-cp310-cp310-linux_x86_64.whl && tar -xvf resource.tar"
)

from cosyvoice.cli.cosyvoice import CosyVoice2
from cosyvoice.utils.file_utils import load_wav, logging
from cosyvoice.utils.common import set_all_random_seed

# Constants
PROMPT_SR = 16000
TARGET_SR = 24000
MAX_VAL = 0.8
DEFAULT_DATA = np.zeros(TARGET_SR)

app = Flask(__name__)


def postprocess(speech, top_db=60, hop_length=220, win_length=440):
    """Post-process the generated speech."""
    speech, _ = librosa.effects.trim(
        speech, top_db=top_db, frame_length=win_length, hop_length=hop_length
    )
    if speech.abs().max() > MAX_VAL:
        speech = speech / speech.abs().max() * MAX_VAL
    speech = torch.concat([speech, torch.zeros(1, int(TARGET_SR * 0.2))], dim=1)
    return speech


def load_reference_audio():
    """Load and transcribe the reference audio file at startup."""
    reference_path = "sample.wav"
    if not os.path.exists(reference_path):
        raise FileNotFoundError(f"Reference audio file {reference_path} not found!")

    # Get transcription using ASR
    transcription = asr_model.generate(
        input=reference_path, language="auto", use_itn=True
    )[0]["text"].split("|>")[-1]

    # Load and process audio
    reference_audio = postprocess(load_wav(reference_path, PROMPT_SR))

    return reference_audio, transcription


def generate_audio(text, seed=42):
    """Generate audio using the reference voice."""
    set_all_random_seed(seed)

    # Collect all audio segments
    audio_segments = []

    # Get all outputs at once without streaming
    for output in cosyvoice.inference_zero_shot(
        text, reference_transcription, reference_audio, stream=False, speed=1.0
    ):
        audio_segments.append(output["tts_speech"].numpy().flatten())

    # Concatenate all audio segments into one
    if audio_segments:
        combined_audio = np.concatenate(audio_segments)
        return combined_audio
    else:
        return DEFAULT_DATA


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/generate", methods=["POST"])
def generate():
    try:
        data = request.get_json()
        text = data.get("text", "")
        seed = int(data.get("seed", 42))

        # Generate audio
        audio_data = generate_audio(text, seed)

        # Convert to WAV format
        buffer = io.BytesIO()
        sf.write(buffer, audio_data, TARGET_SR, format="WAV")
        buffer.seek(0)

        # Return the audio file
        return send_file(
            buffer,
            mimetype="audio/wav",
            as_attachment=True,
            download_name="generated_speech.wav",
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    # Initialize ASR model
    asr_model = AutoModel(
        model="iic/SenseVoiceSmall",
        disable_update=True,
        log_level="DEBUG",
        device="cuda:0",
    )

    # Initialize TTS model
    cosyvoice = CosyVoice2(
        "pretrained_models/CosyVoice2-0.5B",
        load_jit=False,
        load_onnx=False,
        load_trt=False,
    )

    # Load reference audio and get transcription
    reference_audio, reference_transcription = load_reference_audio()
    print(f"Reference audio loaded with transcription: {reference_transcription}")

    # Start the Flask app
    app.run(host="0.0.0.0", port=7860)
