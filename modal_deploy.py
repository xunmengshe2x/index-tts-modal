import os
import requests
from pathlib import Path
from typing import Dict, Optional

from modal import Image, Stub, method, Secret
import torch
import torchaudio

from modal_config import get_image, GPU_CONFIG, MODEL_URLS, DIRS
from indextts.infer import IndexTTS

# Create Modal stub
stub = Stub("index-tts")

@stub.cls(
    image=get_image(),
    gpu=GPU_CONFIG,
    secrets=[Secret.from_name("huggingface-token")]
)
class IndexTTSDeployment:
    def download_file(self, url: str, save_path: str, token: Optional[str] = None) -> None:
        """Download file from URL with optional token authentication"""
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        response = requests.get(url, headers=headers, stream=True)
        response.raise_for_status()

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

    def __enter__(self):
        """Initialize model and download weights"""
        # Create required directories
        for dir_path in DIRS.values():
            os.makedirs(dir_path, exist_ok=True)

        # Download model files
        hf_token = os.environ.get("HUGGINGFACE_TOKEN")
        for file_name, url in MODEL_URLS.items():
            save_path = os.path.join(DIRS["checkpoints"], os.path.basename(url))
            if not os.path.exists(save_path):
                print(f"Downloading {file_name} from {url}")
                self.download_file(url, save_path, hf_token)

        # Initialize model
        self.model = IndexTTS(
            model_dir=DIRS["checkpoints"],
            cfg_path=os.path.join(DIRS["checkpoints"], "config.yaml"),
            is_fp16=True,
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        print("Model initialized successfully")

    @method()
    def generate_speech(
        self,
        text: str,
        reference_audio_url: str,
        max_text_tokens_per_sentence: int = 100,
        sentences_bucket_max_size: int = 4,
    ) -> Dict:
        """Generate speech from text using reference audio"""
        # Download reference audio
        ref_audio_path = os.path.join(DIRS["test_data"], "reference.wav")
        self.download_file(reference_audio_url, ref_audio_path)

        # Generate speech
        output_path = os.path.join(DIRS["test_data"], "output.wav")
        try:
            self.model.infer_fast(
                audio_prompt=ref_audio_path,
                text=text,
                output_path=output_path,
                max_text_tokens_per_sentence=max_text_tokens_per_sentence,
                sentences_bucket_max_size=sentences_bucket_max_size
            )

            # Read generated audio
            audio, sr = torchaudio.load(output_path)
            audio_data = audio.numpy().T

            return {
                "sampling_rate": sr,
                "audio_data": audio_data.tolist()
            }

        except Exception as e:
            raise Exception(f"Speech generation failed: {str(e)}")

    @method()
    def health_check(self) -> Dict:
        """Check if the model is loaded and running"""
        return {
            "status": "healthy",
            "model_version": self.model.model_version if hasattr(self.model, "model_version") else "unknown"
        }