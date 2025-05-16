import os
import wget
from pathlib import Path
from typing import Optional

from modal import App, method, Image, Secret

from modal_config import (
    configure_image, MODEL_FILES, GPU_CONFIG, CHECKPOINT_DIR
)

# Create app with configured image
app = App("indextts-service")
image = configure_image()

@app.cls(
    image=image,
    gpu=GPU_CONFIG,
    secrets=[Secret.from_name("huggingface-token")]
)
class IndexTTSService:
    def __enter__(self):
        """Initialize the model and download required files"""
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)

        # Download model files if they don't exist
        for filename, url in MODEL_FILES.items():
            filepath = os.path.join(CHECKPOINT_DIR, filename)
            if not os.path.exists(filepath):
                print(f"Downloading {filename}...")
                try:
                    wget.download(url, filepath)
                except Exception as e:
                    raise Exception(f"Failed to download {filename}: {str(e)}")

        # Initialize the TTS model
        from indextts.infer import IndexTTS
        self.model = IndexTTS(
            model_dir=CHECKPOINT_DIR,
            cfg_path=os.path.join(CHECKPOINT_DIR, "config.yaml"),
            is_fp16=True
        )
        print("Model initialized successfully")

    @method()
    def generate_speech(
        self,
        text: str,
        reference_audio: bytes,
        output_format: str = "wav"
    ) -> bytes:
        """Generate speech from text using the provided reference audio"""
        try:
            # Save reference audio temporarily
            ref_path = "/tmp/reference.wav"
            with open(ref_path, "wb") as f:
                f.write(reference_audio)

            # Generate output path
            output_path = "/tmp/output.wav"

            # Generate speech
            self.model.infer(
                audio_prompt=ref_path,
                text=text,
                output_path=output_path
            )

            # Read and return the generated audio
            with open(output_path, "rb") as f:
                audio_data = f.read()

            # Cleanup
            os.remove(ref_path)
            os.remove(output_path)

            return audio_data

        except Exception as e:
            raise Exception(f"Speech generation failed: {str(e)}")

    @method()
    def health_check(self) -> dict:
        """Check if the service is healthy"""
        return {
            "status": "healthy",
            "model_loaded": hasattr(self, "model"),
            "checkpoint_dir": os.path.exists(CHECKPOINT_DIR)
        }

# Example usage
@app.local_entrypoint()
def main():
    service = IndexTTSService()
    # Add test code here if needed
