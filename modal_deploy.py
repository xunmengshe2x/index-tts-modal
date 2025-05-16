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
    # First check if service is healthy
    service = IndexTTSService()
    health_status = service.health_check()
    
    if not health_status["model_loaded"]:
        print("Error: Model not loaded properly")
        return
        
    # Reference audio path
    ref_audio_path = "path_to_your_reference.wav"
    
    # Check if reference audio exists
    if not os.path.exists(ref_audio_path):
        print(f"Error: Reference audio file not found at {ref_audio_path}")
        return
        
    try:
        # Read reference audio
        with open(ref_audio_path, "rb") as f:
            reference_audio = f.read()
            
        # Text to convert
        text = "Hello, this is a test of the IndexTTS system."
        
        # Generate speech
        print("Generating speech...")
        audio_data = service.generate_speech(
            text=text,
            reference_audio=reference_audio
        )
        
        # Save output
        output_path = "output.wav"
        with open(output_path, "wb") as f:
            f.write(audio_data)
            
        print(f"Speech generated successfully! Saved to {output_path}")
        
    except Exception as e:
        print(f"Error during speech generation: {str(e)}")
