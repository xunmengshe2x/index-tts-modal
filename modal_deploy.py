import os
import wget
import torch
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
    mounts=[app.mount.from_local_dir(".", remote_path="/root/app")]
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
        
        # Determine device
        if torch.cuda.is_available():
            device = "cuda"
            is_fp16 = True
        else:
            device = "cpu"
            is_fp16 = False
            print("WARNING: Running on CPU. This will be significantly slower.")

        self.model = IndexTTS(
            model_dir=CHECKPOINT_DIR,
            cfg_path=os.path.join(CHECKPOINT_DIR, "config.yaml"),
            is_fp16=is_fp16,
            device=device
        )
        print(f"Model initialized successfully on {device}")

    @method()
    def generate_speech(
        self,
        text: str,
        reference_audio: bytes,
        use_fast: bool = True,
        output_format: str = "wav"
    ) -> bytes:
        """Generate speech from text using the provided reference audio"""
        try:
            if not text.strip():
                raise ValueError("Text input cannot be empty")

            # Save reference audio temporarily
            ref_path = "/tmp/reference.wav"
            with open(ref_path, "wb") as f:
                f.write(reference_audio)

            # Generate output path
            output_path = "/tmp/output.wav"

            # Generate speech using either fast or regular inference
            print(f"Generating speech using {'fast' if use_fast else 'regular'} inference...")
            if use_fast:
                self.model.infer_fast(
                    audio_prompt=ref_path,
                    text=text,
                    output_path=output_path,
                    verbose=True
                )
            else:
                self.model.infer(
                    audio_prompt=ref_path,
                    text=text,
                    output_path=output_path,
                    verbose=True
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
        required_files = ["config.yaml", "gpt.pth", "dvae.pth", "bigvgan_generator.pth"]
        files_exist = all(
            os.path.exists(os.path.join(CHECKPOINT_DIR, f)) 
            for f in required_files
        )
        
        return {
            "status": "healthy" if files_exist else "missing_files",
            "model_loaded": hasattr(self, "model"),
            "checkpoint_dir": os.path.exists(CHECKPOINT_DIR),
            "files_present": files_exist,
            "device": self.model.device if hasattr(self, "model") else None,
            "is_fp16": self.model.is_fp16 if hasattr(self, "model") else None
        }

@app.exception_handler(Exception)
def exception_handler(exc: Exception):
    """Handle exceptions globally"""
    print(f"Error occurred: {str(exc)}")
    return {"error": str(exc)}, 500

# Example usage with new Modal syntax
@app.local_entrypoint()
def main():
    with modal.enable_output():  # Enable Modal output logging
        # First check if service is healthy
        service = IndexTTSService()
        health_status = service.health_check()
        
        if not health_status["model_loaded"]:
            print("Error: Model not loaded properly")
            print("Health status:", health_status)
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
                reference_audio=reference_audio,
                use_fast=True
            )
            
            # Save output
            output_path = "output.wav"
            with open(output_path, "wb") as f:
                f.write(audio_data)
                
            print(f"Speech generated successfully! Saved to {output_path}")
            
        except Exception as e:
            print(f"Error during speech generation: {str(e)}")
            if hasattr(e, "__traceback__"):
                import traceback
                traceback.print_exc()
