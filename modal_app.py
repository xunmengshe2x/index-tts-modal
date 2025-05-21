import os
import modal

# Define a custom image with all dependencies
image = modal.Image.debian_slim().pip_install(
    "accelerate==0.25.0",
    "transformers==4.36.2",
    "tokenizers==0.15.0",
    "cn2an==0.5.22",
    "ffmpeg-python==0.2.0",
    "Cython==3.0.7",
    "g2p-en==2.1.0",
    "jieba==0.42.1",
    "keras==2.9.0",
    "numba==0.58.1",
    "numpy==1.26.2",
    "pandas==2.1.3",
    "matplotlib==3.8.2",
    "opencv-python==4.9.0.80",
    "vocos==0.1.0",
    "tensorboard==2.9.1",
    "omegaconf",
    "sentencepiece",
    "librosa",
    "tqdm",
    "torch",
    "torchaudio"
)

# Add CUDA support
image = image.apt_install("ffmpeg")

# Create a Modal volume to store model files
volume = modal.Volume.from_name("index-tts-models", create_if_missing=True)

# Create a Modal app
app = modal.App("index-tts-inference", image=image)

@app.function(
    gpu="A10G",  # You can change this to "T4", "A100", etc. based on your needs
    timeout=600,  # 10-minute timeout
    volumes={"/checkpoints": volume}
)
def download_models():
    """Download Index-TTS model files to the volume."""
    import subprocess
    import os
    
    # Create checkpoints directory if it doesn't exist
    os.makedirs("/checkpoints", exist_ok=True)
    
    # Check if models are already downloaded
    if os.path.exists("/checkpoints/gpt.pth"):
        print("Models already downloaded.")
        return
    
    # Download models using huggingface-cli or wget
    commands = [
        "wget https://huggingface.co/IndexTeam/IndexTTS-1.5/resolve/main/bigvgan_discriminator.pth -P /checkpoints",
        "wget https://huggingface.co/IndexTeam/IndexTTS-1.5/resolve/main/bigvgan_generator.pth -P /checkpoints",
        "wget https://huggingface.co/IndexTeam/IndexTTS-1.5/resolve/main/bpe.model -P /checkpoints",
        "wget https://huggingface.co/IndexTeam/IndexTTS-1.5/resolve/main/dvae.pth -P /checkpoints",
        "wget https://huggingface.co/IndexTeam/IndexTTS-1.5/resolve/main/gpt.pth -P /checkpoints",
        "wget https://huggingface.co/IndexTeam/IndexTTS-1.5/resolve/main/unigram_12000.vocab -P /checkpoints",
        "wget https://huggingface.co/IndexTeam/IndexTTS-1.5/resolve/main/config.yaml -P /checkpoints"
    ]
    
    for cmd in commands:
        subprocess.run(cmd, shell=True, check=True )
    
    print("Models downloaded successfully.")
    return True

@app.function(
    gpu="A10G",  # You can change this to "T4", "A100", etc. based on your needs
    timeout=600,  # 10-minute timeout
    volumes={"/checkpoints": volume}
)
def run_inference(
    text: str,
    voice_url: str,
    output_filename: str = "output.wav"
):
    """Run Index-TTS inference with the given text and voice prompt."""
    import os
    import subprocess
    import tempfile
    import urllib.request
    
    # Create a temporary directory for input/output files
    with tempfile.TemporaryDirectory() as temp_dir:
        # Download the voice prompt
        voice_path = os.path.join(temp_dir, "voice_prompt.wav")
        urllib.request.urlretrieve(voice_url, voice_path)
        
        # Set up output path
        output_path = os.path.join(temp_dir, output_filename)
        
        # Clone the Index-TTS repository
        subprocess.run(
            "git clone https://github.com/index-tts/index-tts.git",
            shell=True,
            check=True,
            cwd=temp_dir
         )
        
        # Run inference using the CLI
        cmd = [
            "python",
            "-m",
            "indextts.cli",
            f'"{text}"',
            "--voice", voice_path,
            "--output_path", output_path,
            "--config", "/checkpoints/config.yaml",
            "--model_dir", "/checkpoints",
            "--fp16"
        ]
        
        subprocess.run(
            " ".join(cmd),
            shell=True,
            check=True,
            cwd=os.path.join(temp_dir, "index-tts")
        )
        
        # Read the output file
        with open(output_path, "rb") as f:
            output_data = f.read()
        
        return output_data

@app.function(
    gpu="A10G",  # You can change this to "T4", "A100", etc. based on your needs
    timeout=600,  # 10-minute timeout
    volumes={"/checkpoints": volume}
)
def save_inference_result(
    text: str,
    voice_url: str,
    output_path: str
):
    """Run inference and save the result to a file."""
    import os
    
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Run inference
    output_data = run_inference.remote(text, voice_url)
    
    # Save the output
    with open(output_path, "wb") as f:
        f.write(output_data)
    
    return f"Saved output to {output_path}"

# Define a web endpoint for inference
@app.function(
    gpu="A10G",  # You can change this to "T4", "A100", etc. based on your needs
    timeout=600,  # 10-minute timeout
    volumes={"/checkpoints": volume}
)
@modal.web_endpoint(method="POST")
async def inference_api(request):
    """Web endpoint for Index-TTS inference."""
    import json
    import base64
    
    # Parse the request
    data = await request.json()
    text = data.get("text")
    voice_url = data.get("voice_url")
    
    if not text or not voice_url:
        return {"error": "Missing required parameters: text and voice_url"}
    
    # Run inference
    output_data = run_inference.remote(text, voice_url)
    
    # Encode the output as base64
    encoded_output = base64.b64encode(output_data).decode("utf-8")
    
    return {"audio_base64": encoded_output}

# Define a local entrypoint for running the script directly
@app.local_entrypoint()
def main(
    text: str = "Hello, this is a test of Index-TTS running on Modal.",
    voice_url: str = "https://example.com/voice_prompt.wav",
    output_path: str = "output.wav"
 ):
    """Local entrypoint for running inference."""
    # First, ensure models are downloaded
    download_models.remote()
    
    # Run inference and save the result
    result = save_inference_result.remote(text, voice_url, output_path)
    print(result)
