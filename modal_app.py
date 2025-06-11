import concurrent.futures
import os
import modal
from fastapi import Request
# Additional imports needed for chunking
import base64
import io
import re
import sys
import tempfile
import urllib.request
from typing import List
import numpy as np
import torch
import torchaudio
import logging
import subprocess
import asyncio

# Define a custom image with all dependencies
image = modal.Image.debian_slim().pip_install(
    "ninja-build",  # Add before deepspeed
    #"deepspeed",    # Install after ninja-build
    "deepspeed",  # Add DeepSpeed
    #"accelerate==0.25.0",
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
    "torchaudio",
    "WeTextProcessing",
    "fastapi[standard]",  # Required for web endpoints
    "pydantic>=2.0.0",    # Explicitly add Pydantic
    "typing-extensions"   # Often needed with Pydantic
)

# Add CUDA support, ffmpeg, wget, and git
image = image.apt_install("ffmpeg", "wget", "git", "build-essential", "ninja-build", "cuda-nvcc-11-8")   

# Create a Modal volume to store model files
volume = modal.Volume.from_name("index-tts-models", create_if_missing=True)

# Create a Modal app
app = modal.App("index-tts-inference", image=image)

def split_into_chunks(text: str, max_chunk_size: int = 20) -> List[str]:
    """Split text into chunks based on sentences and maximum chunk size."""
    # Clean input text
    text = text.strip()
    if not text:
        return []
        
    # Split on sentence boundaries
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]
    chunks = []
    current_chunk = []
    current_length = 0
    
    for sentence in sentences:
        sentence_length = len(sentence.split())
        
        if current_length + sentence_length > max_chunk_size:
            if current_chunk:
                chunks.append(' '.join(current_chunk))
            current_chunk = [sentence]
            current_length = sentence_length
        else:
            current_chunk.append(sentence)
            current_length += sentence_length
    
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return [chunk for chunk in chunks if chunk.strip()]

def concatenate_audio_files(audio_files: List[bytes]) -> bytes:
    """Concatenate multiple audio files into a single audio file."""
    if not audio_files:
        raise ValueError("No audio files to concatenate")
        
    waveforms = []
    sample_rate = None
    
    for audio_data in audio_files:
        if not audio_data:
            continue
            
        audio_io = io.BytesIO(audio_data)
        try:
            waveform, sr = torchaudio.load(audio_io)
            
            if sample_rate is None:
                sample_rate = sr
            elif sr != sample_rate:
                raise ValueError("All audio files must have the same sample rate")
            
            waveforms.append(waveform)
        except Exception as e:
            logging.error(f"Error loading audio data: {str(e)}")
            continue
    
    if not waveforms:
        raise ValueError("No valid waveforms to concatenate")
        
    concatenated = torch.cat(waveforms, dim=1)
    output_buffer = io.BytesIO()
    torchaudio.save(output_buffer, concatenated, sample_rate, format="wav")
    return output_buffer.getvalue()

@app.function(
    gpu="A10G",
    timeout=600,
    volumes={"/checkpoints": volume}
)
def download_models():
    """Download Index-TTS model files to the volume."""
    # Create checkpoints directory if it doesn't exist
    os.makedirs("/checkpoints", exist_ok=True)

    # Check if models are already downloaded
    if os.path.exists("/checkpoints/gpt.pth"):
        print("Models already downloaded.")
        return

    # List of model URLs and their corresponding filenames
    model_urls = [
        ("https://huggingface.co/IndexTeam/IndexTTS-1.5/resolve/main/bigvgan_discriminator.pth", "bigvgan_discriminator.pth"),
        ("https://huggingface.co/IndexTeam/IndexTTS-1.5/resolve/main/bigvgan_generator.pth", "bigvgan_generator.pth"),
        ("https://huggingface.co/IndexTeam/IndexTTS-1.5/resolve/main/bpe.model", "bpe.model"),
        ("https://huggingface.co/IndexTeam/IndexTTS-1.5/resolve/main/dvae.pth", "dvae.pth"),
        ("https://huggingface.co/IndexTeam/IndexTTS-1.5/resolve/main/gpt.pth", "gpt.pth"),
        ("https://huggingface.co/IndexTeam/IndexTTS-1.5/resolve/main/unigram_12000.vocab", "unigram_12000.vocab"),
        ("https://huggingface.co/IndexTeam/IndexTTS-1.5/resolve/main/config.yaml", "config.yaml")
    ]

    # Function to download a single model file
    def download_model(url, filename):
        subprocess.run(f"wget {url} -P /checkpoints", shell=True, check=True)
        print(f"Downloaded {filename}")

    # Use ThreadPoolExecutor to download models in parallel
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(download_model, url, filename) for url, filename in model_urls]
        concurrent.futures.wait(futures)

    print("Models downloaded successfully.")
    return True

@app.function(
    gpu="A10G",
    timeout=600,
    volumes={"/checkpoints": volume}
)
def download_repository():
    """Download the Index-TTS repository to the volume."""
    import subprocess
    import os
    import shutil # This import is also needed inside the function for Modal's environment

    # Define the target directory for the repository
    repo_dir = "/checkpoints/index-tts"
    
    # Ensure the /checkpoints directory exists
    os.makedirs("/checkpoints", exist_ok=True)

    # If the repository directory already exists, remove it to ensure a fresh clone
    if os.path.exists(repo_dir):
        print(f"Removing existing repository at {repo_dir}...")
        shutil.rmtree(repo_dir)
        print("Existing repository removed.")

    print(f"Cloning repository into {repo_dir}...")
    # Clone your repository into the specified directory name 'index-tts'
    subprocess.run(
        f"git clone https://github.com/xunmengshe2x/index-tts-modal.git {repo_dir}",
        shell=True,
        check=True,
        cwd="/checkpoints" # This ensures the clone command is run from /checkpoints
     )

    print("Repository downloaded successfully.")
    return True

@app.function(
    gpu="A10G",
    timeout=600,
    volumes={"/checkpoints": volume}
)
def run_inference(
    text: str,
    voice_path: str,
    output_filename: str = "output.wav",
    is_url: bool = True
):
    """Run Index-TTS inference with the given text and voice prompt."""
    import os
    import subprocess
    import urllib.request
    import logging
    import sys

    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Create inputs directory if it doesn't exist
    inputs_dir = "/checkpoints/inputs"
    os.makedirs(inputs_dir, exist_ok=True)

    # Handle voice prompt (either from URL or local path)
    local_voice_path = os.path.join(inputs_dir, "voice_prompt.wav")
    if is_url:
        urllib.request.urlretrieve(voice_path, local_voice_path)
    else:
        local_voice_path = voice_path

    # Debug: Check if the voice prompt file exists
    if not os.path.exists(local_voice_path):
        logger.error(f"Voice prompt file does not exist: {local_voice_path}")
        raise FileNotFoundError(f"Voice prompt file does not exist: {local_voice_path}")

    # Set up output path
    outputs_dir = "/checkpoints/outputs"
    os.makedirs(outputs_dir, exist_ok=True)
    output_path = os.path.join(outputs_dir, output_filename)

    # Add the cloned repository to the Python path
    sys.path.append("/checkpoints/index-tts")

    # Initialize IndexTTS
    from indextts.infer import IndexTTS
    tts = IndexTTS(cfg_path="/checkpoints/config.yaml", model_dir="/checkpoints")

    # Run inference
    tts.infer(audio_prompt=local_voice_path, text=text, output_path=output_path)

    # Debug: Check if the output file exists
    if not os.path.exists(output_path):
        logger.error(f"Output file does not exist: {output_path}")
        raise FileNotFoundError(f"Output file does not exist: {output_path}")

    # Read the output file
    with open(output_path, "rb") as f:
        output_data = f.read()

    return output_data

@app.function(
    gpu="A10G",
    timeout=600,
    volumes={"/checkpoints": volume}
)
@modal.fastapi_endpoint(method="POST")
async def inference_api(request: Request):
    """Web endpoint for Index-TTS inference using a voice URL."""
    import base64

    # Parse the request body
    data = await request.json()
    text = data.get("text")
    voice_url = data.get("voice_url")

    if not text or not voice_url:
        return {"error": "Missing required parameters: text and voice_url"}

    # Ensure models and repository are downloaded
    download_models.remote()
    download_repository.remote()

    # Run inference
    output_data = run_inference.remote(text, voice_url, is_url=True)

    # Encode the output as base64
    encoded_output = base64.b64encode(output_data).decode("utf-8")

    return {"audio_base64": encoded_output}

@app.function(
    gpu="A10G",
    timeout=600,
    volumes={"/checkpoints": volume}
)
@modal.fastapi_endpoint(method="POST")
async def inference_api_with_file(request: Request):
    """Web endpoint for Index-TTS inference with direct file upload."""
    import base64
    import os
    import urllib.request
    import logging
    import sys
    import importlib.util

    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Parse the request body
    data = await request.json()
    text = data.get("text")
    voice_base64 = data.get("voice_base64")
    chunk_size = data.get("chunk_size", 20)  # Default changed to 20
    max_text_tokens_per_sentence = data.get("max_text_tokens_per_sentence", 100)
    sentences_bucket_max_size = data.get("sentences_bucket_max_size", 4)

    if not text or not voice_base64:
        return {"error": "Missing required parameters: text and voice_base64"}

    #download_repository.remote()
    # Create inputs directory if it doesn't exist
    inputs_dir = "/checkpoints/inputs"
    os.makedirs(inputs_dir, exist_ok=True)
    voice_path = os.path.join(inputs_dir, "voice_prompt.wav")

    with open(voice_path, "wb") as temp_file:
        temp_file.write(base64.b64decode(voice_base64))

    try:
        # Debug: Check if the voice prompt file exists
        if not os.path.exists(voice_path):
            logger.error(f"Voice prompt file does not exist: {voice_path}")
            raise FileNotFoundError(f"Voice prompt file does not exist: {voice_path}")

        # Set up output paths
        outputs_dir = "/checkpoints/outputs"
        os.makedirs(outputs_dir, exist_ok=True)

        # Add the cloned repository to the Python path
        sys.path.append("/checkpoints/index-tts")

        # Change the current working directory to /checkpoints
        os.chdir("/checkpoints")

        # Print the contents of the current directory
        current_dir = os.getcwd()
        print(f"Contents of current directory {current_dir}: {os.listdir(current_dir)}")

        # Dynamically import the module
        module_path = "/checkpoints/index-tts/indextts/infer.py"
        spec = importlib.util.spec_from_file_location("indextts.infer", module_path)
        indextts_infer = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(indextts_infer)

        # Initialize IndexTTS
        tts = indextts_infer.IndexTTS(cfg_path="/checkpoints/config.yaml", model_dir="/checkpoints")

        # Split text into chunks
        chunks = split_into_chunks(text, max_chunk_size=chunk_size)
        logger.info(f"Split text into {len(chunks)} chunks")

        # Process chunks sequentially
        audio_chunks = []

        for idx, chunk in enumerate(chunks):
            if not chunk or chunk.isspace():
                logger.warning(f"Skipping empty chunk {idx}")
                continue
                
            chunk_output = f"chunk_{idx}.wav"
            chunk_output_path = os.path.join(outputs_dir, chunk_output)
            
            try:
                # Clean the chunk text
                chunk = chunk.strip()
                if not chunk:
                    logger.warning(f"Skipping empty chunk after cleaning {idx}")
                    continue
                    
                tts.infer_fast(audio_prompt=voice_path, text=chunk, output_path=chunk_output_path, max_text_tokens_per_sentence=max_text_tokens_per_sentence, sentences_bucket_max_size=sentences_bucket_max_size)
                
                if not os.path.exists(chunk_output_path):
                    logger.error(f"Output file not created for chunk {idx}")
                    continue
                    
                with open(chunk_output_path, "rb") as f:
                    audio_data = f.read()
                
                audio_chunks.append((idx, audio_data))
            except Exception as e:
                logger.error(f"Error processing chunk {idx}: {str(e)}")
            finally:
                if os.path.exists(chunk_output_path):
                    os.remove(chunk_output_path)

        # Check if we have any successful chunks
        if not audio_chunks:
            return {"error": "No audio chunks were successfully processed"}

        # Sort chunks by index and concatenate
        audio_chunks.sort(key=lambda x: x[0])
        final_audio = concatenate_audio_files([chunk[1] for chunk in audio_chunks])

        # Encode the output as base64
        encoded_output = base64.b64encode(final_audio).decode("utf-8")

        return {"audio_base64": encoded_output}
    finally:
        # Clean up the file
        if os.path.exists(voice_path):
            os.remove(voice_path)

# Define a health check endpoint
@app.function()
@modal.fastapi_endpoint(method="GET")
async def health():
    """Health check endpoint."""
    return {"status": "ok", "service": "index-tts-inference"}
