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
    "fastapi[standard]",
    "pydantic>=2.0.0",
    "typing-extensions"
)

# Add CUDA support, ffmpeg, wget, and git
image = image.apt_install("ffmpeg", "wget", "git")   

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
    import shutil

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
        cwd="/checkpoints"
     )

    print("Repository downloaded successfully.")
    return True

# CRITICAL: Create a class to manage persistent model state
@app.cls(
    gpu="A10G",
    timeout=1200,  # Increased timeout for model loading
    volumes={"/checkpoints": volume},
    container_idle_timeout=300,  # Keep container alive for 5 minutes
    allow_concurrent_inputs=10,  # Allow multiple concurrent requests
)
class IndexTTSModel:
    """Persistent model class that keeps the TTS model loaded in memory."""
    
    def __init__(self):
        self.tts = None
        self.model_initialized = False
        
    @modal.enter()
    def setup_model(self):
        """Initialize the model once when the container starts."""
        import importlib.util
        import logging
        
        # Set up logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        self.logger.info("Setting up IndexTTS model...")
        
        # Ensure dependencies are available
        self._ensure_dependencies()
        
        # Add the repository to Python path
        sys.path.append("/checkpoints/index-tts")
        os.chdir("/checkpoints")
        
        # Dynamically import the module
        module_path = "/checkpoints/index-tts/indextts/infer.py"
        spec = importlib.util.spec_from_file_location("indextts.infer", module_path)
        indextts_infer = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(indextts_infer)
        
        # Initialize IndexTTS model - this is the slow part we want to do once
        self.logger.info("Loading IndexTTS model (this may take a while)...")
        self.tts = indextts_infer.IndexTTS(cfg_path="/checkpoints/config.yaml", model_dir="/checkpoints")
        self.model_initialized = True
        self.logger.info("IndexTTS model loaded successfully!")
        
    def _ensure_dependencies(self):
        """Ensure models and repository are available."""
        import subprocess
        import shutil
        import concurrent.futures
        
        # Create checkpoints directory if it doesn't exist
        os.makedirs("/checkpoints", exist_ok=True)

        # Download models if not already present
        if not os.path.exists("/checkpoints/gpt.pth"):
            self.logger.info("Downloading models...")
            model_urls = [
                ("https://huggingface.co/IndexTeam/IndexTTS-1.5/resolve/main/bigvgan_discriminator.pth", "bigvgan_discriminator.pth"),
                ("https://huggingface.co/IndexTeam/IndexTTS-1.5/resolve/main/bigvgan_generator.pth", "bigvgan_generator.pth"),
                ("https://huggingface.co/IndexTeam/IndexTTS-1.5/resolve/main/bpe.model", "bpe.model"),
                ("https://huggingface.co/IndexTeam/IndexTTS-1.5/resolve/main/dvae.pth", "dvae.pth"),
                ("https://huggingface.co/IndexTeam/IndexTTS-1.5/resolve/main/gpt.pth", "gpt.pth"),
                ("https://huggingface.co/IndexTeam/IndexTTS-1.5/resolve/main/unigram_12000.vocab", "unigram_12000.vocab"),
                ("https://huggingface.co/IndexTeam/IndexTTS-1.5/resolve/main/config.yaml", "config.yaml")
            ]

            def download_model(url, filename):
                subprocess.run(f"wget {url} -P /checkpoints", shell=True, check=True)
                self.logger.info(f"Downloaded {filename}")

            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = [executor.submit(download_model, url, filename) for url, filename in model_urls]
                concurrent.futures.wait(futures)
            self.logger.info("Models downloaded successfully.")

        # Download repository if not already present
        repo_dir = "/checkpoints/index-tts"
        if not os.path.exists(repo_dir):
            self.logger.info("Cloning repository...")
            subprocess.run(
                f"git clone https://github.com/xunmengshe2x/index-tts-modal.git {repo_dir}",
                shell=True,
                check=True,
                cwd="/checkpoints"
            )
            self.logger.info("Repository downloaded successfully.")

    @modal.method()
    def generate_audio_chunk(self, voice_base64: str, text: str, chunk_index: int, 
                           max_text_tokens_per_sentence: int = 300, 
                           sentences_bucket_max_size: int = 8):
        """Generate audio for a single text chunk using the persistent model."""
        if not self.model_initialized:
            raise RuntimeError("Model not initialized")
            
        # Create inputs directory
        inputs_dir = "/checkpoints/inputs"
        os.makedirs(inputs_dir, exist_ok=True)
        
        # Create a unique voice file for this request to avoid conflicts
        voice_path = os.path.join(inputs_dir, f"voice_prompt_{chunk_index}.wav")
        
        try:
            # Write the voice file
            with open(voice_path, "wb") as temp_file:
                temp_file.write(base64.b64decode(voice_base64))
            
            # Set up output path
            outputs_dir = "/checkpoints/outputs"
            os.makedirs(outputs_dir, exist_ok=True)
            chunk_output_path = os.path.join(outputs_dir, f"chunk_{chunk_index}.wav")
            
            # Generate audio using the persistent model
            self.tts.infer_fast(
                audio_prompt=voice_path,
                text=text,
                output_path=chunk_output_path,
                max_text_tokens_per_sentence=max_text_tokens_per_sentence,
                sentences_bucket_max_size=sentences_bucket_max_size
            )
            
            # Read the generated audio
            with open(chunk_output_path, "rb") as f:
                audio_data = f.read()
            
            return audio_data
            
        finally:
            # Clean up temporary files
            for temp_file in [voice_path, chunk_output_path]:
                if os.path.exists(temp_file):
                    os.remove(temp_file)

# Create a persistent model instance
model_instance = IndexTTSModel()

@app.function()
@modal.fastapi_endpoint(method="POST")
async def inference_api_optimized(request: Request):
    """Optimized streaming endpoint using persistent model."""
    from fastapi.responses import StreamingResponse
    import json
    
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Parse the request body
    data = await request.json()
    text = data.get("text")
    voice_base64 = data.get("voice_base64")
    chunk_size = data.get("chunk_size", 60)
    max_text_tokens_per_sentence = data.get("max_text_tokens_per_sentence", 300)
    sentences_bucket_max_size = data.get("sentences_bucket_max_size", 8)

    if not text or not voice_base64:
        return {"error": "Missing required parameters: text and voice_base64"}

    # Split text into chunks
    chunks = split_into_chunks(text, max_chunk_size=chunk_size)
    logger.info(f"Split text into {len(chunks)} chunks")

    async def generate_chunks():
        """Generator that yields audio chunks as they're produced."""
        try:
            # Send initial metadata
            metadata = {
                "type": "metadata",
                "total_chunks": len(chunks),
                "chunk_size": chunk_size
            }
            yield json.dumps(metadata) + "\n"
            
            for idx, chunk in enumerate(chunks):
                if not chunk or chunk.isspace():
                    logger.warning(f"Skipping empty chunk {idx}")
                    continue
                    
                try:
                    # Clean the chunk text
                    chunk = chunk.strip()
                    if not chunk:
                        logger.warning(f"Skipping empty chunk after cleaning {idx}")
                        continue
                        
                    logger.info(f"Processing chunk {idx}/{len(chunks)}: {chunk[:50]}...")
                    
                    # Generate audio using the persistent model - this should be MUCH faster now
                    audio_data = model_instance.generate_audio_chunk.remote(
                        voice_base64=voice_base64,
                        text=chunk,
                        chunk_index=idx,
                        max_text_tokens_per_sentence=max_text_tokens_per_sentence,
                        sentences_bucket_max_size=sentences_bucket_max_size
                    )
                    
                    # Stream the chunk
                    chunk_response = {
                        "type": "chunk",
                        "chunk_index": idx,
                        "total_chunks": len(chunks),
                        "text": chunk,
                        "audio_base64": base64.b64encode(audio_data).decode("utf-8"),
                        "is_final": idx == len(chunks) - 1
                    }
                    
                    logger.info(f"Streaming chunk {idx} ({len(audio_data)} bytes)")
                    yield json.dumps(chunk_response) + "\n"
                    
                    # Small delay to ensure proper streaming
                    await asyncio.sleep(0.1)
                    
                except Exception as e:
                    error_msg = f"Error processing chunk {idx}: {str(e)}"
                    logger.error(error_msg)
                    yield json.dumps({
                        "type": "error",
                        "chunk_index": idx,
                        "error": error_msg
                    }) + "\n"

            # Send completion signal
            completion = {
                "type": "complete",
                "total_chunks_processed": len([c for c in chunks if c.strip()])
            }
            yield json.dumps(completion) + "\n"
            
        except Exception as e:
            logger.error(f"Stream generation error: {str(e)}")
            yield json.dumps({
                "type": "error", 
                "error": f"Stream generation failed: {str(e)}"
            }) + "\n"

    return StreamingResponse(
        generate_chunks(),
        media_type="application/x-ndjson",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )

# Keep the old endpoint for backward compatibility
@app.function(
    gpu="A10G",
    timeout=900,
    volumes={"/checkpoints": volume}
)
@modal.fastapi_endpoint(method="POST")
async def inference_api_with_file(request: Request):
    """Original endpoint - kept for backward compatibility."""
    # [Keep your original implementation here if needed]
    return {"message": "Please use the optimized endpoint: /inference_api_optimized"}

# Define a health check endpoint
@app.function()
@modal.fastapi_endpoint(method="GET")
async def health():
    """Health check endpoint."""
    return {"status": "ok", "service": "index-tts-inference"}
