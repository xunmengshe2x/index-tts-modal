import concurrent.futures
import os
import modal
from fastapi import Request
from fastapi.responses import StreamingResponse
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
import json
import shutil
import importlib.util

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
    os.makedirs("/checkpoints", exist_ok=True)

    if os.path.exists("/checkpoints/gpt.pth"):
        print("Models already downloaded.")
        return

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
        print(f"Downloaded {filename}")

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
    repo_dir = "/checkpoints/index-tts"
    os.makedirs("/checkpoints", exist_ok=True)

    if os.path.exists(repo_dir):
        print(f"Removing existing repository at {repo_dir}...")
        shutil.rmtree(repo_dir)
        print("Existing repository removed.")

    print(f"Cloning repository into {repo_dir}...")
    subprocess.run(
        f"git clone https://github.com/xunmengshe2x/index-tts-modal.git {repo_dir}",
        shell=True,
        check=True,
        cwd="/checkpoints"
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

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    inputs_dir = "/checkpoints/inputs"
    os.makedirs(inputs_dir, exist_ok=True)

    local_voice_path = os.path.join(inputs_dir, "voice_prompt.wav")
    if is_url:
        urllib.request.urlretrieve(voice_path, local_voice_path)
    else:
        local_voice_path = voice_path

    if not os.path.exists(local_voice_path):
        logger.error(f"Voice prompt file does not exist: {local_voice_path}")
        raise FileNotFoundError(f"Voice prompt file does not exist: {local_voice_path}")

    outputs_dir = "/checkpoints/outputs"
    os.makedirs(outputs_dir, exist_ok=True)
    output_path = os.path.join(outputs_dir, output_filename)

    sys.path.append("/checkpoints/index-tts")
    from indextts.infer import IndexTTS
    tts = IndexTTS(cfg_path="/checkpoints/config.yaml", model_dir="/checkpoints")
    tts.infer(audio_prompt=local_voice_path, text=text, output_path=output_path)

    if not os.path.exists(output_path):
        logger.error(f"Output file does not exist: {output_path}")
        raise FileNotFoundError(f"Output file does not exist: {output_path}")

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
    data = await request.json()
    text = data.get("text")
    voice_url = data.get("voice_url")

    if not text or not voice_url:
        return {"error": "Missing required parameters: text and voice_url"}

    download_models.remote()
    download_repository.remote()
    output_data = run_inference.remote(text, voice_url, is_url=True)
    encoded_output = base64.b64encode(output_data).decode("utf-8")

    return {"audio_base64": encoded_output}

@app.function(
    gpu="A10G",
    timeout=900,
    volumes={"/checkpoints": volume},
    keep_warm=1  # Keep one instance warm for faster response
)
@modal.fastapi_endpoint(method="POST")
async def inference_api_with_file(request: Request):
    """Web endpoint for Index-TTS inference with direct file upload - TRUE STREAMING VERSION."""
    import sys
    import os
    import logging
    import importlib.util
    import json
    import base64
    import asyncio
    import subprocess
    import concurrent.futures

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

    async def generate_chunks():
        """Generator that yields audio chunks as they're produced - TRUE STREAMING."""
        tts = None
        voice_path = None
        
        try:
            # Step 1: Setup (done once, before streaming starts)
            logger.info("Setting up models and environment...")
            
            # Ensure models and repository are available
            os.makedirs("/checkpoints", exist_ok=True)

            # Download models if not already present (this blocks initially but only once)
            if not os.path.exists("/checkpoints/gpt.pth"):
                logger.info("Downloading models...")
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
                    logger.info(f"Downloaded {filename}")

                with concurrent.futures.ThreadPoolExecutor() as executor:
                    futures = [executor.submit(download_model, url, filename) for url, filename in model_urls]
                    concurrent.futures.wait(futures)
                logger.info("Models downloaded successfully.")

            # Download repository if not already present
            repo_dir = "/checkpoints/index-tts"
            if not os.path.exists(repo_dir):
                logger.info("Cloning repository...")
                subprocess.run(
                    f"git clone https://github.com/xunmengshe2x/index-tts-modal.git {repo_dir}",
                    shell=True,
                    check=True,
                    cwd="/checkpoints"
                )
                logger.info("Repository downloaded successfully.")

            # Setup directories and voice file
            inputs_dir = "/checkpoints/inputs"
            outputs_dir = "/checkpoints/outputs"
            os.makedirs(inputs_dir, exist_ok=True)
            os.makedirs(outputs_dir, exist_ok=True)
            
            # Create unique voice file to avoid conflicts
            import time
            voice_filename = f"voice_prompt_{int(time.time())}.wav"
            voice_path = os.path.join(inputs_dir, voice_filename)
            
            with open(voice_path, "wb") as temp_file:
                temp_file.write(base64.b64decode(voice_base64))

            if not os.path.exists(voice_path):
                raise FileNotFoundError(f"Failed to create voice prompt file: {voice_path}")

            file_size = os.path.getsize(voice_path)
            logger.info(f"Voice file created: {voice_path} (size: {file_size} bytes)")

            # Initialize TTS model
            sys.path.append("/checkpoints/index-tts")
            os.chdir("/checkpoints")

            module_path = "/checkpoints/index-tts/indextts/infer.py"
            spec = importlib.util.spec_from_file_location("indextts.infer", module_path)
            indextts_infer = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(indextts_infer)

            tts = indextts_infer.IndexTTS(cfg_path="/checkpoints/config.yaml", model_dir="/checkpoints")
            logger.info("TTS model initialized successfully.")

            # Split text into chunks
            chunks = split_into_chunks(text, max_chunk_size=chunk_size)
            logger.info(f"Text split into {len(chunks)} chunks")

            # Step 2: Send metadata immediately to start the stream
            metadata = {
                "type": "metadata",
                "total_chunks": len(chunks),
                "chunk_size": chunk_size,
                "status": "initialized"
            }
            yield json.dumps(metadata) + "\n"
            
            # Force flush the first message
            await asyncio.sleep(0.01)

            # Step 3: Process chunks one by one and stream each immediately
            for idx, chunk in enumerate(chunks):
                if not chunk or chunk.isspace():
                    logger.warning(f"Skipping empty chunk {idx}")
                    continue
                    
                chunk = chunk.strip()
                if not chunk:
                    continue
                
                try:
                    logger.info(f"Processing chunk {idx + 1}/{len(chunks)}: '{chunk[:50]}...'")
                    
                    # Generate unique filename for this chunk
                    chunk_filename = f"chunk_{idx}_{int(time.time())}.wav"
                    chunk_output_path = os.path.join(outputs_dir, chunk_filename)
                    
                    # Send processing update
                    processing_update = {
                        "type": "processing",
                        "chunk_index": idx,
                        "total_chunks": len(chunks),
                        "text": chunk,
                        "status": "generating_audio"
                    }
                    yield json.dumps(processing_update) + "\n"
                    
                    # This is the expensive operation - generate audio for this chunk
                    tts.infer_fast(
                        audio_prompt=voice_path,
                        text=chunk,
                        output_path=chunk_output_path,
                        max_text_tokens_per_sentence=max_text_tokens_per_sentence,
                        sentences_bucket_max_size=sentences_bucket_max_size
                    )
                    
                    if not os.path.exists(chunk_output_path):
                        error_msg = f"Output file not created for chunk {idx}"
                        logger.error(error_msg)
                        yield json.dumps({
                            "type": "error",
                            "chunk_index": idx,
                            "error": error_msg
                        }) + "\n"
                        continue
                        
                    # Read and encode the audio data
                    with open(chunk_output_path, "rb") as f:
                        audio_data = f.read()
                    
                    # Stream this chunk immediately after generation
                    chunk_response = {
                        "type": "chunk",
                        "chunk_index": idx,
                        "total_chunks": len(chunks),
                        "text": chunk,
                        "audio_base64": base64.b64encode(audio_data).decode("utf-8"),
                        "is_final": idx == len(chunks) - 1,
                        "audio_size_bytes": len(audio_data)
                    }
                    
                    logger.info(f"Streaming chunk {idx + 1}/{len(chunks)} ({len(audio_data)} bytes)")
                    yield json.dumps(chunk_response) + "\n"
                    
                    # Clean up chunk file immediately
                    if os.path.exists(chunk_output_path):
                        os.remove(chunk_output_path)
                    
                    # Small async sleep to ensure proper streaming and prevent blocking
                    await asyncio.sleep(0.01)
                    
                except Exception as e:
                    error_msg = f"Error processing chunk {idx}: {str(e)}"
                    logger.error(error_msg)
                    yield json.dumps({
                        "type": "error",
                        "chunk_index": idx,
                        "error": error_msg
                    }) + "\n"
                    
                    # Clean up on error
                    chunk_output_path = os.path.join(outputs_dir, f"chunk_{idx}_{int(time.time())}.wav")
                    if os.path.exists(chunk_output_path):
                        os.remove(chunk_output_path)

            # Send completion signal
            completion = {
                "type": "complete",
                "total_chunks_processed": len([c for c in chunks if c.strip()]),
                "status": "finished"
            }
            yield json.dumps(completion) + "\n"
            
        except Exception as e:
            logger.error(f"Stream generation error: {str(e)}")
            yield json.dumps({
                "type": "error", 
                "error": f"Stream generation failed: {str(e)}"
            }) + "\n"
        finally:
            # Clean up voice file
            if voice_path and os.path.exists(voice_path):
                logger.info(f"Cleaning up voice file: {voice_path}")
                try:
                    os.remove(voice_path)
                except Exception as e:
                    logger.warning(f"Failed to clean up voice file: {e}")

    return StreamingResponse(
        generate_chunks(),
        media_type="application/x-ndjson",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable nginx buffering
            "Transfer-Encoding": "chunked"  # Ensure chunked encoding
        }
    )

# Define a health check endpoint
@app.function()
@modal.fastapi_endpoint(method="GET")
async def health():
    """Health check endpoint."""
    return {"status": "ok", "service": "index-tts-inference"}
