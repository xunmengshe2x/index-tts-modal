import base64
import io
import modal
import os
import re
import sys
import tempfile
import urllib.request
from typing import List
import numpy as np
import torch
import torchaudio
from concurrent.futures import ThreadPoolExecutor, as_completed
from fastapi import Request
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

stub = modal.Stub("index-tts-inference")
volume = modal.NetworkFileSystem.persisted("index-tts-checkpoints")

def download_checkpoints():
    os.makedirs("/checkpoints", exist_ok=True)
    
    # Define checkpoint files and their URLs
    checkpoints = {
        "bigvgan_discriminator.pth": "https://huggingface.co/hf-internal-testing/index-tts/resolve/main/bigvgan_discriminator.pth",
        "bigvgan_generator.pth": "https://huggingface.co/hf-internal-testing/index-tts/resolve/main/bigvgan_generator.pth",
        "bpe.model": "https://huggingface.co/hf-internal-testing/index-tts/resolve/main/bpe.model",
        "dvae.pth": "https://huggingface.co/hf-internal-testing/index-tts/resolve/main/dvae.pth",
        "gpt.pth": "https://huggingface.co/hf-internal-testing/index-tts/resolve/main/gpt.pth",
        "unigram_12000.vocab": "https://huggingface.co/hf-internal-testing/index-tts/resolve/main/unigram_12000.vocab",
        "config.yaml": "https://huggingface.co/hf-internal-testing/index-tts/resolve/main/config.yaml",
    }
    
    for filename, url in checkpoints.items():
        filepath = f"/checkpoints/{filename}"
        if not os.path.exists(filepath):
            print(f"Downloading {filename}...")
            urllib.request.urlretrieve(url, filepath)

def split_into_chunks(text: str, max_chunk_size: int = 500) -> List[str]:
    """Split text into chunks based on sentences and maximum chunk size."""
    # Split into sentences first
    sentences = re.split(r'(?<=[.!?])\s+', text)
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
    
    return chunks

def concatenate_audio_files(audio_files: List[bytes]) -> bytes:
    """Concatenate multiple audio files into a single audio file."""
    waveforms = []
    sample_rate = None
    
    for audio_data in audio_files:
        # Create a temporary file-like object
        audio_io = io.BytesIO(audio_data)
        waveform, sr = torchaudio.load(audio_io)
        
        if sample_rate is None:
            sample_rate = sr
        elif sr != sample_rate:
            raise ValueError("All audio files must have the same sample rate")
        
        waveforms.append(waveform)
    
    # Concatenate along time dimension
    concatenated = torch.cat(waveforms, dim=1)
    
    # Convert back to bytes
    output_buffer = io.BytesIO()
    torchaudio.save(output_buffer, concatenated, sample_rate, format="wav")
    return output_buffer.getvalue()

image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        "torch==2.0.1",
        "torchaudio==2.0.2",
        "transformers",
        "accelerate",
        "tokenizers",
        "matplotlib",
        "numpy",
        "librosa",
        "pandas",
        "requests",
        "fastapi",
        "python-multipart",
        "sentencepiece",
        "wget",
    )
    .pip_install("git+https://github.com/index-tts/index-tts.git")
    .run_function(download_checkpoints)
)

@stub.function(
    gpu="A10G",
    timeout=600,
    image=image,
    volumes={"/checkpoints": volume}
)
def run_inference(
    text: str,
    voice_path: str,
    output_filename: str = "output.wav",
    is_url: bool = True,
    chunk_size: int = 500,
    max_parallel_chunks: int = 4
):
    """Run Index-TTS inference with parallel processing for long texts."""
    import os
    import urllib.request
    import logging
    import sys
    
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Handle voice prompt
    inputs_dir = "/checkpoints/inputs"
    os.makedirs(inputs_dir, exist_ok=True)
    local_voice_path = os.path.join(inputs_dir, "voice_prompt.wav")
    
    if is_url:
        urllib.request.urlretrieve(voice_path, local_voice_path)
    else:
        local_voice_path = voice_path
    
    if not os.path.exists(local_voice_path):
        raise FileNotFoundError(f"Voice prompt file does not exist: {local_voice_path}")
    
    # Set up output paths
    outputs_dir = "/checkpoints/outputs"
    os.makedirs(outputs_dir, exist_ok=True)
    
    # Initialize TTS model
    sys.path.append("/checkpoints/index-tts")
    from indextts.infer import IndexTTS
    tts = IndexTTS(cfg_path="/checkpoints/config.yaml", model_dir="/checkpoints")
    
    # Split text into chunks
    chunks = split_into_chunks(text, max_chunk_size=chunk_size)
    logger.info(f"Split text into {len(chunks)} chunks")
    
    # Process chunks in parallel
    audio_chunks = []
    
    def process_chunk(chunk: str, chunk_idx: int) -> bytes:
        chunk_output = f"chunk_{chunk_idx}.wav"
        chunk_output_path = os.path.join(outputs_dir, chunk_output)
        tts.infer(audio_prompt=local_voice_path, text=chunk, output_path=chunk_output_path)
        
        with open(chunk_output_path, "rb") as f:
            audio_data = f.read()
        
        # Cleanup
        os.remove(chunk_output_path)
        return audio_data
    
    # Process chunks in batches
    for i in range(0, len(chunks), max_parallel_chunks):
        batch_chunks = chunks[i:i + max_parallel_chunks]
        
        with ThreadPoolExecutor(max_workers=max_parallel_chunks) as executor:
            future_to_chunk = {
                executor.submit(process_chunk, chunk, idx + i): idx + i
                for idx, chunk in enumerate(batch_chunks)
            }
            
            for future in as_completed(future_to_chunk):
                chunk_idx = future_to_chunk[future]
                try:
                    audio_data = future.result()
                    audio_chunks.append((chunk_idx, audio_data))
                except Exception as e:
                    logger.error(f"Error processing chunk {chunk_idx}: {str(e)}")
                    raise
    
    # Sort chunks by index and concatenate
    audio_chunks.sort(key=lambda x: x[0])
    final_audio = concatenate_audio_files([chunk[1] for chunk in audio_chunks])
    
    # Save final output
    output_path = os.path.join(outputs_dir, output_filename)
    with open(output_path, "wb") as f:
        f.write(final_audio)
    
    return final_audio

@stub.function(
    gpu="A10G",
    timeout=600,
    image=image,
    volumes={"/checkpoints": volume}
)
@modal.asgi_app()
def fastapi_app():
    from fastapi import FastAPI
    app = FastAPI()

    @app.post("/inference")
    async def inference_api(request: Request):
        data = await request.json()
        text = data.get("text")
        voice_url = data.get("voice_url")
        chunk_size = data.get("chunk_size", 500)
        max_parallel_chunks = data.get("max_parallel_chunks", 4)
        
        if not text or not voice_url:
            return {"error": "Missing required parameters"}
        
        try:
            output_data = run_inference(
                text=text,
                voice_path=voice_url,
                chunk_size=chunk_size,
                max_parallel_chunks=max_parallel_chunks
            )
            return {"audio": base64.b64encode(output_data).decode()}
        except Exception as e:
            return {"error": str(e)}

    @app.post("/inference_with_file")
    async def inference_api_with_file(request: Request):
        data = await request.json()
        text = data.get("text")
        voice_base64 = data.get("voice_base64")
        chunk_size = data.get("chunk_size", 500)
        max_parallel_chunks = data.get("max_parallel_chunks", 4)
        
        if not text or not voice_base64:
            return {"error": "Missing required parameters"}
        
        try:
            # Decode base64 audio and save to temporary file
            voice_data = base64.b64decode(voice_base64)
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                temp_file.write(voice_data)
                voice_path = temp_file.name
            
            output_data = run_inference(
                text=text,
                voice_path=voice_path,
                is_url=False,
                chunk_size=chunk_size,
                max_parallel_chunks=max_parallel_chunks
            )
            
            # Cleanup temporary file
            os.unlink(voice_path)
            
            return {"audio": base64.b64encode(output_data).decode()}
        except Exception as e:
            return {"error": str(e)}

    return app
