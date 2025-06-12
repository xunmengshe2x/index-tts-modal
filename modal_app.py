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
from typing import List, Optional, Dict, Any
import numpy as np
import torch
import torchaudio
import logging
import subprocess
import asyncio
import threading

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
    "fastapi[standard]",  # Required for web endpoints
    "pydantic>=2.0.0",    # Explicitly add Pydantic
    "typing-extensions"   # Often needed with Pydantic
)

# Add CUDA support, ffmpeg, wget, and git
image = image.apt_install("ffmpeg", "wget", "git")   

# Create a Modal volume to store model files
volume = modal.Volume.from_name("index-tts-models", create_if_missing=True)

# Create a Modal app
app = modal.App("index-tts-inference", image=image)

class ModelPool:
    def __init__(self, num_instances: int = 8):
        self.num_instances = num_instances
        self.models: Dict[int, Any] = {}
        self.locks = [threading.Lock() for _ in range(num_instances)]
        self.initialization_lock = threading.Lock()
        self.streams = {}
        
    def initialize_model(self, model_id: int):
        """Initialize a single model instance if it doesn't exist"""
        if model_id not in self.models:
            with self.initialization_lock:
                if model_id not in self.models:  # Double-check pattern
                    # Create a new CUDA stream for this model
                    stream = torch.cuda.Stream()
                    self.streams[model_id] = stream
                    
                    # Initialize model with its own stream
                    with torch.cuda.stream(stream):
                        model = indextts_infer.IndexTTS(
                            cfg_path="/checkpoints/config.yaml",
                            model_dir="/checkpoints",
                            is_fp16=True  # Use FP16 for memory efficiency
                        )
                        self.models[model_id] = model
                        
    def get_model(self) -> Optional[tuple[int, Any]]:
        """Get next available model instance"""
        for model_id in range(self.num_instances):
            self.initialize_model(model_id)
            if self.locks[model_id].acquire(blocking=False):
                return model_id, self.models[model_id]
        return None, None

    def release_model(self, model_id: Optional[int]):
        """Release a model instance back to the pool"""
        if model_id is not None and model_id < len(self.locks):
            self.locks[model_id].release()

# Global model pool
model_pool = None

[Previous functions remain unchanged: split_into_chunks, concatenate_audio_files, 
download_models, download_repository, run_inference, inference_api]

@app.function(
    gpu="h100",
    timeout=900,
    volumes={"/checkpoints": volume}
)
@modal.concurrent(max_inputs=20)
@modal.fastapi_endpoint(method="POST")
async def inference_api_with_file(request: Request):
    """Web endpoint for Index-TTS inference with direct file upload."""
    global model_pool
    
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Initialize model pool if not exists
    if model_pool is None:
        model_pool = ModelPool(num_instances=8)  # Adjust based on H100 memory
    
    # Parse the request body
    data = await request.json()
    text = data.get("text")
    voice_base64 = data.get("voice_base64")
    chunk_size = data.get("chunk_size", 60)
    max_text_tokens_per_sentence = data.get("max_text_tokens_per_sentence", 300)
    sentences_bucket_max_size = data.get("sentences_bucket_max_size", 8)

    if not text or not voice_base64:
        return {"error": "Missing required parameters: text and voice_base64"}

    # Create directories and save voice prompt
    inputs_dir = "/checkpoints/inputs"
    outputs_dir = "/checkpoints/outputs"
    os.makedirs(inputs_dir, exist_ok=True)
    os.makedirs(outputs_dir, exist_ok=True)
    voice_path = os.path.join(inputs_dir, "voice_prompt.wav")

    with open(voice_path, "wb") as temp_file:
        temp_file.write(base64.b64decode(voice_base64))

    try:
        if not os.path.exists(voice_path):
            logger.error(f"Voice prompt file does not exist: {voice_path}")
            raise FileNotFoundError(f"Voice prompt file does not exist: {voice_path}")

        # Set up Python path and imports
        sys.path.append("/checkpoints/index-tts")
        os.chdir("/checkpoints")

        # Dynamic import
        module_path = "/checkpoints/index-tts/indextts/infer.py"
        spec = importlib.util.spec_from_file_location("indextts.infer", module_path)
        indextts_infer = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(indextts_infer)

        # Split text into chunks and process
        chunks = split_into_chunks(text, max_chunk_size=chunk_size)
        logger.info(f"Split text into {len(chunks)} chunks")
        
        async def process_chunk(chunk: str, chunk_idx: int):
            if not chunk or chunk.isspace():
                return None
                
            chunk_output = f"chunk_{chunk_idx}.wav"
            chunk_output_path = os.path.join(outputs_dir, chunk_output)
            
            try:
                # Get available model from pool
                model_id, model = model_pool.get_model()
                if model is None:
                    logger.error("No available model in pool")
                    return None
                    
                try:
                    chunk = chunk.strip()
                    if not chunk:
                        return None
                        
                    # Process with selected model instance
                    with torch.cuda.stream(model_pool.streams[model_id]):
                        model.infer_fast(
                            audio_prompt=voice_path,
                            text=chunk,
                            output_path=chunk_output_path,
                            max_text_tokens_per_sentence=max_text_tokens_per_sentence,
                            sentences_bucket_max_size=sentences_bucket_max_size
                        )
                    
                    if not os.path.exists(chunk_output_path):
                        logger.error(f"Output file not created for chunk {chunk_idx}")
                        return None
                        
                    with open(chunk_output_path, "rb") as f:
                        audio_data = f.read()
                    
                    return chunk_idx, audio_data
                finally:
                    model_pool.release_model(model_id)
                    
            except Exception as e:
                logger.error(f"Error processing chunk {chunk_idx}: {str(e)}")
                return None
            finally:
                if os.path.exists(chunk_output_path):
                    os.remove(chunk_output_path)

        # Process chunks concurrently
        tasks = [process_chunk(chunk, idx) for idx, chunk in enumerate(chunks)]
        results = await asyncio.gather(*tasks)
        
        # Filter out None results and add valid chunks
        audio_chunks = [result for result in results if result is not None]

        if not audio_chunks:
            return {"error": "No audio chunks were successfully processed"}

        # Sort chunks by index and concatenate
        audio_chunks.sort(key=lambda x: x[0])
        final_audio = concatenate_audio_files([chunk[1] for chunk in audio_chunks])

        # Return base64 encoded audio
        return {"audio_base64": base64.b64encode(final_audio).decode("utf-8")}
    finally:
        if os.path.exists(voice_path):
            os.remove(voice_path)

# Health check endpoint
@app.function()
@modal.fastapi_endpoint(method="GET")
async def health():
    """Health check endpoint."""
    return {"status": "ok", "service": "index-tts-inference"}
