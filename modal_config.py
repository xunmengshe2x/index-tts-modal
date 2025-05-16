from modal import Image, Secret, NetworkFileSystem

# Create network storage for persisting model files
MODEL_VOLUME = NetworkFileSystem.persisted("indextts-model-cache")

# Base image configuration
def configure_image():
    # Start with CUDA-enabled base image since GPU is required
    return (
        Image.debian_slim(python_version="3.10")
        .apt_install([
            "ffmpeg",  # Required for audio processing
            "build-essential",  # For compiling extensions
            "git",
            "wget",
        ])
        # Install Python dependencies from requirements.txt
        .pip_install([
            "torch>=2.1.2",
            "torchaudio",
            "transformers==4.36.2",
            "accelerate==0.25.0",
            "tokenizers==0.15.0",
            "einops==0.8.1",
            "matplotlib==3.8.2",
            "omegaconf",
            "sentencepiece",
            "librosa",
            "numpy==1.26.2",
            "WeTextProcessing==1.0.3",
            "cn2an==0.5.22",
            "ffmpeg-python==0.2.0",
            "Cython==3.0.7",
            "g2p-en==2.1.0",
            "jieba==0.42.1",
            "numba==0.58.1",
            "pandas==2.1.3",
            "opencv-python==4.9.0.80",
            "vocos==0.1.0",
            "tensorboard==2.9.1",
            "gradio",
            "wget",
        ])
        # Install the package itself
        .pip_install("git+https://github.com/index-tts/index-tts.git")
    )

# Required model files based on repository analysis
MODEL_FILES = {
    "bigvgan_discriminator.pth": "https://huggingface.co/IndexTeam/IndexTTS-1.5/resolve/main/bigvgan_discriminator.pth",
    "bigvgan_generator.pth": "https://huggingface.co/IndexTeam/IndexTTS-1.5/resolve/main/bigvgan_generator.pth",
    "bpe.model": "https://huggingface.co/IndexTeam/IndexTTS-1.5/resolve/main/bpe.model",
    "dvae.pth": "https://huggingface.co/IndexTeam/IndexTTS-1.5/resolve/main/dvae.pth",
    "gpt.pth": "https://huggingface.co/IndexTeam/IndexTTS-1.5/resolve/main/gpt.pth",
    "unigram_12000.vocab": "https://huggingface.co/IndexTeam/IndexTTS-1.5/resolve/main/unigram_12000.vocab",
    "config.yaml": "https://huggingface.co/IndexTeam/IndexTTS-1.5/resolve/main/config.yaml"
}

# GPU configuration
GPU_CONFIG = "A10G"  # Based on repository requirements

# Paths configuration
CHECKPOINT_DIR = "/model/checkpoints"
