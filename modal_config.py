from modal import Image, Secret, gpu

# Base image configuration
def get_image():
    return (
        Image.debian_slim(python_version="3.10")
        .apt_install([
            "ffmpeg",  # Required per README.md
            "ninja-build",  # Required for CUDA kernel compilation
            "build-essential",  # Required for compilation
        ])
        .pip_install([
            # Core dependencies from requirements.txt
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
            "gradio",
            "tqdm",
            "WeTextProcessing",
            "torch",
            "torchaudio",
        ])
    )

# GPU configuration - A10G required based on repository CUDA requirements
GPU_CONFIG = gpu.A10G()

# Model URLs extracted from repository
MODEL_URLS = {
    "config": "https://huggingface.co/IndexTeam/IndexTTS-1.5/resolve/main/config.yaml",
    "bigvgan_discriminator": "https://huggingface.co/IndexTeam/IndexTTS-1.5/resolve/main/bigvgan_discriminator.pth",
    "bigvgan_generator": "https://huggingface.co/IndexTeam/IndexTTS-1.5/resolve/main/bigvgan_generator.pth",
    "bpe_model": "https://huggingface.co/IndexTeam/IndexTTS-1.5/resolve/main/bpe.model",
    "dvae": "https://huggingface.co/IndexTeam/IndexTTS-1.5/resolve/main/dvae.pth",
    "gpt": "https://huggingface.co/IndexTeam/IndexTTS-1.5/resolve/main/gpt.pth",
    "vocab": "https://huggingface.co/IndexTeam/IndexTTS-1.5/resolve/main/unigram_12000.vocab"
}

# Required directories
DIRS = {
    "checkpoints": "/content/checkpoints",
    "test_data": "/content/test_data",
}