import os
import modal
import torch
import pickle
import numpy as np
from PIL import Image
from io import BytesIO
from datetime import datetime
from omegaconf import OmegaConf
from fastapi import FastAPI, File, UploadFile, Form, Request

# Define a custom image with all dependencies
image = modal.Image.debian_slim().pip_install(
    "torch==1.12.1+cu113",
    "torchvision==0.13.1+cu113",
    "torchaudio==0.12.1",
    extra_index_url="https://download.pytorch.org/whl/cu113",
).pip_install(
    "numpy>=1.21.0",  # Updated to be compatible with Python 3.10
    "Pillow==9.5.0",
    "torchmetrics==0.5",
    "albumentations==0.4.3",
    "pytorch-lightning==1.4.2",
    "opencv-python",
    "omegaconf",
    "tensorboard",
    "editdistance",
    "einops",
    "tqdm",
    "fastapi[standard]",
    "pydantic>=2.0.0",
    "typing-extensions"
)

# Add CUDA support and other tools
image = image.apt_install("ffmpeg", "wget", "git")

# Create a Modal volume to store model files
volume = modal.Volume.from_name("rs-ste-models", create_if_missing=True)

# Create a Modal app
app = modal.App("rs-ste", image=image)

@app.function(
    gpu="A10G",
    timeout=600,
    volumes={"/model": volume}
)
def download_model():
    """Download RS-STE model files to the volume."""
    import os
    import subprocess
    
    # Create model directory if it doesn't exist
    os.makedirs("/model", exist_ok=True)
    
    # Check if model is already downloaded
    if os.path.exists("/model/rsste-finetune.ckpt"):
        print("Model already downloaded.")
        return
    
    # Clone the repository to get config files
    if not os.path.exists("/model/rs-ste"):
        print("Cloning repository for config files...")
        subprocess.run(
            "git clone https://github.com/xunmengshe2x/rs-ste.git",
            shell=True,
            check=True,
            cwd="/model"
        )
    
    # Download the model checkpoint from Hugging Face
    model_url = "https://huggingface.co/v4mmko/RS-STE/resolve/main/rsste-finetune.ckpt"
    print(f"Downloading checkpoint from {model_url}...")
    subprocess.run(f"wget {model_url} -O /model/rsste-finetune.ckpt", shell=True, check=True)
    
    print("Model and configs downloaded successfully.")
    return True

def crop_and_resize(x, size, ori_size, k):
    img = x.detach().cpu()
    img = torch.clamp(img, -1., 1.)
    img = (img + 1.0) / 2.0
    img = img.permute(0, 2, 3, 1).numpy()[k]
    img = (img * 255).astype(np.uint8)
    return img

# Root endpoint
@app.function(
    gpu="A10G",
    timeout=600,
    volumes={"/model": volume}
)
@modal.fastapi_endpoint(method="GET")
def read_root():
    return {"message": "RS-STE API is running. Use /inference_with_file endpoint for text editing in images."}

# Inference endpoint
@app.function(
    gpu="A10G",
    timeout=600,
    volumes={"/model": volume}
)
@modal.fastapi_endpoint(method="POST")
async def inference_with_file(request: Request):
    """Web endpoint for RS-STE inference with direct file upload."""
    import base64
    import os
    import sys
    import logging
    
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Parse the request body
    data = await request.json()
    target_text = data.get("target_text")
    image_base64 = data.get("image_base64")
    
    if not target_text or not image_base64:
        return {"error": "Missing required parameters: target_text and image_base64"}
    
    # Ensure model is downloaded
    download_model.remote()
    
    # Create directories
    model_dir = "/model"
    repo_dir = os.path.join(model_dir, "rs-ste")
    input_dir = os.path.join(model_dir, "inputs")
    output_dir = os.path.join(model_dir, "outputs")
    annotation_dir = os.path.join(model_dir, "data/annotation")
    
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(annotation_dir, exist_ok=True)
    
    # Save the input image
    input_path = os.path.join(input_dir, "input_image.png")
    with open(input_path, "wb") as f:
        f.write(base64.b64decode(image_base64))
    
    # Create annotation file
    annotation_data = {
        "image1_paths": [input_path],
        "image2_paths": [],
        "image1_rec": [""],  # Will be recognized by the model
        "image2_rec": [target_text]  # Target text to edit into the image
    }
    
    annotation_file = os.path.join(annotation_dir, "temp_inference.pkl")
    with open(annotation_file, "wb") as f:
        pickle.dump(annotation_data, f)
    
    # Set up output path
    output_path = os.path.join(output_dir, f"result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
    
    try:
        # Add the repository to the Python path
        sys.path.append(repo_dir)
        
        # Import necessary modules
        from main import instantiate_from_config, get_obj_from_str
        
        # Load model configurations from the cloned repository
        vqgan_config = os.path.join(repo_dir, "configs/vqgan_decoder.yaml")
        transformer_config = os.path.join(repo_dir, "configs/synth_pair.yaml")
        checkpoint_path = os.path.join(model_dir, "rsste-finetune.ckpt")
        
        decoder_config = OmegaConf.load(vqgan_config)
        config = OmegaConf.load(transformer_config)
        config.model.params.decoder_config = decoder_config.model
        config.model.params.ckpt_path = checkpoint_path
        
        # Initialize model
        model = instantiate_from_config(config.model).to('cuda')
        model.eval()
        
        # Create dataset for the single image
        from data.dataset import InferenceDataset
        from torch.utils.data import DataLoader
        
        dataset = InferenceDataset(config.data.params.validation.params.size, annotation_file)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
        
        # Run inference
        with torch.no_grad():
            for batch in dataloader:
                img1 = batch["image1"].to('cuda')
                img1 = img1.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format)
                if img1.dtype == torch.double:
                    img1 = img1.float()
                
                img1_quant_latent_ = model.conv(img1).flatten(2).permute(0, 2, 1)
                rec2_indices, _ = model.str_converter.encode(batch["rec2"])
                rec2_indices = rec2_indices.to(img1_quant_latent_.device)
                rec2_embd = model.str_embd(rec2_indices)
                
                rec1_mask = model.masked_rec_embd.weight[0][None, None, :].expand(rec2_indices.shape[0], rec2_indices.shape[1], -1)
                img2_mask = model.masked_img_imbd.weight[0][None, None, :].expand(img1_quant_latent_.shape[0], img1_quant_latent_.shape[1], -1)
                
                inputs = torch.cat([rec2_embd, img1_quant_latent_, rec1_mask, img2_mask], dim=1)
                embeddings, logits = model.transformer(inputs)
                
                rec1_indices = torch.topk(logits[:,288:320,:], k=1, dim=-1)[1].view(batch["image1"].shape[0],-1)
                pred_rec = model.str_converter.decode(rec1_indices)
                
                img2_rec_quant_latent = model.conv_o(embeddings[:,320:576,:].permute(0, 2, 1).contiguous().view(embeddings.shape[0], -1, 8, 32))
                img2_rec = model.decoder.decode(img2_rec_quant_latent)
                
                edited_img = crop_and_resize(img2_rec, batch["image1_size"], batch['ori_size'], 0)
                Image.fromarray(edited_img).save(output_path)
        
        # Read the result image
        with open(output_path, "rb") as f:
            result_image_bytes = f.read()
        
        # Encode the result image as base64
        result_base64 = base64.b64encode(result_image_bytes).decode("utf-8")
        
        return {
            "original_text": pred_rec[0],  # Return the recognized text from the original image
            "edited_text": target_text,
            "result_image_base64": result_base64
        }
        
    except Exception as e:
        logger.error(f"Error during inference: {str(e)}")
        return {"error": f"Inference failed: {str(e)}"}
    
    finally:
        # Clean up temporary files
        if os.path.exists(input_path):
            os.remove(input_path)
        if os.path.exists(annotation_file):
            os.remove(annotation_file)

# Health check endpoint
@app.function()
@modal.fastapi_endpoint(method="GET")
def health():
    """Health check endpoint."""
    return {"status": "ok", "service": "rs-ste"}
