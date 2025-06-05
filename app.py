import streamlit as st
from PIL import Image
from pathlib import Path
import torch
import os
import sys
import requests
import numpy as np
import tensorflow as tf

# Set page config as the FIRST Streamlit command
st.set_page_config(layout="wide", page_title="AI Image Generation Hub")

# --- ADD THIS LINE FOR DEBUGGING ---
st.write(f"TensorFlow Version: {tf.__version__}")

# --- Configuration & Path Setup ---
BASE_DIR = Path(__file__).resolve().parent
PYTORCH_SD_DIR = BASE_DIR / "pytorch-stable-diffusion-main"
DATA_DIR = PYTORCH_SD_DIR / "data"
SD_CODE_DIR = PYTORCH_SD_DIR / "sd"

# Add the 'sd' directory to Python's path
if str(SD_CODE_DIR) not in sys.path:
    sys.path.append(str(SD_CODE_DIR))

# Attempt to import Stable Diffusion components
try:
    import model_loader
    import pipeline
    from transformers import CLIPTokenizer
    sd_components_available = True
except ImportError as e:
    st.error(f"Failed to import Stable Diffusion components: {e}")
    sd_components_available = False
    class DummyPipeline:
        def generate(self, *args, **kwargs):
            raise NotImplementedError("Stable Diffusion pipeline is not available.")
    pipeline = DummyPipeline()
    model_loader = None
    CLIPTokenizer = None

# --- Model Configuration ---
DEVICE = "cpu"
ALLOW_CUDA = False
ALLOW_MPS = False

if torch.cuda.is_available() and ALLOW_CUDA:
    DEVICE = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available() and ALLOW_MPS:
    DEVICE = "mps"

# --- Function to Download Stable Diffusion Model ---
MODEL_URL = "https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5/resolve/main/v1-5-pruned-emaonly.ckpt"
STABLE_DIFFUSION_MODEL_FILENAME = "v1-5-pruned-emaonly.ckpt"
MODEL_PATH_FOR_DOWNLOAD = DATA_DIR / STABLE_DIFFUSION_MODEL_FILENAME

@st.cache_resource
def download_sd_model_if_needed():
    """Downloads the Stable Diffusion model if it doesn't already exist."""
    if not MODEL_PATH_FOR_DOWNLOAD.exists():
        st.info(f"Downloading Stable Diffusion checkpoint...")
        MODEL_PATH_FOR_DOWNLOAD.parent.mkdir(parents=True, exist_ok=True)
        try:
            with requests.get(MODEL_URL, stream=True) as r:
                r.raise_for_status()
                total_size = int(r.headers.get('content-length', 0))
                progress_bar = st.progress(0, text="Downloading Stable Diffusion Model (4.3 GB)...")
                bytes_downloaded = 0
                with open(MODEL_PATH_FOR_DOWNLOAD, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192 * 16):
                        if chunk:
                            f.write(chunk)
                            bytes_downloaded += len(chunk)
                            if total_size > 0:
                                progress_bar.progress(min(bytes_downloaded / total_size, 1.0))
                progress_bar.progress(1.0)
            st.success("✅ Model downloaded successfully.")
        except Exception as e:
            st.error(f"❌ Error downloading model: {e}")
            if MODEL_PATH_FOR_DOWNLOAD.exists():
                os.remove(MODEL_PATH_FOR_DOWNLOAD)
            return False
    return True

# --- Stable Diffusion Model Loading ---
@st.cache_resource
def load_sd_models_cached():
    model_available = download_sd_model_if_needed()
    if not model_available:
        st.error("Stable Diffusion model checkpoint not available. Cannot proceed.")
        return None, None
    if not sd_components_available:
        st.error("Stable Diffusion Python components are not loaded. Cannot proceed.")
        return None, None
    try:
        tokenizer_path = DATA_DIR / "vocab.json"
        merges_path = DATA_DIR / "merges.txt"
        model_file_path = MODEL_PATH_FOR_DOWNLOAD
        tokenizer = CLIPTokenizer(str(tokenizer_path), merges_file=str(merges_path))
        models = model_loader.preload_models_from_standard_weights(str(model_file_path), DEVICE)
        st.success("✅ Stable Diffusion models loaded successfully.")
        return tokenizer, models
    except Exception as e:
        st.error(f"❌ Error loading Stable Diffusion models: {e}")
        return None, None

# --- Real GAN Model Functions ---
@st.cache_resource
def load_gan_model(model_path):
    """Loads the Keras/TensorFlow model from the specified .h5 file."""
    try:
        model = tf.keras.models.load_model(str(model_path))
        st.success(f"GAN model '{model_path.name}' loaded successfully.")
        return model
    except Exception as e:
        st.error(f"Error loading GAN model: {e}")
        return None

def generate_with_gan(model):
    """Generates an image using the loaded GAN model."""
    try:
        # Most GANs take a 100-dimensional noise vector as input.
        # If your model was trained with a different size, change 100 to that number.
        latent_dim = 100
        noise = np.random.randn(1, latent_dim)

        with st.spinner("Generating image..."):
            generated_image_array = model.predict(noise)
            # Post-process: remove batch dimension
            generated_image_array = generated_image_array[0]
            # Post-process: rescale pixel values from [-1, 1] to [0, 255]
            # If your GAN's last layer is 'sigmoid', use (generated_image_array * 255)
            image_array_processed = (generated_image_array * 127.5 + 127.5).astype(np.uint8)
            # Convert to displayable image
            image = Image.fromarray(image_array_processed)
            st.image(image, caption="Generated by your GAN Model", use_container_width=True)
    except Exception as e:
        st.error(f"Error during GAN image generation: {e}")
        import traceback
        st.error(traceback.format_exc())

# --- Streamlit App UI ---
st.title("🎨 AI Image Generation Hub")
st.markdown("Welcome! Select a model from the sidebar to start generating images.")

# Sidebar for Model Selection
st.sidebar.header("⚙️ Model Configuration")
model_choice = st.sidebar.selectbox(
    "Choose a Model:",
    ("--- Select ---", "Vanilla Generator", "DCGAN Generator", "Stable Diffusion")
)
st.sidebar.markdown("---")

# Main Content Area
if model_choice == "--- Select ---":
    st.info("👈 Select a model from the sidebar to get started!")

elif model_choice in ["Vanilla Generator", "DCGAN Generator"]:
    st.header(f"🖼️ {model_choice}")
    model_file_name = "Vanilla_Generator_epoch_0573.h5" if model_choice == "Vanilla Generator" else "DCGAN_generator_epoch_0210.h5"
    model_path = BASE_DIR / model_file_name

    # Load the selected GAN model (this will be cached)
    gan_model = load_gan_model(model_path)

    if gan_model:
        # Show the button only if the model was loaded successfully
        if st.button(f"Generate Image with {model_choice}", key=f"generate_{model_choice.replace(' ', '_')}"):
            generate_with_gan(gan_model)
    else:
        st.error(f"Cannot proceed because the model at {model_path} could not be loaded.")

elif model_choice == "Stable Diffusion":
    st.header("✨ Stable Diffusion Image Generation")
    
    # Load models only when the SD option is selected
    tokenizer, sd_models = load_sd_models_cached()
    
    if tokenizer and sd_models:
        st.sidebar.subheader("Stable Diffusion Settings")
        sd_input_type = st.sidebar.radio(
            "Select Input Type:",
            ("Text-to-Image", "Image-to-Image", "Text-and-Image"),
            key="sd_input_type"
        )
        st.sidebar.markdown("---")

        st.subheader("✏️ Prompts & Guidance")
        col1, col2 = st.columns(2)
        with col1:
            prompt = st.text_area("Prompt:", "A photorealistic cat astronaut exploring a neon-lit alien jungle, cinematic lighting, 8k", height=100)
        with col2:
            uncond_prompt = st.text_area("Negative Prompt (Optional):", "blurry, low quality, watermark, text, signature, ugly, deformed", height=100)

        cfg_scale = st.slider("CFG Scale (Guidance Strength):", 1.0, 20.0, 7.5, 0.5)

        input_image_pil = None
        strength = 0.8

        if sd_input_type in ["Image-to-Image", "Text-and-Image"]:
            st.subheader("🖼️ Input Image (for Image-to-Image)")
            uploaded_file = st.file_uploader("Upload an Image:", type=["png", "jpg", "jpeg"], key="sd_image_upload")
            if uploaded_file:
                input_image_pil = Image.open(uploaded_file).convert("RGB")
                st.image(input_image_pil, caption="Uploaded Image", width=300)
            strength = st.slider("Strength:", 0.0, 1.0, 0.8, 0.05)

        st.sidebar.markdown("---")
        st.sidebar.subheader("Sampling Parameters")
        sampler = st.sidebar.selectbox("Sampler:", ["ddpm", "ddim"], index=0)
        num_inference_steps = st.sidebar.slider("Inference Steps:", 10, 150, 50, 1)
        seed = st.sidebar.number_input("Seed:", value=42, step=1)

        st.markdown("---")
        if st.button("🚀 Generate with Stable Diffusion", key="generate_sd", type="primary"):
            effective_prompt = prompt if prompt.strip() else " " if sd_input_type == "Image-to-Image" else ""
            if not effective_prompt:
                st.error("🚨 Prompt is required.")
            elif sd_input_type in ["Image-to-Image", "Text-and-Image"] and not input_image_pil:
                st.warning("⚠️ No image uploaded. Proceeding as Text-to-Image.")
                current_input_image_for_pipeline = None
            else:
                current_input_image_for_pipeline = input_image_pil
            
                with st.spinner("⏳ Generating image... This might take a moment!"):
                    try:
                        output_image_array = pipeline.generate(
                            prompt=effective_prompt,
                            uncond_prompt=uncond_prompt,
                            input_image=current_input_image_for_pipeline,
                            strength=strength if current_input_image_for_pipeline else 1.0,
                            do_cfg=True,
                            cfg_scale=cfg_scale,
                            sampler_name=sampler,
                            n_inference_steps=num_inference_steps,
                            seed=seed,
                            models=sd_models,
                            device=DEVICE,
                            idle_device="cpu",
                            tokenizer=tokenizer,
                        )
                        final_image = Image.fromarray(output_image_array)
                        st.image(final_image, caption=f"Generated Image (Seed: {seed})", use_column_width=True)
                    except Exception as e:
                        st.error(f"❌ An error occurred during generation: {e}")
                        st.error(traceback.format_exc())

st.sidebar.markdown("---")
st.sidebar.caption(f"Running on: **{DEVICE.upper()}**")