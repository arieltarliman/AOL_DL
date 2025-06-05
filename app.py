import streamlit as st
from PIL import Image
from pathlib import Path
import torch
import os
import sys
import requests # Added import

# --- Configuration & Path Setup ---
# Assuming app.py is in the 'deploy' directory
BASE_DIR = Path(__file__).resolve().parent
PYTORCH_SD_DIR = BASE_DIR / "pytorch-stable-diffusion-main"
DATA_DIR = PYTORCH_SD_DIR / "data"
SD_CODE_DIR = PYTORCH_SD_DIR / "sd"

# Add the 'sd' directory to Python's path to allow direct imports
# Corrected the str() call here
if str(SD_CODE_DIR) not in sys.path:
    sys.path.append(str(SD_CODE_DIR))

# Attempt to import Stable Diffusion components
try:
    import model_loader
    import pipeline
    from transformers import CLIPTokenizer
    sd_components_available = True
except ImportError as e:
    st.error(f"Failed to import Stable Diffusion components from {SD_CODE_DIR}: {e}. Ensure the directory is correct and all dependencies are installed.")
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

# --- Function to Download Large Model File ---
# !!! REPLACE "YOUR_DIRECT_DOWNLOAD_LINK_HERE" WITH THE ACTUAL URL !!!
MODEL_URL = "https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5/resolve/main/v1-5-pruned-emaonly.ckpt"
STABLE_DIFFUSION_MODEL_FILENAME = "v1-5-pruned-emaonly.ckpt"
# Use DATA_DIR for consistency
MODEL_PATH_FOR_DOWNLOAD = DATA_DIR / STABLE_DIFFUSION_MODEL_FILENAME

@st.cache_resource # Caching the download process is good
def download_sd_model_if_needed():
    """Downloads the Stable Diffusion model if it doesn't already exist."""
    if not MODEL_PATH_FOR_DOWNLOAD.exists():
        st.info(f"Downloading Stable Diffusion checkpoint ({STABLE_DIFFUSION_MODEL_FILENAME})...")
        st.write(f"Download URL: {MODEL_URL} (This may take a while for large files)")
        
        # Ensure parent directory exists
        MODEL_PATH_FOR_DOWNLOAD.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with requests.get(MODEL_URL, stream=True) as r:
                r.raise_for_status() # Raises an HTTPError for bad responses (4XX or 5XX)
                total_size = int(r.headers.get('content-length', 0))
                
                # Simple progress bar
                progress_bar = st.progress(0)
                bytes_downloaded = 0

                with open(MODEL_PATH_FOR_DOWNLOAD, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192 * 16): # Increased chunk size
                        if chunk: # filter out keep-alive new chunks
                            f.write(chunk)
                            bytes_downloaded += len(chunk)
                            if total_size > 0:
                                progress_bar.progress(min(bytes_downloaded / total_size, 1.0))
                if total_size == 0: # If content-length was not provided
                    progress_bar.progress(1.0) # Mark as complete

            st.success(f"✅ Model '{STABLE_DIFFUSION_MODEL_FILENAME}' downloaded successfully to {MODEL_PATH_FOR_DOWNLOAD}")
        except requests.exceptions.RequestException as e:
            st.error(f"❌ Error downloading model: {e}")
            # Attempt to remove partially downloaded file
            if MODEL_PATH_FOR_DOWNLOAD.exists():
                try:
                    os.remove(MODEL_PATH_FOR_DOWNLOAD)
                    st.info("Removed partially downloaded model file.")
                except OSError as ose:
                    st.warning(f"Could not remove partially downloaded file: {ose}")
            return False # Indicate download failure
        except Exception as e:
            st.error(f"❌ An unexpected error occurred during download: {e}")
            return False
    else:
        st.info(f"Model '{STABLE_DIFFUSION_MODEL_FILENAME}' already exists at {MODEL_PATH_FOR_DOWNLOAD}.")
    return True # Indicate model is available (either pre-existing or downloaded)

# --- Stable Diffusion Model Loading (Cached) ---
@st.cache_resource
def load_sd_models_cached():
    st.write(f"Attempting to load Stable Diffusion models on device: {DEVICE}")

    # ** Call the download function here **
    model_available = download_sd_model_if_needed()
    if not model_available:
        st.error("Stable Diffusion model checkpoint is not available. Cannot proceed.")
        return None, None

    if not sd_components_available or not CLIPTokenizer or not model_loader:
        st.error("Stable Diffusion Python components are not loaded. Cannot proceed.")
        return None, None

    try:
        tokenizer_path = DATA_DIR / "vocab.json"
        merges_path = DATA_DIR / "merges.txt"
        # model_file_path now refers to MODEL_PATH_FOR_DOWNLOAD which is checked by download_sd_model_if_needed
        model_file_path = MODEL_PATH_FOR_DOWNLOAD # This is DATA_DIR / STABLE_DIFFUSION_MODEL_FILENAME

        if not tokenizer_path.exists():
            st.error(f"Tokenizer vocab file not found: {tokenizer_path}")
            return None, None
        if not merges_path.exists():
            st.error(f"Tokenizer merges file not found: {merges_path}")
            return None, None
        
        # The download function already checks if model_file_path exists and tries to download it.
        # We add an explicit check here again just in case download_sd_model_if_needed had an issue but didn't propagate error correctly
        # or if it was skipped.
        if not model_file_path.exists():
            st.error(f"Stable Diffusion model file not found after download attempt: {model_file_path}")
            st.error("Please ensure the MODEL_URL is correct and the file can be downloaded.")
            return None, None

        tokenizer = CLIPTokenizer(str(tokenizer_path), merges_file=str(merges_path))
        models = model_loader.preload_models_from_standard_weights(str(model_file_path), DEVICE)
        st.success("✅ Stable Diffusion models loaded successfully.")
        return tokenizer, models
    except Exception as e:
        st.error(f"❌ Error loading Stable Diffusion models: {e}")
        st.error("Please ensure the model files are in the 'pytorch-stable-diffusion-main/data' directory (downloaded or placed manually) and all dependencies are correctly installed.")
        import traceback
        st.error(traceback.format_exc())
        return None, None

# --- GAN Model Placeholder Functions ---
def load_gan_model_placeholder(model_path):
    """Placeholder for loading GAN models."""
    if not model_path.exists():
        st.error(f"GAN model not found: {model_path}")
        return None
    st.info(f"GAN model '{model_path.name}' would be loaded here (this is a placeholder).")
    return model_path.name

def generate_with_gan_placeholder(model_name_placeholder):
    """Placeholder for GAN image generation."""
    st.write(f"Generating image with {model_name_placeholder} (placeholder)...")
    try:
        img = Image.new('RGB', (256, 256), color = ('#ADD8E6' if 'Vanilla' in model_name_placeholder else '#FFB6C1'))
        st.image(img, caption=f"Generated by {model_name_placeholder} (Placeholder)", use_column_width=True)
    except Exception as e:
        st.error(f"Error creating placeholder image: {e}")


# --- Streamlit App UI ---
# (The rest of your UI code remains the same)
# ... (your UI code from st.set_page_config onwards) ...
st.set_page_config(layout="wide", page_title="AI Image Generation Hub")
st.title("🎨 AI Image Generation Hub")
st.markdown("Welcome! Select a model from the sidebar to start generating images.")

# Sidebar for Model Selection and Global Settings
st.sidebar.header("⚙️ Model Configuration")
model_choice = st.sidebar.selectbox(
    "Choose a Model:",
    ("--- Select ---", "Vanilla Generator", "DCGAN Generator", "Stable Diffusion")
)
st.sidebar.markdown("---")

# --- Main Content Area ---
if model_choice == "--- Select ---":
    st.info("👈 Select a model from the sidebar to get started!")
    st.markdown("""
        ### Available Models:
        - **Vanilla Generator**: A basic GAN model.
        - **DCGAN Generator**: A Deep Convolutional GAN model.
        - **Stable Diffusion**: A powerful latent diffusion model for text-to-image and image-to-image generation.
    """)

elif model_choice in ["Vanilla Generator", "DCGAN Generator"]:
    st.header(f"🖼️ {model_choice}")
    model_file_name = "Vanilla_Generator_epoch_0573.h5" if model_choice == "Vanilla Generator" else "DCGAN_generator_epoch_0210.h5"
    model_path = BASE_DIR / model_file_name

    st.info(f"This section is for the **{model_choice}**. Actual model loading and generation for this GAN type needs to be implemented if you have the Python code for it (e.g., using TensorFlow/Keras for `.h5` files).")

    if st.button(f"Generate Image with {model_choice}", key=f"generate_{model_choice.replace(' ', '_')}"):
        if model_path.exists():
            loaded_model_placeholder = load_gan_model_placeholder(model_path)
            if loaded_model_placeholder:
                generate_with_gan_placeholder(loaded_model_placeholder)
        else:
            st.error(f"Model file not found: {model_path}. Please ensure it's in the 'deploy' directory.")

elif model_choice == "Stable Diffusion":
    st.header("✨ Stable Diffusion Image Generation")

    if not sd_components_available:
        st.error("Stable Diffusion functionality is currently unavailable due to import errors. Please check the console for details.")
    else:
        # Load models (will be cached after first run, download will also be cached)
        tokenizer, sd_models = load_sd_models_cached() # This now handles the download

        if tokenizer and sd_models:
            st.sidebar.subheader("Stable Diffusion Settings")
            sd_input_type = st.sidebar.radio(
                "Select Input Type:",
                ("Text-to-Image", "Image-to-Image", "Text-and-Image"),
                key="sd_input_type",
                help="Choose how you want to generate the image."
            )
            st.sidebar.markdown("---")

            # --- Stable Diffusion Inputs ---
            st.subheader("✏️ Prompts & Guidance")
            col1, col2 = st.columns(2)
            with col1:
                prompt = st.text_area("Prompt:", "A photorealistic cat astronaut exploring a neon-lit alien jungle, cinematic lighting, 8k", height=100, help="Describe what you want to see.")
            with col2:
                uncond_prompt = st.text_area("Negative Prompt (Optional):", "blurry, low quality, watermark, text, signature, ugly, deformed", height=100, help="Describe what you DON'T want to see.")

            cfg_scale = st.slider("CFG Scale (Guidance Strength):", min_value=1.0, max_value=20.0, value=7.5, step=0.5, help="How strictly the model should follow the prompt. Higher values mean stricter adherence.")

            input_image_pil = None
            strength = 0.8

            if sd_input_type in ["Image-to-Image", "Text-and-Image"]:
                st.subheader("🖼️ Input Image (for Image-to-Image)")
                uploaded_file = st.file_uploader("Upload an Image:", type=["png", "jpg", "jpeg"], key="sd_image_upload")
                if uploaded_file:
                    try:
                        input_image_pil = Image.open(uploaded_file).convert("RGB")
                        st.image(input_image_pil, caption="Uploaded Image", width=300)
                    except Exception as e:
                        st.error(f"Error opening image: {e}")
                        input_image_pil = None

                strength = st.slider("Strength (for Image-to-Image):", min_value=0.0, max_value=1.0, value=0.8, step=0.05,
                                     help="Controls how much the input image influences the output. 0.0: output is very similar to input; 1.0: input image is almost completely ignored (becomes like text-to-image).")

            st.sidebar.markdown("---")
            st.sidebar.subheader("Sampling Parameters")
            sampler_options = ["ddpm", "ddim"]
            sampler = st.sidebar.selectbox("Sampler:", sampler_options, index=0, help="Algorithm used for denoising.")
            num_inference_steps = st.sidebar.slider("Inference Steps:", min_value=10, max_value=150, value=50, step=1, help="Number of denoising steps. More steps can improve quality but take longer.")
            seed = st.sidebar.number_input("Seed:", value=42, step=1, help="Controls randomness. Same seed + same parameters = same image.")

            st.markdown("---")
            if st.button("🚀 Generate with Stable Diffusion", key="generate_sd", type="primary"):
                if not prompt.strip() and sd_input_type in ["Text-to-Image", "Text-and-Image"]:
                    st.error("🚨 Prompt is required for Text-to-Image and Text-and-Image modes.")
                elif sd_input_type in ["Image-to-Image", "Text-and-Image"] and not input_image_pil:
                    st.warning("⚠️ No image uploaded for Image-to-Image or Text-and-Image mode. The model will primarily use the text prompt (if provided).")
                    current_input_image_for_pipeline = None
                elif sd_input_type == "Image-to-Image" and not input_image_pil:
                     st.error("🚨 Please upload an image for Image-to-Image mode.")
                else:
                    current_input_image_for_pipeline = input_image_pil

                    with st.spinner("⏳ Generating image with Stable Diffusion... This might take a moment!"):
                        try:
                            effective_prompt = prompt
                            if sd_input_type == "Image-to-Image" and not prompt.strip():
                                effective_prompt = " "

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
                            if output_image_array is not None:
                                final_image = Image.fromarray(output_image_array)
                                st.image(final_image, caption=f"Generated Image (Seed: {seed})", use_column_width=True)
                            else:
                                st.error("Generation failed to produce an image array.")
                        except Exception as e:
                            st.error(f"❌ An error occurred during Stable Diffusion generation: {e}")
                            import traceback
                            st.error(f"Traceback: {traceback.format_exc()}")
        else:
            # This part is reached if load_sd_models_cached returns (None, None)
            # which can happen if download fails or other setup issues occur.
            st.error("Stable Diffusion models could not be loaded or components are missing. Please check error messages above.")

st.sidebar.markdown("---")
st.sidebar.caption(f"Running on: **{DEVICE.upper()}**")